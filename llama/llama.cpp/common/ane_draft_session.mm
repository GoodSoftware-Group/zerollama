// In-process ANE draft session for llama-server (B1).
// Compile-once at dflash init; IOSurface owned in same PID as ggml Metal backend.
// Why conv2 fallback: chained conv MIL may fail compile on device; conv1 keeps lab unblocked.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include "ane_draft_session.h"
#include "ane_iosurface_map.h"
#include "ane_bridge.h"
#include "log.h"

#include <cstdlib>
#include <cmath>
#include <vector>
#include <atomic>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <limits.h>
#include <dispatch/dispatch.h>

static mach_timebase_info_data_t g_tb;
static bool                      g_tb_init = false;

static int ane_conv_depth_cap(void) {
    const char * v = getenv("ZEROLLAMA_ANE_DRAFT_CONV_DEPTH");
    if (!v || !v[0]) {
        return 0;
    }
    char * end = NULL;
    const long d = strtol(v, &end, 10);
    if (end == v || d < 1 || d > INT_MAX) {
        return 0;
    }
    return (int) d;
}

static double ane_ticks_to_ms(uint64_t t) {
    if (!g_tb_init) {
        mach_timebase_info(&g_tb);
        g_tb_init = true;
    }
    return (double) t * g_tb.numer / g_tb.denom / 1e6;
}

static NSString * ane_gen_conv_mil(int ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        @"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"
        @"            string x_to_fp16_dtype_0 = const()[name = string(\"x_to_fp16_dtype_0\"), val = string(\"fp16\")];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> x_to_fp16 = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string(\"cast_in\")];\n"
        @"            tensor<fp16, [%d, %d, 1, 1]> W0 = const()[name = string(\"W0\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> c0 = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W0, x = x_to_fp16)[name = string(\"c0\")];\n"
        @"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        @"            tensor<fp32, [1, %d, 1, %d]> c = cast(dtype = to_fp32, x = c0)[name = string(\"cast_out\")];\n"
        @"        } -> (c);\n"
        @"}\n",
        ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];
}

static NSString * ane_gen_draft_matmul_mil(int ic, int oc, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        @"            string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        @"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        @"            bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"
        @"            tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to_fp16, x=x)[name=string(\"cast_in\")];\n"
        @"            tensor<fp16, [1,1,%d,%d]> x2 = reshape(shape=ra, x=x16)[name=string(\"x2\")];\n"
        @"            tensor<int32, [2]> pm = const()[name=string(\"pm\"), val=tensor<int32, [2]>([0,1,3,2])];\n"
        @"            tensor<fp16, [1,1,%d,%d]> x3 = transpose(perm=pm, x=x2)[name=string(\"x3\")];\n"
        @"            tensor<fp16, [%d, %d, 1, 1]> W0 = const()[name = string(\"W0\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        @"            tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        @"            tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw, x=W0)[name=string(\"W\")];\n"
        @"            tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF, transpose_y=bF, x=x3, y=W)[name=string(\"yh\")];\n"
        @"            tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n"
        @"            tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        @"            tensor<fp16, [1,%d,1,%d]> y16 = reshape(shape=ro, x=yt)[name=string(\"y16\")];\n"
        @"            tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to_fp32, x=y16)[name=string(\"cast_out\")];\n"
        @"        } -> (y);\n"
        @"}\n",
        ic, seq, ic, seq, ic, seq, seq, ic, ic, oc, ic, oc, ic, oc, seq, oc, seq, oc, oc, seq, oc, seq, oc, seq];
}

static NSData * ane_load_matmul_weight_blob(const char * path, int ic, int oc) {
    if (!path || !path[0] || ic <= 0 || oc <= 0) {
        return nil;
    }
    NSData * data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data) {
        return nil;
    }
    NSUInteger expected = 64 + 64 + (NSUInteger) ic * (NSUInteger) oc * 2;
    if ([data length] != expected) {
        return nil;
    }
    return data;
}

static BOOL ane_compile_matmul_kernel(
        ANEKernelHandle ** out_kernel,
        NSData * weight_blob,
        int ic,
        int oc,
        int seq,
        size_t in_bytes,
        size_t out_bytes) {
    if (!out_kernel || !weight_blob || ic <= 0 || oc <= 0 || seq <= 0) {
        return NO;
    }
    NSString * mil = ane_gen_draft_matmul_mil(ic, oc, seq);
    NSData * milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        return NO;
    }
    const char * weightNames[1] = { "@model_path/weights/weight.bin" };
    const uint8_t * weightDatas[1] = { (const uint8_t *) [weight_blob bytes] };
    size_t weightLens[1] = { [weight_blob length] };
    *out_kernel = ane_bridge_compile_multi_weights(
        (const char *) [milData bytes], [milData length],
        weightNames, weightDatas, weightLens, 1,
        1, &in_bytes, 1, &out_bytes);
    return *out_kernel != NULL;
}

static void ane_append_dyn_matmul(NSMutableString * m, const char * prefix,
                                  int ic, int oc, int seq, int actSpOff, int wSpOff,
                                  const char * inputVar) {
    [m appendFormat:@"        tensor<int32, [4]> %s_ba = const()[name=string(\"%s_ba\"), val=tensor<int32, [4]>([0,0,0,%d])];\n",
        prefix, prefix, actSpOff];
    [m appendFormat:@"        tensor<int32, [4]> %s_sa = const()[name=string(\"%s_sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
        prefix, prefix, ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_act = slice_by_size(x=%s,begin=%s_ba,size=%s_sa)[name=string(\"%s_act\")];\n",
        ic, seq, prefix, inputVar, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_bw = const()[name=string(\"%s_bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n",
        prefix, prefix, wSpOff];
    [m appendFormat:@"        tensor<int32, [4]> %s_sw = const()[name=string(\"%s_sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
        prefix, prefix, ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_wt = slice_by_size(x=%s,begin=%s_bw,size=%s_sw)[name=string(\"%s_wt\")];\n",
        ic, oc, prefix, inputVar, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_ra = const()[name=string(\"%s_ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n",
        prefix, prefix, ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_a2 = reshape(shape=%s_ra,x=%s_act)[name=string(\"%s_a2\")];\n",
        ic, seq, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_pm = const()[name=string(\"%s_pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n",
        prefix, prefix];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_a3 = transpose(perm=%s_pm,x=%s_a2)[name=string(\"%s_a3\")];\n",
        seq, ic, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_rw = const()[name=string(\"%s_rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n",
        prefix, prefix, ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_W = reshape(shape=%s_rw,x=%s_wt)[name=string(\"%s_W\")];\n",
        ic, oc, prefix, prefix, prefix, prefix];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_yh = matmul(transpose_x=bF,transpose_y=bF,x=%s_a3,y=%s_W)[name=string(\"%s_yh\")];\n",
        seq, oc, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_yt = transpose(perm=%s_pm,x=%s_yh)[name=string(\"%s_yt\")];\n",
        oc, seq, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_ro = const()[name=string(\"%s_ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
        prefix, prefix, oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_y = reshape(shape=%s_ro,x=%s_yt)[name=string(\"%s_y\")];\n",
        oc, seq, prefix, prefix, prefix, prefix];
}

static NSString * ane_gen_draft_dynamic_matmul_mil(int ic, int oc, int seq) {
    const int spIn = seq + oc;
    NSMutableString * m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ic, spIn];
    [m appendString:
        @"        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n"
        @"        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to_fp16, x=x)[name=string(\"cast_in\")];\n", ic, spIn];
    ane_append_dyn_matmul(m, "mm", ic, oc, seq, 0, seq, "x16");
    [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to_fp32, x=mm_y)[name=string(\"cast_out\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static float ane_fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.f : 0.f;
        }
        const float f = (float) mant / 1024.f * std::pow(2.f, -14.f);
        return sign ? -f : f;
    }
    if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }
    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static BOOL ane_load_matmul_weight_matrix(const char * path, int ic, int oc, float ** out_w, size_t * out_n) {
    if (!path || !path[0] || ic <= 0 || oc <= 0 || !out_w || !out_n) {
        return NO;
    }
    NSData * data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data) {
        return NO;
    }
    const NSUInteger expected = 64 + 64 + (NSUInteger) ic * (NSUInteger) oc * 2;
    if ([data length] != expected) {
        return NO;
    }
    const size_t n = (size_t) ic * (size_t) oc;
    float * w = (float *) calloc(n, sizeof(float));
    if (!w) {
        return NO;
    }
    const uint8_t * fp16 = (const uint8_t *) [data bytes] + 128;
    for (size_t idx = 0; idx < n; ++idx) {
        uint16_t bits = (uint16_t) fp16[idx * 2] | ((uint16_t) fp16[idx * 2 + 1] << 8);
        w[idx] = ane_fp16_to_f32(bits);
    }
    *out_w = w;
    *out_n = n;
    return YES;
}

static BOOL ane_compile_dynamic_matmul_kernel(
        ANEKernelHandle ** out_kernel,
        int ic,
        int oc,
        int seq,
        size_t in_bytes,
        size_t out_bytes) {
    if (!out_kernel || ic <= 0 || oc <= 0 || seq <= 0) {
        return NO;
    }
    NSString * mil = ane_gen_draft_dynamic_matmul_mil(ic, oc, seq);
    NSData * milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        return NO;
    }
    *out_kernel = ane_bridge_compile(
        (const char *) [milData bytes], [milData length],
        NULL, 0,
        1, &in_bytes, 1, &out_bytes);
    return *out_kernel != NULL;
}

static NSString * ane_gen_conv_gamma_mil(int ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        @"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"
        @"            string x_to_fp16_dtype_0 = const()[name = string(\"x_to_fp16_dtype_0\"), val = string(\"fp16\")];\n"
        @"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> x_to_fp16 = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string(\"cast_in\")];\n"
        @"            tensor<fp16, [%d, %d, 1, 1]> W0 = const()[name = string(\"W0\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        @"            tensor<fp16, [1, %d, 1, 1]> G0 = const()[name = string(\"G0\"), val = tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/gamma.bin\"), offset = uint64(64)))];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> c0 = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W0, x = x_to_fp16)[name = string(\"c0\")];\n"
        @"            tensor<fp32, [1, %d, 1, %d]> c = cast(dtype = to_fp32, x = c0)[name = string(\"cast_conv\")];\n"
        @"            tensor<fp16, [1, %d, 1, 1]> g16 = cast(dtype = x_to_fp16_dtype_0, x = G0)[name = string(\"cast_gamma\")];\n"
        @"            tensor<fp32, [1, %d, 1, 1]> g = cast(dtype = to_fp32, x = g16)[name = string(\"cast_gamma_fp32\")];\n"
        @"            tensor<fp32, [1, %d, 1, %d]> y = mul(x = c, y = g)[name = string(\"mul_gamma\")];\n"
        @"        } -> (y);\n"
        @"}\n",
        ch, sp, ch, sp, ch, ch, ch, ch, ch, ch, ch, sp, ch, sp, ch, ch, ch, ch, sp];
}

static NSString * ane_gen_conv2_mil(int ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        @"            string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"
        @"            string fp16 = const()[name = string(\"fp16\"), val = string(\"fp16\")];\n"
        @"            string fp32 = const()[name = string(\"fp32\"), val = string(\"fp32\")];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = fp16, x = x)[name = string(\"cast_in\")];\n"
        @"            tensor<fp16, [%d, %d, 1, 1]> W0 = const()[name = string(\"W0\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        @"            tensor<fp16, [%d, %d, 1, 1]> W1 = const()[name = string(\"W1\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight2.bin\"), offset = uint64(64)))];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> c0 = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = W0, x = x16)[name = string(\"c0\")];\n"
        @"            tensor<fp16, [1, %d, 1, %d]> c1 = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = W1, x = c0)[name = string(\"c1\")];\n"
        @"            tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = fp32, x = c1)[name = string(\"cast_out\")];\n"
        @"        } -> (y);\n"
        @"}\n",
        ch, sp, ch, sp, ch, ch, ch, ch, ch, ch, ch, ch, ch, sp, ch, sp, ch, sp];
}

static NSData * ane_build_weight_blob(int ch) {
    NSUInteger wsize = (NSUInteger) ch * (NSUInteger) ch * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t * buf = (uint8_t *) calloc(total, 1);
    if (!buf) {
        return nil;
    }
    buf[0] = 0x01;
    buf[4] = 0x02;
    uint8_t * chunk = buf + 64;
    chunk[0] = 0xEF;
    chunk[1] = 0xBE;
    chunk[2] = 0xAD;
    chunk[3] = 0xDE;
    chunk[4] = 0x01;
    chunk[10] = 0x08;
    uint16_t * fp16 = (uint16_t *) (chunk + 64);
    for (NSUInteger j = 0; j < wsize / 2; j++) {
        fp16[j] = 0x3400;
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSData * ane_load_weight_blob(const char * path, int ch) {
    if (!path || !path[0]) {
        return nil;
    }
    NSData * data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data) {
        return nil;
    }
    NSUInteger expected = 64 + 64 + (NSUInteger) ch * (NSUInteger) ch * 2;
    if ([data length] != expected) {
        return nil;
    }
    return data;
}

static NSData * ane_load_gamma_blob(const char * path, int ch) {
    if (!path || !path[0]) {
        return nil;
    }
    NSData * data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data) {
        return nil;
    }
    NSUInteger expected = 64 + 64 + (NSUInteger) ch * 2;
    if ([data length] != expected) {
        return nil;
    }
    return data;
}

static id<MTLComputePipelineState> ane_build_fill_pipeline(id<MTLDevice> device, NSError ** err) {
    NSString * src = @""
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void fill_buf(device float *buf [[buffer(0)]], constant float &val [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
        "    buf[gid] = val;\n"
        "}\n";
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:err];
    if (!lib) {
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"fill_buf"];
    if (!fn) {
        return nil;
    }
    return [device newComputePipelineStateWithFunction:fn error:err];
}

static BOOL ane_metal_fill(id<MTLDevice> /*device*/,
                           id<MTLCommandQueue> queue,
                           id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> buffer,
                           size_t tensorBytes,
                           size_t tensorOffset,
                           float value) {
    if (tensorBytes == 0 || tensorBytes + tensorOffset > buffer.length) {
        return NO;
    }
    size_t nfloats = tensorBytes / sizeof(float);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buffer offset:tensorOffset atIndex:0];
    [enc setBytes:&value length:sizeof(value) atIndex:1];
    NSUInteger tg = MIN((NSUInteger) 256, nfloats);
    [enc dispatchThreads:MTLSizeMake(nfloats, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

typedef struct {
    ANEKernelHandle * kernel;
    ANEKernelHandle * kernel2; // B6: second conv1 kernel when WEIGHT_FILE2 set (chained eval)
    ANEKernelHandle * kernel3; // B8: third conv1 kernel when WEIGHT_FILE3 set (attn_gate proxy)
    ANEKernelHandle * kernel4; // B9: fourth conv1 kernel when WEIGHT_FILE4 set (ffn_down proxy)
    ANEKernelHandle * kernel5; // B10: fifth conv1 kernel when WEIGHT_FILE5 set (blk.1 ffn_gate)
    ANEKernelHandle * kernel6; // B11: sixth conv1 kernel when WEIGHT_FILE6 set (blk.1 ffn_up)
    ANEKernelHandle * kernel7; // B12: seventh conv1 kernel when WEIGHT_FILE7 set (blk.1 attn_gate)
    ANEKernelHandle * kernel8; // B13: eighth conv1 kernel when WEIGHT_FILE8 set (blk.1 ffn_down)
    ANEKernelHandle * kernel9; // P9: ninth matmul kernel when WEIGHT_FILE9 set (blk.1 ffn_down)
    uint32_t          inSID;
    uint32_t          inSID2;
    uint32_t          inSID3;
    uint32_t          inSID4;
    uint32_t          inSID5;
    uint32_t          inSID6;
    uint32_t          inSID7;
    uint32_t          inSID8;
    uint32_t          inSID9;
    size_t            ioBytes;
    int               ch;
    int               sp;
    float *           outBuf;
    id<MTLDevice>             device;
    id<MTLCommandQueue>       queue;
    id<MTLComputePipelineState> pipeline;
    int               stepCount;
    bool              conv2Active;
    bool              conv3Active;
    bool              conv4Active;
    bool              conv5Active;
    bool              conv6Active;
    bool              conv7Active;
    bool              conv8Active;
    bool              matmulActive;
    bool              matmulDynamic;
    bool              matmulWeightsPrimed;
    int               matmulOc;
    float *           matmulW;
    size_t            matmulWCount;
    bool              matmul2Active;
    int               matmul2Ic;
    int               matmul2Oc;
    size_t            ioBytes2;
    bool              matmul2Dynamic;
    bool              matmul2WeightsPrimed;
    float *           matmul2W;
    size_t            matmul2WCount;
    bool              matmul3Active;
    int               matmul3Ic;
    int               matmul3Oc;
    size_t            ioBytes3;
    bool              matmul3Dynamic;
    bool              matmul3WeightsPrimed;
    float *           matmul3W;
    size_t            matmul3WCount;
    bool              matmul4Active;
    int               matmul4Ic;
    int               matmul4Oc;
    size_t            ioBytes4;
    bool              matmul4Dynamic;
    bool              matmul4WeightsPrimed;
    float *           matmul4W;
    size_t            matmul4WCount;
    bool              matmul5Active;
    int               matmul5Ic;
    int               matmul5Oc;
    size_t            ioBytes5;
    bool              matmul5Dynamic;
    bool              matmul5WeightsPrimed;
    float *           matmul5W;
    size_t            matmul5WCount;
    bool              matmul6Active;
    int               matmul6Ic;
    int               matmul6Oc;
    size_t            ioBytes6;
    bool              matmul6Dynamic;
    bool              matmul6WeightsPrimed;
    float *           matmul6W;
    size_t            matmul6WCount;
    bool              matmul7Active;
    int               matmul7Ic;
    int               matmul7Oc;
    size_t            ioBytes7;
    bool              matmul7Dynamic;
    bool              matmul7WeightsPrimed;
    float *           matmul7W;
    size_t            matmul7WCount;
    bool              dflashFcActive;
    bool              dflashChain11Active;
    bool              dflashChain12Active;
    bool              dflashChain13Active;
    bool              dflashChain14Active;
    bool              dflashChain15Active;
    bool              dflashChain16Active;
    bool              dflashChain17Active;
    float *           outputNormGamma;
    int               outputNormGammaLen;
    float *           attnPostNormGamma;
    int               attnPostNormGammaLen;
    float *           lastDflashWoHidden;
    int               lastDflashWoHiddenLen;
    float *           lastDflashQ;
    float *           lastDflashK;
    float *           lastDflashV;
    int               lastDflashAttnLen;
    float *           hiddenNormGamma;
    int               hiddenNormGammaLen;
    bool              matmul9Active;
    int               matmul9Ic;
    int               matmul9Oc;
    size_t            ioBytes9;
    bool              matmul9Dynamic;
    bool              matmul9WeightsPrimed;
    float *           matmul9W;
    size_t            matmul9WCount;
    bool              matmul10Active;
    int               matmul10Ic;
    int               matmul10Oc;
    size_t            ioBytes10;
    bool              matmul10Dynamic;
    bool              matmul10WeightsPrimed;
    float *           matmul10W;
    size_t            matmul10WCount;
    float *           lastQkvHidden;
    int               lastQkvHiddenLen;
    float *           lastDownHidden;
    int               lastDownHiddenLen;
    int               matmulChain;
    float *           lastHidden;
    int               lastHiddenLen;
    float *           dflashFcHost;
    int               dflashFcHostLen;
    size_t            outIoBytes;
} ANEDraftSession;

static ANEDraftSession g_session {};

static void ane_session_clear(ANEDraftSession * st) {
    if (st->kernel) {
        ane_bridge_free(st->kernel);
        st->kernel = NULL;
    }
    if (st->kernel2) {
        ane_bridge_free(st->kernel2);
        st->kernel2 = NULL;
    }
    if (st->kernel3) {
        ane_bridge_free(st->kernel3);
        st->kernel3 = NULL;
    }
    if (st->kernel4) {
        ane_bridge_free(st->kernel4);
        st->kernel4 = NULL;
    }
    if (st->kernel5) {
        ane_bridge_free(st->kernel5);
        st->kernel5 = NULL;
    }
    if (st->kernel6) {
        ane_bridge_free(st->kernel6);
        st->kernel6 = NULL;
    }
    if (st->kernel7) {
        ane_bridge_free(st->kernel7);
        st->kernel7 = NULL;
    }
    if (st->kernel8) {
        ane_bridge_free(st->kernel8);
        st->kernel8 = NULL;
    }
    if (st->kernel9) {
        ane_bridge_free(st->kernel9);
        st->kernel9 = NULL;
    }
    if (st->outBuf) {
        free(st->outBuf);
        st->outBuf = NULL;
    }
    if (st->matmulW) {
        free(st->matmulW);
        st->matmulW = NULL;
    }
    if (st->matmul2W) {
        free(st->matmul2W);
        st->matmul2W = NULL;
    }
    if (st->matmul3W) {
        free(st->matmul3W);
        st->matmul3W = NULL;
    }
    if (st->matmul4W) {
        free(st->matmul4W);
        st->matmul4W = NULL;
    }
    if (st->matmul5W) {
        free(st->matmul5W);
        st->matmul5W = NULL;
    }
    if (st->matmul6W) {
        free(st->matmul6W);
        st->matmul6W = NULL;
    }
    if (st->lastHidden) {
        free(st->lastHidden);
        st->lastHidden = NULL;
    }
    st->inSID   = 0;
    st->inSID2  = 0;
    st->inSID3  = 0;
    st->inSID4  = 0;
    st->inSID5  = 0;
    st->inSID6  = 0;
    st->inSID7  = 0;
    st->inSID8  = 0;
    st->ioBytes = 0;
    st->ch      = 0;
    st->sp      = 0;
    st->device  = nil;
    st->queue   = nil;
    st->pipeline = nil;
    st->stepCount = 0;
    st->conv2Active = false;
    st->conv3Active = false;
    st->conv4Active = false;
    st->conv5Active = false;
    st->conv6Active = false;
    st->conv7Active = false;
    st->conv8Active = false;
    st->matmulActive = false;
    st->matmulDynamic = false;
    st->matmulWeightsPrimed = false;
    st->matmulOc = 0;
    st->matmulW = NULL;
    st->matmulWCount = 0;
    st->matmul2Active = false;
    st->matmul2Ic = 0;
    st->matmul2Oc = 0;
    st->matmul2Dynamic = false;
    st->matmul2WeightsPrimed = false;
    st->matmul2W = NULL;
    st->matmul2WCount = 0;
    st->matmul3Active = false;
    st->matmul3Ic = 0;
    st->matmul3Oc = 0;
    st->ioBytes3 = 0;
    st->matmul3Dynamic = false;
    st->matmul3WeightsPrimed = false;
    st->matmul3W = NULL;
    st->matmul3WCount = 0;
    st->matmul4Active = false;
    st->matmul4Ic = 0;
    st->matmul4Oc = 0;
    st->ioBytes4 = 0;
    st->matmul4Dynamic = false;
    st->matmul4WeightsPrimed = false;
    st->matmul4W = NULL;
    st->matmul4WCount = 0;
    st->matmul5Active = false;
    st->matmul5Ic = 0;
    st->matmul5Oc = 0;
    st->ioBytes5 = 0;
    st->matmul5Dynamic = false;
    st->matmul5WeightsPrimed = false;
    st->matmul5W = NULL;
    st->matmul5WCount = 0;
    st->matmul6Active = false;
    st->matmul6Ic = 0;
    st->matmul6Oc = 0;
    st->ioBytes6 = 0;
    st->matmul6Dynamic = false;
    st->matmul6WeightsPrimed = false;
    st->matmul6W = NULL;
    st->matmul6WCount = 0;
    st->matmul7Active = false;
    st->matmul7Ic = 0;
    st->matmul7Oc = 0;
    st->ioBytes7 = 0;
    st->matmul7Dynamic = false;
    st->matmul7WeightsPrimed = false;
    if (st->matmul7W) {
        free(st->matmul7W);
        st->matmul7W = NULL;
    }
    st->matmul7WCount = 0;
    st->dflashFcActive = false;
    st->dflashChain11Active = false;
    st->dflashChain12Active = false;
    st->dflashChain13Active = false;
    st->dflashChain14Active = false;
    st->dflashChain15Active = false;
    st->dflashChain16Active = false;
    st->dflashChain17Active = false;
    if (st->outputNormGamma) {
        free(st->outputNormGamma);
        st->outputNormGamma = NULL;
    }
    st->outputNormGammaLen = 0;
    if (st->attnPostNormGamma) {
        free(st->attnPostNormGamma);
        st->attnPostNormGamma = NULL;
    }
    st->attnPostNormGammaLen = 0;
    if (st->lastDflashWoHidden) {
        free(st->lastDflashWoHidden);
        st->lastDflashWoHidden = NULL;
    }
    st->lastDflashWoHiddenLen = 0;
    if (st->lastDflashQ) {
        free(st->lastDflashQ);
        st->lastDflashQ = NULL;
    }
    if (st->lastDflashK) {
        free(st->lastDflashK);
        st->lastDflashK = NULL;
    }
    if (st->lastDflashV) {
        free(st->lastDflashV);
        st->lastDflashV = NULL;
    }
    st->lastDflashAttnLen = 0;
    if (st->hiddenNormGamma) {
        free(st->hiddenNormGamma);
        st->hiddenNormGamma = NULL;
    }
    st->hiddenNormGammaLen = 0;
    st->matmul9Active = false;
    st->matmul9Ic = 0;
    st->matmul9Oc = 0;
    st->ioBytes9 = 0;
    st->matmul9Dynamic = false;
    st->matmul9WeightsPrimed = false;
    if (st->matmul9W) {
        free(st->matmul9W);
        st->matmul9W = NULL;
    }
    st->matmul9WCount = 0;
    st->matmul10Active = false;
    st->matmul10Ic = 0;
    st->matmul10Oc = 0;
    st->ioBytes10 = 0;
    st->matmul10Dynamic = false;
    st->matmul10WeightsPrimed = false;
    if (st->matmul10W) {
        free(st->matmul10W);
        st->matmul10W = NULL;
    }
    st->matmul10WCount = 0;
    if (st->lastQkvHidden) {
        free(st->lastQkvHidden);
        st->lastQkvHidden = NULL;
    }
    st->lastQkvHiddenLen = 0;
    if (st->lastDownHidden) {
        free(st->lastDownHidden);
        st->lastDownHidden = NULL;
    }
    st->lastDownHiddenLen = 0;
    st->matmulChain = 0;
    st->lastHidden = NULL;
    st->lastHiddenLen = 0;
    if (st->dflashFcHost) {
        free(st->dflashFcHost);
        st->dflashFcHost = NULL;
    }
    st->dflashFcHostLen = 0;
    st->ioBytes2 = 0;
    st->outIoBytes = 0;
}

static BOOL ane_compile_conv_kernel(
        ANEKernelHandle ** out_kernel,
        NSData * weight_blob,
        int channels,
        int spatial,
        size_t io_bytes) {
    if (!out_kernel || !weight_blob) {
        return NO;
    }
    NSString * mil = ane_gen_conv_mil(channels, spatial);
    NSData * milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        return NO;
    }
    const char * weightNames[1] = { "@model_path/weights/weight.bin" };
    const uint8_t * weightDatas[1] = { (const uint8_t *) [weight_blob bytes] };
    size_t weightLens[1] = { [weight_blob length] };
    *out_kernel = ane_bridge_compile_multi_weights(
        (const char *) [milData bytes], [milData length],
        weightNames, weightDatas, weightLens, 1,
        1, &io_bytes, 1, &io_bytes);
    return *out_kernel != NULL;
}

static bool ane_session_write_surface(uint32_t surface_id, size_t io_bytes, const float * src) {
    if (surface_id == 0 || !src || io_bytes == 0) {
        return false;
    }
    IOSurfaceRef surface = ane_ggml_surface_from_id(surface_id);
    if (!surface) {
        return false;
    }
    IOSurfaceLock(surface, 0, NULL);
    float * base = (float *) IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return false;
    }
    memcpy(base, src, io_bytes);
    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    return true;
}

static bool ane_session_prime_matmul_weights(void) {
    if (!g_session.matmulDynamic || !g_session.matmulW || g_session.matmulWCount == 0 || g_session.inSID == 0) {
        return false;
    }
    const int ic = g_session.ch;
    const int oc = g_session.matmulOc > 0 ? g_session.matmulOc : ic;
    const int seq = g_session.sp;
    const int spIn = seq + oc;
    if (ic <= 0 || oc <= 0 || seq <= 0) {
        return false;
    }
    if ((size_t) ic * (size_t) oc > g_session.matmulWCount) {
        return false;
    }
    IOSurfaceRef surface = ane_ggml_surface_from_id(g_session.inSID);
    if (!surface) {
        return false;
    }
    IOSurfaceLock(surface, 0, NULL);
    float * base = (float *) IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return false;
    }
    for (int c = 0; c < ic; ++c) {
        for (int o = 0; o < oc; ++o) {
            base[(size_t) c * (size_t) spIn + (size_t) seq + (size_t) o] =
                g_session.matmulW[(size_t) c * (size_t) oc + (size_t) o];
        }
    }
    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    g_session.matmulWeightsPrimed = true;
    return true;
}

static bool ane_session_prime_matmul_weights(uint32_t surface_id, const float * W, size_t wcount, int ic, int oc, int seq) {
    if (!W || wcount == 0 || surface_id == 0 || ic <= 0 || oc <= 0 || seq <= 0) {
        return false;
    }
    const int spIn = seq + oc;
    if ((size_t) ic * (size_t) oc > wcount) {
        return false;
    }
    IOSurfaceRef surface = ane_ggml_surface_from_id(surface_id);
    if (!surface) {
        return false;
    }
    IOSurfaceLock(surface, 0, NULL);
    float * base = (float *) IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return false;
    }
    for (int c = 0; c < ic; ++c) {
        for (int o = 0; o < oc; ++o) {
            base[(size_t) c * (size_t) spIn + (size_t) seq + (size_t) o] =
                W[(size_t) c * (size_t) oc + (size_t) o];
        }
    }
    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    return true;
}

static bool ane_session_prime_matmul2_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul2W || g_session.matmul2WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul2W, g_session.matmul2WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul2WeightsPrimed = true;
    return true;
}

static bool ane_session_prime_matmul3_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul3W || g_session.matmul3WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul3W, g_session.matmul3WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul3WeightsPrimed = true;
    return true;
}

static bool ane_session_prime_matmul4_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul4W || g_session.matmul4WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul4W, g_session.matmul4WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul4WeightsPrimed = true;
    return true;
}

static bool ane_session_prime_matmul5_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul5W || g_session.matmul5WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul5W, g_session.matmul5WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul5WeightsPrimed = true;
    return true;
}

static bool ane_session_prime_matmul6_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul6W || g_session.matmul6WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul6W, g_session.matmul6WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul6WeightsPrimed = true;
    return true;
}

static bool ane_session_prime_matmul7_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul7W || g_session.matmul7WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul7W, g_session.matmul7WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul7WeightsPrimed = true;
    return true;
}

static bool ane_session_read_matmul_activations(uint32_t surface_id, float * dst, int dst_len, int ic, int oc, int seq) {
    if (!dst || surface_id == 0 || ic <= 0 || oc <= 0 || seq <= 0 || dst_len <= 0) {
        return false;
    }
    const int spIn = seq + oc;
    IOSurfaceRef surface = ane_ggml_surface_from_id(surface_id);
    if (!surface) {
        return false;
    }
    IOSurfaceLock(surface, 0, NULL);
    const float * base = (const float *) IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return false;
    }
    const int n = ic < dst_len ? ic : dst_len;
    for (int c = 0; c < n; ++c) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) base[(size_t) c * (size_t) spIn + (size_t) s];
        }
        dst[c] = (float) (sum / (double) (seq > 0 ? seq : 1));
    }
    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    return true;
}

static bool ane_session_pack_matmul2_activations(uint32_t surface_id, const float * act, int act_len, int ic, int oc, int seq) {
    if (!act || surface_id == 0 || ic <= 0 || oc <= 0 || seq <= 0) {
        return false;
    }
    const bool weights_ok = (surface_id == g_session.inSID9)
        ? g_session.matmul10WeightsPrimed
        : (surface_id == g_session.inSID8)
            ? g_session.matmul9WeightsPrimed
            : (surface_id == g_session.inSID7)
                ? g_session.matmul7WeightsPrimed
                : (surface_id == g_session.inSID6)
                    ? g_session.matmul6WeightsPrimed
                    : (surface_id == g_session.inSID5)
                        ? g_session.matmul5WeightsPrimed
                        : (surface_id == g_session.inSID4)
                            ? g_session.matmul4WeightsPrimed
                            : (surface_id == g_session.inSID3)
                                ? g_session.matmul3WeightsPrimed
                                : g_session.matmul2WeightsPrimed;
    if (!weights_ok) {
        return false;
    }
    const int spIn = seq + oc;
    IOSurfaceRef surface = ane_ggml_surface_from_id(surface_id);
    if (!surface) {
        return false;
    }
    IOSurfaceLock(surface, 0, NULL);
    float * base = (float *) IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return false;
    }
    for (int c = 0; c < ic; ++c) {
        const float v = (c < act_len) ? act[c] : 0.f;
        for (int s = 0; s < seq; ++s) {
            base[(size_t) c * (size_t) spIn + (size_t) s] = v;
        }
    }
    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    return true;
}

static bool ane_session_prime_matmul9_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul9W || g_session.matmul9WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul9W, g_session.matmul9WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul9WeightsPrimed = true;
    return true;
}

static bool ane_session_prime_matmul10_weights(uint32_t surface_id, int ic, int oc, int seq) {
    if (!g_session.matmul10W || g_session.matmul10WCount == 0) {
        return false;
    }
    if (!ane_session_prime_matmul_weights(surface_id, g_session.matmul10W, g_session.matmul10WCount, ic, oc, seq)) {
        return false;
    }
    g_session.matmul10WeightsPrimed = true;
    return true;
}

static float ane_silu(float x) {
    return x / (1.f + expf(-x));
}

static void ane_host_matmul_seq(const float * act, int ic, int oc, int seq,
                                  const float * W, float * out) {
    if (!act || !W || !out || ic <= 0 || oc <= 0 || seq <= 0) {
        return;
    }
    for (int o = 0; o < oc; ++o) {
        double sum = 0.0;
        for (int i = 0; i < ic; ++i) {
            sum += (double) act[i] * (double) W[(size_t) i * (size_t) oc + (size_t) o];
        }
        const float v = (float) sum;
        for (int s = 0; s < seq; ++s) {
            out[(size_t) o * (size_t) seq + (size_t) s] = v;
        }
    }
}

static int ane_matmul_chain_target(void) {
    int chain = 1;
    bool chain_explicit = false;
    if (const char * ce = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN")) {
        const int v = (int) strtol(ce, NULL, 10);
        if (v > 0) {
            chain = v;
            chain_explicit = true;
        }
    }
    const char * w2path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
    const char * w3path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
    const char * w4path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
    const char * w5path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
    const char * w6path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
    const char * w7path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
    const char * w8path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
    const char * w9path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9");
    if (!chain_explicit) {
        if (chain < 10 && w9path && w9path[0] && w8path && w8path[0] && w7path && w7path[0] && w6path && w6path[0] && w5path && w5path[0] && w4path && w4path[0] && w3path && w3path[0] && w2path && w2path[0]) {
            chain = 10;
        }
        if (chain < 9 && w8path && w8path[0] && w7path && w7path[0] && w6path && w6path[0] && w5path && w5path[0] && w4path && w4path[0] && w3path && w3path[0] && w2path && w2path[0]) {
            chain = 9;
        }
        if (chain < 7 && w7path && w7path[0] && w6path && w6path[0] && w5path && w5path[0] && w4path && w4path[0] && w3path && w3path[0] && w2path && w2path[0]) {
            chain = 7;
        }
        if (chain < 6 && w6path && w6path[0] && w5path && w5path[0] && w4path && w4path[0] && w3path && w3path[0] && w2path && w2path[0]) {
            chain = 6;
        }
        if (chain < 5 && w5path && w5path[0] && w4path && w4path[0] && w3path && w3path[0] && w2path && w2path[0]) {
            chain = 5;
        }
        if (chain < 4 && w4path && w4path[0] && w3path && w3path[0] && w2path && w2path[0]) {
            chain = 4;
        }
        if (chain < 3 && w3path && w3path[0] && w2path && w2path[0]) {
            chain = 3;
        }
        if (chain < 2 && w2path && w2path[0]) {
            chain = 2;
        }
    } else if (chain == 8) {
        // P7b: standalone dflash_fc — do not auto-extend from WEIGHT_FILE2..7.
        return 8;
    } else if (chain == 9) {
        return 9;
    } else if (chain == 10) {
        return 10;
    } else if (chain == 11) {
        return 11;
    } else if (chain == 12) {
        return 12;
    } else if (chain == 13) {
        return 13;
    } else if (chain == 14) {
        return 14;
    } else if (chain == 15) {
        return 15;
    } else if (chain == 16) {
        return 16;
    } else if (chain == 17) {
        return 17;
    }
    if (chain < 1) {
        chain = 1;
    }
    return chain;
}

static bool ane_session_init_dflash_chain11(int seq);
static bool ane_session_eval_dflash_chain11(int seq);
static bool ane_session_init_dflash_chain12(int seq);
static bool ane_session_eval_dflash_chain12(int seq);
static bool ane_session_init_dflash_chain13(int seq);
static bool ane_session_init_dflash_chain14(int seq);
static bool ane_session_init_dflash_chain15(int seq);
static bool ane_session_init_dflash_chain16(int seq);
static bool ane_session_init_dflash_chain17(int seq);

static bool ane_session_init_matmul2(int seq, int chain) {
    const char * w2path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
    if (chain < 2 || !w2path || !w2path[0] || !g_session.matmulDynamic) {
        return true;
    }
    int ic2 = 0;
    int oc2 = 0;
    if (chain >= 3) {
        ic2 = g_session.ch;
        oc2 = g_session.matmulOc;
    } else {
        ic2 = g_session.matmulOc;
        oc2 = g_session.ch;
    }
    if (const char * oc2_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2")) {
        const int v = (int) strtol(oc2_env, NULL, 10);
        if (v > 0) {
            oc2 = v;
        }
    }
    if (ic2 <= 0 || oc2 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w2path, ic2, oc2, &g_session.matmul2W, &wcount)) {
        LOG_WRN("%s: matmul2 weight load failed path=%s ic=%d oc=%d chain=%d\n", __func__, w2path, ic2, oc2, chain);
        return false;
    }
    g_session.matmul2WCount = wcount;
    g_session.matmul2Ic = ic2;
    g_session.matmul2Oc = oc2;
    const int spIn2 = seq + oc2;
    g_session.ioBytes2 = (size_t) ic2 * (size_t) spIn2 * sizeof(float);
    const size_t outBytes2 = (size_t) oc2 * (size_t) seq * sizeof(float);
    if (chain < 3) {
        g_session.outIoBytes = outBytes2;
    }
    g_session.matmul2Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel2, ic2, oc2, seq, g_session.ioBytes2, outBytes2)) {
        LOG_WRN("%s: matmul2 dynamic compile failed ic=%d oc=%d seq=%d\n", __func__, ic2, oc2, seq);
        return false;
    }
    g_session.inSID2 = ane_bridge_input_surface_id(g_session.kernel2, 0);
    if (g_session.inSID2 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul2_weights(g_session.inSID2, ic2, oc2, seq)) {
        LOG_WRN("%s: matmul2 weight prime failed ic=%d oc=%d\n", __func__, ic2, oc2);
        return false;
    }
    g_session.matmul2Active = true;
    if (chain < 3) {
        if (g_session.outBuf) {
            free(g_session.outBuf);
        }
        g_session.outBuf = (float *) calloc((size_t) oc2 * (size_t) seq, sizeof(float));
        if (!g_session.outBuf) {
            return false;
        }
        LOG_INF("%s: P2 matmul chain gate+silu+up ic=%d→oc=%d then ic=%d→oc=%d seq=%d\n",
                __func__, g_session.ch, g_session.matmulOc, ic2, oc2, seq);
    } else {
        LOG_INF("%s: P3 matmul chain up proj ic=%d→oc=%d seq=%d\n", __func__, ic2, oc2, seq);
    }
    return true;
}

static bool ane_session_init_matmul3(int seq, int chain) {
    const char * w3path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
    if (chain < 3 || !w3path || !w3path[0] || !g_session.matmulDynamic || !g_session.matmul2Active) {
        return true;
    }
    int ic3 = g_session.matmulOc;
    int oc3 = g_session.ch;
    if (const char * oc3_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC3")) {
        const int v = (int) strtol(oc3_env, NULL, 10);
        if (v > 0) {
            oc3 = v;
        }
    }
    if (ic3 <= 0 || oc3 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w3path, ic3, oc3, &g_session.matmul3W, &wcount)) {
        LOG_WRN("%s: matmul3 weight load failed path=%s ic=%d oc=%d\n", __func__, w3path, ic3, oc3);
        return false;
    }
    g_session.matmul3WCount = wcount;
    g_session.matmul3Ic = ic3;
    g_session.matmul3Oc = oc3;
    const int spIn3 = seq + oc3;
    g_session.ioBytes3 = (size_t) ic3 * (size_t) spIn3 * sizeof(float);
    g_session.outIoBytes = (size_t) oc3 * (size_t) seq * sizeof(float);
    g_session.matmul3Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel3, ic3, oc3, seq, g_session.ioBytes3, g_session.outIoBytes)) {
        LOG_WRN("%s: matmul3 dynamic compile failed ic=%d oc=%d seq=%d\n", __func__, ic3, oc3, seq);
        return false;
    }
    g_session.inSID3 = ane_bridge_input_surface_id(g_session.kernel3, 0);
    if (g_session.inSID3 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul3_weights(g_session.inSID3, ic3, oc3, seq)) {
        LOG_WRN("%s: matmul3 weight prime failed ic=%d oc=%d\n", __func__, ic3, oc3);
        return false;
    }
    g_session.matmul3Active = true;
    if (g_session.outBuf) {
        free(g_session.outBuf);
    }
    g_session.outBuf = (float *) calloc((size_t) oc3 * (size_t) seq, sizeof(float));
    if (!g_session.outBuf) {
        return false;
    }
    LOG_INF("%s: P3 matmul chain swiglu+down ic=%d→oc=%d seq=%d (gate=%d up=%d)\n",
            __func__, ic3, oc3, seq, g_session.ch, g_session.matmulOc);
    return true;
}

static bool ane_session_init_matmul4(int seq, int chain) {
    const char * w4path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
    if (chain < 4 || !w4path || !w4path[0] || !g_session.matmulDynamic || !g_session.matmul3Active) {
        return true;
    }
    int ic4 = g_session.matmul3Oc;
    int oc4 = g_session.matmulOc;
    if (const char * oc4_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4")) {
        const int v = (int) strtol(oc4_env, NULL, 10);
        if (v > 0) {
            oc4 = v;
        }
    }
    if (ic4 <= 0 || oc4 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w4path, ic4, oc4, &g_session.matmul4W, &wcount)) {
        LOG_WRN("%s: matmul4 weight load failed path=%s ic=%d oc=%d\n", __func__, w4path, ic4, oc4);
        return false;
    }
    g_session.matmul4WCount = wcount;
    g_session.matmul4Ic = ic4;
    g_session.matmul4Oc = oc4;
    const int spIn4 = seq + oc4;
    g_session.ioBytes4 = (size_t) ic4 * (size_t) spIn4 * sizeof(float);
    g_session.outIoBytes = (size_t) oc4 * (size_t) seq * sizeof(float);
    g_session.matmul4Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel4, ic4, oc4, seq, g_session.ioBytes4, g_session.outIoBytes)) {
        LOG_WRN("%s: matmul4 dynamic compile failed ic=%d oc=%d seq=%d\n", __func__, ic4, oc4, seq);
        return false;
    }
    g_session.inSID4 = ane_bridge_input_surface_id(g_session.kernel4, 0);
    if (g_session.inSID4 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul4_weights(g_session.inSID4, ic4, oc4, seq)) {
        LOG_WRN("%s: matmul4 weight prime failed ic=%d oc=%d\n", __func__, ic4, oc4);
        return false;
    }
    g_session.matmul4Active = true;
    if (g_session.outBuf) {
        free(g_session.outBuf);
    }
    g_session.outBuf = (float *) calloc((size_t) oc4 * (size_t) seq, sizeof(float));
    if (!g_session.outBuf) {
        return false;
    }
    if (g_session.lastDownHiddenLen != ic4) {
        free(g_session.lastDownHidden);
        g_session.lastDownHidden = (float *) calloc((size_t) ic4, sizeof(float));
        g_session.lastDownHiddenLen = g_session.lastDownHidden ? ic4 : 0;
    }
    LOG_INF("%s: P4 matmul chain attn_gate ic=%d→oc=%d seq=%d (after ffn_down=%d)\n",
            __func__, ic4, oc4, seq, ic4);
    return true;
}

static bool ane_session_init_matmul5(int seq, int chain) {
    const char * w5path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
    if (chain < 5 || !w5path || !w5path[0] || !g_session.matmulDynamic || !g_session.matmul4Active) {
        return true;
    }
    int ic5 = g_session.matmul3Oc;
    int oc5 = g_session.matmulOc;
    if (const char * oc5_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5")) {
        const int v = (int) strtol(oc5_env, NULL, 10);
        if (v > 0) {
            oc5 = v;
        }
    }
    if (ic5 <= 0 || oc5 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w5path, ic5, oc5, &g_session.matmul5W, &wcount)) {
        LOG_WRN("%s: matmul5 weight load failed path=%s ic=%d oc=%d\n", __func__, w5path, ic5, oc5);
        return false;
    }
    g_session.matmul5WCount = wcount;
    g_session.matmul5Ic = ic5;
    g_session.matmul5Oc = oc5;
    const int spIn5 = seq + oc5;
    g_session.ioBytes5 = (size_t) ic5 * (size_t) spIn5 * sizeof(float);
    g_session.outIoBytes = (size_t) oc5 * (size_t) seq * sizeof(float);
    g_session.matmul5Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel5, ic5, oc5, seq, g_session.ioBytes5, g_session.outIoBytes)) {
        LOG_WRN("%s: matmul5 dynamic compile failed ic=%d oc=%d seq=%d\n", __func__, ic5, oc5, seq);
        return false;
    }
    g_session.inSID5 = ane_bridge_input_surface_id(g_session.kernel5, 0);
    if (g_session.inSID5 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul5_weights(g_session.inSID5, ic5, oc5, seq)) {
        LOG_WRN("%s: matmul5 weight prime failed ic=%d oc=%d\n", __func__, ic5, oc5);
        return false;
    }
    g_session.matmul5Active = true;
    if (g_session.outBuf) {
        free(g_session.outBuf);
    }
    g_session.outBuf = (float *) calloc((size_t) oc5 * (size_t) seq, sizeof(float));
    if (!g_session.outBuf) {
        return false;
    }
    LOG_INF("%s: P5 matmul chain ssm_out ic=%d→oc=%d seq=%d (parallel branch from ffn_down)\n",
            __func__, ic5, oc5, seq);
    return true;
}

static bool ane_session_init_matmul6(int seq, int chain) {
    const char * w6path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
    if (chain < 6 || !w6path || !w6path[0] || !g_session.matmulDynamic) {
        return true;
    }
    int ic6 = g_session.ch;
    int oc6 = g_session.matmulOc;
    if (const char * oc6_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6")) {
        const int v = (int) strtol(oc6_env, NULL, 10);
        if (v > 0) {
            oc6 = v;
        }
    }
    if (ic6 <= 0 || oc6 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w6path, ic6, oc6, &g_session.matmul6W, &wcount)) {
        LOG_WRN("%s: matmul6 qkv weight load failed path=%s ic=%d oc=%d\n", __func__, w6path, ic6, oc6);
        return false;
    }
    g_session.matmul6WCount = wcount;
    g_session.matmul6Ic = ic6;
    g_session.matmul6Oc = oc6;
    const int spIn6 = seq + oc6;
    g_session.ioBytes6 = (size_t) ic6 * (size_t) spIn6 * sizeof(float);
    const size_t qkv_out_bytes = (size_t) oc6 * (size_t) seq * sizeof(float);
    g_session.matmul6Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel6, ic6, oc6, seq, g_session.ioBytes6, qkv_out_bytes)) {
        LOG_WRN("%s: matmul6 qkv compile failed ic=%d oc=%d seq=%d\n", __func__, ic6, oc6, seq);
        return false;
    }
    g_session.inSID6 = ane_bridge_input_surface_id(g_session.kernel6, 0);
    if (g_session.inSID6 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul6_weights(g_session.inSID6, ic6, oc6, seq)) {
        LOG_WRN("%s: matmul6 qkv weight prime failed ic=%d oc=%d\n", __func__, ic6, oc6);
        return false;
    }
    g_session.matmul6Active = true;
    if (g_session.lastQkvHiddenLen != oc6) {
        free(g_session.lastQkvHidden);
        g_session.lastQkvHidden = (float *) calloc((size_t) oc6, sizeof(float));
        g_session.lastQkvHiddenLen = g_session.lastQkvHidden ? oc6 : 0;
    }
    LOG_INF("%s: P6 matmul prefix attn_qkv ic=%d→oc=%d seq=%d (before ffn chain)\n",
            __func__, ic6, oc6, seq);
    return true;
}

static bool ane_session_init_matmul7(int seq, int chain) {
    const char * w7path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
    if (chain < 7 || !w7path || !w7path[0] || !g_session.matmulDynamic || !g_session.matmul3Active) {
        return true;
    }
    int ic7 = g_session.matmul3Oc;
    int oc7 = g_session.matmulOc;
    if (const char * oc7_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC7")) {
        const int v = (int) strtol(oc7_env, NULL, 10);
        if (v > 0) {
            oc7 = v;
        }
    }
    if (ic7 <= 0 || oc7 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w7path, ic7, oc7, &g_session.matmul7W, &wcount)) {
        LOG_WRN("%s: matmul7 blk.1 gate weight load failed path=%s ic=%d oc=%d\n", __func__, w7path, ic7, oc7);
        return false;
    }
    g_session.matmul7WCount = wcount;
    g_session.matmul7Ic = ic7;
    g_session.matmul7Oc = oc7;
    const int spIn7 = seq + oc7;
    g_session.ioBytes7 = (size_t) ic7 * (size_t) spIn7 * sizeof(float);
    g_session.outIoBytes = (size_t) oc7 * (size_t) seq * sizeof(float);
    g_session.matmul7Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel7, ic7, oc7, seq, g_session.ioBytes7, g_session.outIoBytes)) {
        LOG_WRN("%s: matmul7 blk.1 gate compile failed ic=%d oc=%d seq=%d\n", __func__, ic7, oc7, seq);
        return false;
    }
    g_session.inSID7 = ane_bridge_input_surface_id(g_session.kernel7, 0);
    if (g_session.inSID7 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul7_weights(g_session.inSID7, ic7, oc7, seq)) {
        LOG_WRN("%s: matmul7 blk.1 gate weight prime failed ic=%d oc=%d\n", __func__, ic7, oc7);
        return false;
    }
    g_session.matmul7Active = true;
    LOG_INF("%s: P7 matmul chain blk.1 ffn_gate ic=%d→oc=%d seq=%d (after blk.0 ffn_down=%d)\n",
            __func__, ic7, oc7, seq, ic7);
    return true;
}

static bool ane_session_init_matmul9(int seq, int chain) {
    const char * w8path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
    if (chain < 9 || !w8path || !w8path[0] || !g_session.matmulDynamic || !g_session.matmul7Active) {
        return true;
    }
    int ic9 = g_session.matmul3Oc;
    int oc9 = g_session.matmulOc;
    if (const char * oc9_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC9")) {
        const int v = (int) strtol(oc9_env, NULL, 10);
        if (v > 0) {
            oc9 = v;
        }
    }
    if (ic9 <= 0 || oc9 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w8path, ic9, oc9, &g_session.matmul9W, &wcount)) {
        LOG_WRN("%s: matmul9 blk.1 up weight load failed path=%s ic=%d oc=%d\n", __func__, w8path, ic9, oc9);
        return false;
    }
    g_session.matmul9WCount = wcount;
    g_session.matmul9Ic = ic9;
    g_session.matmul9Oc = oc9;
    const int spIn9 = seq + oc9;
    g_session.ioBytes9 = (size_t) ic9 * (size_t) spIn9 * sizeof(float);
    const size_t up_out_bytes = (size_t) oc9 * (size_t) seq * sizeof(float);
    g_session.matmul9Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel8, ic9, oc9, seq, g_session.ioBytes9, up_out_bytes)) {
        LOG_WRN("%s: matmul9 blk.1 up compile failed ic=%d oc=%d seq=%d\n", __func__, ic9, oc9, seq);
        return false;
    }
    g_session.inSID8 = ane_bridge_input_surface_id(g_session.kernel8, 0);
    if (g_session.inSID8 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul9_weights(g_session.inSID8, ic9, oc9, seq)) {
        LOG_WRN("%s: matmul9 blk.1 up weight prime failed ic=%d oc=%d\n", __func__, ic9, oc9);
        return false;
    }
    g_session.matmul9Active = true;
    LOG_INF("%s: P8 matmul chain blk.1 ffn_up ic=%d→oc=%d seq=%d (SwiGLU with blk.1 gate)\n",
            __func__, ic9, oc9, seq);
    return true;
}

static bool ane_session_init_matmul10(int seq, int chain) {
    const char * w9path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9");
    if (chain < 10 || !w9path || !w9path[0] || !g_session.matmulDynamic || !g_session.matmul9Active) {
        return true;
    }
    int ic10 = g_session.matmul9Oc;
    int oc10 = g_session.matmul3Oc;
    if (const char * oc10_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC10")) {
        const int v = (int) strtol(oc10_env, NULL, 10);
        if (v > 0) {
            oc10 = v;
        }
    }
    if (ic10 <= 0 || oc10 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w9path, ic10, oc10, &g_session.matmul10W, &wcount)) {
        LOG_WRN("%s: matmul10 blk.1 down weight load failed path=%s ic=%d oc=%d\n", __func__, w9path, ic10, oc10);
        return false;
    }
    g_session.matmul10WCount = wcount;
    g_session.matmul10Ic = ic10;
    g_session.matmul10Oc = oc10;
    const int spIn10 = seq + oc10;
    g_session.ioBytes10 = (size_t) ic10 * (size_t) spIn10 * sizeof(float);
    const size_t down_out_bytes = (size_t) oc10 * (size_t) seq * sizeof(float);
    if (g_session.outBuf) {
        free(g_session.outBuf);
    }
    g_session.outBuf = (float *) calloc((size_t) oc10 * (size_t) seq, sizeof(float));
    if (!g_session.outBuf) {
        return false;
    }
    g_session.outIoBytes = down_out_bytes;
    g_session.matmul10Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel9, ic10, oc10, seq, g_session.ioBytes10, down_out_bytes)) {
        LOG_WRN("%s: matmul10 blk.1 down compile failed ic=%d oc=%d seq=%d\n", __func__, ic10, oc10, seq);
        return false;
    }
    g_session.inSID9 = ane_bridge_input_surface_id(g_session.kernel9, 0);
    if (g_session.inSID9 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul10_weights(g_session.inSID9, ic10, oc10, seq)) {
        LOG_WRN("%s: matmul10 blk.1 down weight prime failed ic=%d oc=%d\n", __func__, ic10, oc10);
        return false;
    }
    g_session.matmul10Active = true;
    LOG_INF("%s: P9 matmul chain blk.1 ffn_down ic=%d→oc=%d seq=%d (after blk.1 SwiGLU)\n",
            __func__, ic10, oc10, seq);
    return true;
}

bool ane_draft_session_supported(void) {
    return true;
}

bool ane_draft_session_init(int channels, int spatial, const char * weight_path, const char * gamma_path) {
    @autoreleasepool {
        if (g_session.kernel) {
            return true;
        }
        if (channels <= 0 || spatial <= 0) {
            return false;
        }

        ane_session_clear(&g_session);
        g_session.ch = channels;
        g_session.sp = spatial;

        if (ane_bridge_init() != 0) {
            ane_session_clear(&g_session);
            return false;
        }

        const char * kernel_mode = getenv("ZEROLLAMA_ANE_DRAFT_KERNEL");
        if (kernel_mode && strcmp(kernel_mode, "matmul") == 0) {
            int oc = channels;
            int seq = spatial > 0 ? spatial : 1;
            if (const char * oc_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC")) {
                const int v = (int) strtol(oc_env, NULL, 10);
                if (v > 0) {
                    oc = v;
                }
            }
            if (const char * seq_env = getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_SEQ")) {
                const int v = (int) strtol(seq_env, NULL, 10);
                if (v > 0) {
                    seq = v;
                }
            }
            // ANE matmul MIL fails eval for seq<16 at ic=oc=256 (see ane-prefill-bench).
            if (seq < 16 && channels >= 128) {
                seq = 16;
            }
            NSData * wb = nil;
            if (weight_path && weight_path[0]) {
                wb = ane_load_matmul_weight_blob(weight_path, channels, oc);
            }
            if (!wb) {
                LOG_WRN("%s: matmul weight load failed path=%s ic=%d oc=%d\n",
                        __func__, weight_path ? weight_path : "(null)", channels, oc);
                ane_session_clear(&g_session);
                return false;
            }
            g_session.ioBytes = (size_t) channels * (size_t) seq * sizeof(float);
            g_session.outIoBytes = (size_t) oc * (size_t) seq * sizeof(float);
            g_session.matmulOc = oc;
            g_session.sp = seq;
            g_session.matmulDynamic = false;
            BOOL matmul_ok = NO;
            if (ane_compile_matmul_kernel(&g_session.kernel, wb, channels, oc, seq, g_session.ioBytes, g_session.outIoBytes)) {
                matmul_ok = YES;
            } else {
                LOG_WRN("%s: static matmul compile failed ic=%d oc=%d seq=%d — trying dynamic MIL\n",
                        __func__, channels, oc, seq);
                ane_bridge_free(g_session.kernel);
                g_session.kernel = NULL;
                size_t wcount = 0;
                if (!ane_load_matmul_weight_matrix(weight_path, channels, oc, &g_session.matmulW, &wcount)) {
                    LOG_WRN("%s: dynamic matmul weight load failed path=%s\n",
                            __func__, weight_path ? weight_path : "(null)");
                    ane_session_clear(&g_session);
                    return false;
                }
                g_session.matmulWCount = wcount;
                const int spIn = seq + oc;
                g_session.ioBytes = (size_t) channels * (size_t) spIn * sizeof(float);
                g_session.matmulDynamic = true;
                matmul_ok = ane_compile_dynamic_matmul_kernel(
                    &g_session.kernel, channels, oc, seq, g_session.ioBytes, g_session.outIoBytes);
            }
            if (!matmul_ok) {
                LOG_WRN("%s: matmul compile failed ic=%d oc=%d seq=%d in=%zu out=%zu dynamic=%d\n",
                        __func__, channels, oc, seq, g_session.ioBytes, g_session.outIoBytes,
                        g_session.matmulDynamic ? 1 : 0);
                ane_session_clear(&g_session);
                return false;
            }
            g_session.inSID = ane_bridge_input_surface_id(g_session.kernel, 0);
            if (g_session.inSID == 0) {
                ane_session_clear(&g_session);
                return false;
            }
            g_session.matmulActive = true;
            g_session.outBuf = (float *) calloc((size_t) oc * (size_t) seq, sizeof(float));
            if (!g_session.outBuf) {
                ane_session_clear(&g_session);
                return false;
            }
            g_session.device = MTLCreateSystemDefaultDevice();
            if (!g_session.device) {
                ane_session_clear(&g_session);
                return false;
            }
            NSError * merr = nil;
            g_session.pipeline = ane_build_fill_pipeline(g_session.device, &merr);
            if (!g_session.pipeline) {
                ane_session_clear(&g_session);
                return false;
            }
            g_session.queue = [g_session.device newCommandQueue];
            if (g_session.matmulDynamic && !ane_session_prime_matmul_weights()) {
                LOG_WRN("%s: dynamic matmul weight prime failed ic=%d oc=%d\n", __func__, channels, oc);
                ane_session_clear(&g_session);
                return false;
            }
            g_session.matmulChain = ane_matmul_chain_target();
            if (g_session.matmulChain == 17) {
                g_session.dflashFcActive = true;
                g_session.dflashChain11Active = true;
                if (!ane_session_init_dflash_chain17(seq)) {
                    LOG_WRN("%s: P16 dflash chain17 init failed\n", __func__);
                    ane_session_clear(&g_session);
                    return false;
                }
                LOG_INF("%s: P16 dflash chain17 active (… + output_norm + tied-embed lm_head)\n", __func__);
                return YES;
            }
            if (g_session.matmulChain == 16) {
                g_session.dflashFcActive = true;
                g_session.dflashChain11Active = true;
                if (!ane_session_init_dflash_chain16(seq)) {
                    LOG_WRN("%s: P15 dflash chain16 init failed\n", __func__);
                    ane_session_clear(&g_session);
                    return false;
                }
                LOG_INF("%s: P15 dflash chain16 active (… + blk.0 ffn_gate/up/SwiGLU/down)\n", __func__);
                return YES;
            }
            if (g_session.matmulChain == 15) {
                g_session.dflashFcActive = true;
                g_session.dflashChain11Active = true;
                if (!ane_session_init_dflash_chain15(seq)) {
                    LOG_WRN("%s: P14 dflash chain15 init failed\n", __func__);
                    ane_session_clear(&g_session);
                    return false;
                }
                LOG_INF("%s: P14 dflash chain15 active (… + blk.0 ffn_gate)\n", __func__);
                return YES;
            }
            if (g_session.matmulChain == 14) {
                g_session.dflashFcActive = true;
                g_session.dflashChain11Active = true;
                if (!ane_session_init_dflash_chain14(seq)) {
                    LOG_WRN("%s: P13 dflash chain14 init failed\n", __func__);
                    ane_session_clear(&g_session);
                    return false;
                }
                LOG_INF("%s: P13 dflash chain14 active (fc + attn_q/k/v + host cross-attn + attn_wo)\n", __func__);
                return YES;
            }
            if (g_session.matmulChain == 13) {
                g_session.dflashFcActive = true;
                g_session.dflashChain11Active = true;
                if (!ane_session_init_dflash_chain13(seq)) {
                    LOG_WRN("%s: P12 dflash chain13 init failed\n", __func__);
                    ane_session_clear(&g_session);
                    return false;
                }
                LOG_INF("%s: P12 dflash chain13 active (fc + hidden_norm + attn_q/k/v + host cross-attn)\n", __func__);
                return YES;
            }
            if (g_session.matmulChain == 12) {
                g_session.dflashFcActive = true;
                g_session.dflashChain11Active = true;
                if (!ane_session_init_dflash_chain12(seq)) {
                    LOG_WRN("%s: P11 dflash chain12 init failed\n", __func__);
                    ane_session_clear(&g_session);
                    return false;
                }
                LOG_INF("%s: P11 dflash chain12 active (fc + hidden_norm + attn_q/k/v)\n", __func__);
                return YES;
            }
            if (g_session.matmulChain == 11) {
                g_session.dflashFcActive = true;
                g_session.dflashChain11Active = true;
                if (!ane_session_init_dflash_chain11(seq)) {
                    LOG_WRN("%s: P10 dflash chain11 init failed\n", __func__);
                    ane_session_clear(&g_session);
                    return false;
                }
                LOG_INF("%s: P10 dflash chain11 active (fc + hidden_norm + attn_q)\n", __func__);
                return YES;
            }
            if (g_session.matmulChain == 8) {
                g_session.dflashFcActive = true;
                LOG_INF("%s: P7b dflash_fc matmul target_hidden@W ic=%d→oc=%d seq=%d (ctx_tgt handoff)\n",
                        __func__, channels, oc, seq);
                return YES;
            }
            if (!ane_session_init_matmul2(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul2 chain init failed\n", __func__);
                ane_session_clear(&g_session);
                return false;
            }
            if (!ane_session_init_matmul3(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul3 chain init failed\n", __func__);
                ane_session_clear(&g_session);
                return false;
            }
            if (!ane_session_init_matmul4(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul4 chain init failed\n", __func__);
                ane_session_clear(&g_session);
                return false;
            }
            if (!ane_session_init_matmul5(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul5 chain init failed\n", __func__);
                ane_session_clear(&g_session);
                return false;
            }
            if (!ane_session_init_matmul6(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul6 qkv prefix init failed\n", __func__);
                ane_session_clear(&g_session);
                return false;
            }
            if (!ane_session_init_matmul7(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul7 blk.1 gate init failed\n", __func__);
                ane_session_clear(&g_session);
                return false;
            }
            if (!ane_session_init_matmul9(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul9 blk.1 up init failed — continuing without P8 (chain 7)\n", __func__);
                if (g_session.matmulChain >= 9) {
                    g_session.matmulChain = 7;
                }
            }
            if (!ane_session_init_matmul10(seq, g_session.matmulChain)) {
                LOG_WRN("%s: matmul10 blk.1 down init failed — continuing without P9 (chain 9)\n", __func__);
                if (g_session.matmulChain >= 10) {
                    g_session.matmulChain = 9;
                }
            }
            return YES;
        }

        const BOOL use_gamma = NO; // B3: gamma applied in ggml handoff pack (ANE mul broadcast unstable in MIL)
        const char * weight2_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * weight3_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * weight4_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        const char * weight5_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        const char * weight6_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
        const char * weight7_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        const char * weight8_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
        const BOOL want_conv2_raw = weight2_path && weight2_path[0];
        const BOOL want_conv3_raw = weight3_path && weight3_path[0];
        const BOOL want_conv4_raw = weight4_path && weight4_path[0];
        const BOOL want_conv5_raw = weight5_path && weight5_path[0];
        const BOOL want_conv6_raw = weight6_path && weight6_path[0];
        const BOOL want_conv7_raw = weight7_path && weight7_path[0];
        const BOOL want_conv8_raw = weight8_path && weight8_path[0];
        const int depth_cap = ane_conv_depth_cap();
        BOOL want_conv2 = want_conv2_raw;
        BOOL want_conv3 = want_conv3_raw;
        BOOL want_conv4 = want_conv4_raw;
        BOOL want_conv5 = want_conv5_raw;
        BOOL want_conv6 = want_conv6_raw;
        BOOL want_conv7 = want_conv7_raw;
        BOOL want_conv8 = want_conv8_raw;
        if (depth_cap > 0) {
            if (depth_cap < 2) { want_conv2 = NO; }
            if (depth_cap < 3) { want_conv3 = NO; }
            if (depth_cap < 4) { want_conv4 = NO; }
            if (depth_cap < 5) { want_conv5 = NO; }
            if (depth_cap < 6) { want_conv6 = NO; }
            if (depth_cap < 7) { want_conv7 = NO; }
            if (depth_cap < 8) { want_conv8 = NO; }
        }
        NSData * wb = nil;
        NSData * wb2 = nil;
        NSData * wb3 = nil;
        NSData * wb4 = nil;
        NSData * wb5 = nil;
        NSData * wb6 = nil;
        NSData * wb7 = nil;
        NSData * wb8 = nil;
        if (weight_path && weight_path[0]) {
            wb = ane_load_weight_blob(weight_path, channels);
        } else {
            wb = ane_build_weight_blob(channels);
        }
        if (want_conv2) {
            wb2 = ane_load_weight_blob(weight2_path, channels);
        }
        if (want_conv3) {
            wb3 = ane_load_weight_blob(weight3_path, channels);
        }
        if (want_conv4) {
            wb4 = ane_load_weight_blob(weight4_path, channels);
        }
        if (want_conv5) {
            wb5 = ane_load_weight_blob(weight5_path, channels);
        }
        if (want_conv6) {
            wb6 = ane_load_weight_blob(weight6_path, channels);
        }
        if (want_conv7) {
            wb7 = ane_load_weight_blob(weight7_path, channels);
        }
        if (want_conv8) {
            wb8 = ane_load_weight_blob(weight8_path, channels);
        }
        (void) use_gamma;
        (void) gamma_path;
        if (!wb || (want_conv2 && !wb2) || (want_conv3 && !wb3) || (want_conv4 && !wb4) || (want_conv5 && !wb5) || (want_conv6 && !wb6) || (want_conv7 && !wb7) || (want_conv8 && !wb8)) {
            ane_session_clear(&g_session);
            return false;
        }

        g_session.ioBytes = (size_t) channels * (size_t) spatial * sizeof(float);

        // Why dual conv1 kernels: chained conv2 MIL fails ANECCompile on device; two evals match math.
        if (!ane_compile_conv_kernel(&g_session.kernel, wb, channels, spatial, g_session.ioBytes)) {
            ane_session_clear(&g_session);
            return false;
        }
        g_session.inSID = ane_bridge_input_surface_id(g_session.kernel, 0);
        if (g_session.inSID == 0) {
            ane_session_clear(&g_session);
            return false;
        }

        g_session.conv2Active = false;
        if (want_conv2 && ane_compile_conv_kernel(&g_session.kernel2, wb2, channels, spatial, g_session.ioBytes)) {
            g_session.inSID2 = ane_bridge_input_surface_id(g_session.kernel2, 0);
            if (g_session.inSID2 != 0) {
                g_session.conv2Active = true;
            } else {
                ane_bridge_free(g_session.kernel2);
                g_session.kernel2 = NULL;
            }
        }

        g_session.conv3Active = false;
        if (want_conv3 && ane_compile_conv_kernel(&g_session.kernel3, wb3, channels, spatial, g_session.ioBytes)) {
            g_session.inSID3 = ane_bridge_input_surface_id(g_session.kernel3, 0);
            if (g_session.inSID3 != 0) {
                g_session.conv3Active = true;
            } else {
                ane_bridge_free(g_session.kernel3);
                g_session.kernel3 = NULL;
            }
        }

        g_session.conv4Active = false;
        if (want_conv4 && ane_compile_conv_kernel(&g_session.kernel4, wb4, channels, spatial, g_session.ioBytes)) {
            g_session.inSID4 = ane_bridge_input_surface_id(g_session.kernel4, 0);
            if (g_session.inSID4 != 0) {
                g_session.conv4Active = true;
            } else {
                ane_bridge_free(g_session.kernel4);
                g_session.kernel4 = NULL;
            }
        }

        g_session.conv5Active = false;
        if (want_conv5 && ane_compile_conv_kernel(&g_session.kernel5, wb5, channels, spatial, g_session.ioBytes)) {
            g_session.inSID5 = ane_bridge_input_surface_id(g_session.kernel5, 0);
            if (g_session.inSID5 != 0) {
                g_session.conv5Active = true;
            } else {
                ane_bridge_free(g_session.kernel5);
                g_session.kernel5 = NULL;
            }
        }

        g_session.conv6Active = false;
        if (want_conv6 && ane_compile_conv_kernel(&g_session.kernel6, wb6, channels, spatial, g_session.ioBytes)) {
            g_session.inSID6 = ane_bridge_input_surface_id(g_session.kernel6, 0);
            if (g_session.inSID6 != 0) {
                g_session.conv6Active = true;
            } else {
                ane_bridge_free(g_session.kernel6);
                g_session.kernel6 = NULL;
            }
        }

        g_session.conv7Active = false;
        if (want_conv7 && ane_compile_conv_kernel(&g_session.kernel7, wb7, channels, spatial, g_session.ioBytes)) {
            g_session.inSID7 = ane_bridge_input_surface_id(g_session.kernel7, 0);
            if (g_session.inSID7 != 0) {
                g_session.conv7Active = true;
            } else {
                ane_bridge_free(g_session.kernel7);
                g_session.kernel7 = NULL;
            }
        }

        g_session.conv8Active = false;
        if (want_conv8 && ane_compile_conv_kernel(&g_session.kernel8, wb8, channels, spatial, g_session.ioBytes)) {
            g_session.inSID8 = ane_bridge_input_surface_id(g_session.kernel8, 0);
            if (g_session.inSID8 != 0) {
                g_session.conv8Active = true;
            } else {
                ane_bridge_free(g_session.kernel8);
                g_session.kernel8 = NULL;
            }
        }

        g_session.outBuf = (float *) calloc((size_t) channels * (size_t) spatial, sizeof(float));
        if (!g_session.outBuf) {
            ane_session_clear(&g_session);
            return false;
        }

        g_session.device = MTLCreateSystemDefaultDevice();
        if (!g_session.device) {
            ane_session_clear(&g_session);
            return false;
        }

        NSError * merr = nil;
        g_session.pipeline = ane_build_fill_pipeline(g_session.device, &merr);
        if (!g_session.pipeline) {
            ane_session_clear(&g_session);
            return false;
        }
        g_session.queue = [g_session.device newCommandQueue];
        return YES;
    }
}

bool ane_draft_session_ready(void) {
    return g_session.kernel != NULL && g_session.inSID != 0;
}

uint32_t ane_draft_session_surface_id(void) {
    return g_session.inSID;
}

size_t ane_draft_session_surface_bytes(void) {
    return g_session.ioBytes;
}

static bool ane_session_map_fill(float fillVal) {
    IOSurfaceRef surface = ane_ggml_surface_from_id(g_session.inSID);
    if (!surface) {
        return false;
    }

    IOSurfaceLock(surface, 0, NULL);
    void * base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return false;
    }

    void * mappedBase = NULL;
    size_t mappedSize = 0;
    id<MTLBuffer> mapped = ane_ggml_map_iosurface_base(
        g_session.device, base, g_session.ioBytes, &mappedBase, &mappedSize);
    if (!mapped) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return false;
    }

    size_t pageOffset = (size_t) ((char *) base - (char *) mappedBase);
    BOOL ok = ane_metal_fill(g_session.device, g_session.queue, g_session.pipeline,
                             mapped, g_session.ioBytes, pageOffset, fillVal);

    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    return ok ? true : false;
}

static bool ane_session_eval_matmul_chain3(int seq) {
    const int oc1 = g_session.matmulOc;
    const int ic2 = g_session.matmul2Ic;
    const int oc2 = g_session.matmul2Oc;
    const int ic3 = g_session.matmul3Ic;
    const int oc3 = g_session.matmul3Oc;
    const size_t gate_bytes = (size_t) oc1 * (size_t) seq * sizeof(float);
    const size_t up_bytes = (size_t) oc2 * (size_t) seq * sizeof(float);

    if (!ane_bridge_eval(g_session.kernel)) {
        return false;
    }
    std::vector<float> gate((size_t) oc1 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel, 0, gate.data(), gate_bytes);

    std::vector<float> hidden((size_t) ic2, 0.f);
    bool have_hidden = false;
    if (g_session.lastHidden && g_session.lastHiddenLen >= ic2) {
        for (int i = 0; i < ic2; ++i) {
            hidden[(size_t) i] = g_session.lastHidden[i];
        }
        have_hidden = true;
    }
    if (!have_hidden) {
        have_hidden = ane_session_read_matmul_activations(
            g_session.inSID, hidden.data(), ic2, ic2, oc1, seq);
    }
    if (!have_hidden) {
        return false;
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID2, hidden.data(), ic2,
            ic2, oc2, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel2)) {
        return false;
    }
    std::vector<float> up((size_t) oc2 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel2, 0, up.data(), up_bytes);

    std::vector<float> swiglu((size_t) ic3);
    for (int i = 0; i < ic3; ++i) {
        double g_sum = 0.0;
        double u_sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            g_sum += (double) gate[(size_t) i * (size_t) seq + (size_t) s];
            u_sum += (double) up[(size_t) i * (size_t) seq + (size_t) s];
        }
        const float g_avg = (float) (g_sum / (double) (seq > 0 ? seq : 1));
        const float u_avg = (float) (u_sum / (double) (seq > 0 ? seq : 1));
        swiglu[(size_t) i] = ane_silu(g_avg) * u_avg;
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID3, swiglu.data(), ic3, ic3, oc3, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel3)) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel3, 0, g_session.outBuf, g_session.outIoBytes);
    g_session.stepCount++;
    return true;
}

static void ane_session_stash_down_from_buf(const float * down_buf, int oc3, int seq) {
    if (!down_buf || oc3 <= 0 || !g_session.lastDownHidden || g_session.lastDownHiddenLen < oc3) {
        return;
    }
    for (int o = 0; o < oc3; ++o) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) down_buf[(size_t) o * (size_t) seq + (size_t) s];
        }
        g_session.lastDownHidden[o] = (float) (sum / (double) (seq > 0 ? seq : 1));
    }
}

static bool ane_session_eval_matmul_chain4(int seq) {
    const int oc1 = g_session.matmulOc;
    const int ic2 = g_session.matmul2Ic;
    const int oc2 = g_session.matmul2Oc;
    const int ic3 = g_session.matmul3Ic;
    const int oc3 = g_session.matmul3Oc;
    const int ic4 = g_session.matmul4Ic;
    const int oc4 = g_session.matmul4Oc;
    const size_t gate_bytes = (size_t) oc1 * (size_t) seq * sizeof(float);
    const size_t up_bytes = (size_t) oc2 * (size_t) seq * sizeof(float);
    const size_t down_bytes = (size_t) oc3 * (size_t) seq * sizeof(float);

    if (!ane_bridge_eval(g_session.kernel)) {
        return false;
    }
    std::vector<float> gate((size_t) oc1 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel, 0, gate.data(), gate_bytes);

    std::vector<float> hidden((size_t) ic2, 0.f);
    bool have_hidden = false;
    if (g_session.lastHidden && g_session.lastHiddenLen >= ic2) {
        for (int i = 0; i < ic2; ++i) {
            hidden[(size_t) i] = g_session.lastHidden[i];
        }
        have_hidden = true;
    }
    if (!have_hidden) {
        have_hidden = ane_session_read_matmul_activations(
            g_session.inSID, hidden.data(), ic2, ic2, oc1, seq);
    }
    if (!have_hidden) {
        return false;
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID2, hidden.data(), ic2, ic2, oc2, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel2)) {
        return false;
    }
    std::vector<float> up((size_t) oc2 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel2, 0, up.data(), up_bytes);

    std::vector<float> swiglu((size_t) ic3);
    for (int i = 0; i < ic3; ++i) {
        double g_sum = 0.0;
        double u_sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            g_sum += (double) gate[(size_t) i * (size_t) seq + (size_t) s];
            u_sum += (double) up[(size_t) i * (size_t) seq + (size_t) s];
        }
        const float g_avg = (float) (g_sum / (double) (seq > 0 ? seq : 1));
        const float u_avg = (float) (u_sum / (double) (seq > 0 ? seq : 1));
        swiglu[(size_t) i] = ane_silu(g_avg) * u_avg;
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID3, swiglu.data(), ic3, ic3, oc3, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel3)) {
        return false;
    }
    std::vector<float> down((size_t) oc3 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel3, 0, down.data(), down_bytes);
    ane_session_stash_down_from_buf(down.data(), oc3, seq);

    if (!ane_session_pack_matmul2_activations(
            g_session.inSID4, g_session.lastDownHidden, oc3, ic4, oc4, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel4)) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel4, 0, g_session.outBuf, g_session.outIoBytes);
    g_session.stepCount++;
    return true;
}

static void ane_session_stash_qkv_from_buf(const float * qkv_buf, int oc6, int seq) {
    if (!qkv_buf || oc6 <= 0 || !g_session.lastQkvHidden || g_session.lastQkvHiddenLen < oc6) {
        return;
    }
    for (int o = 0; o < oc6; ++o) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) qkv_buf[(size_t) o * (size_t) seq + (size_t) s];
        }
        g_session.lastQkvHidden[o] = (float) (sum / (double) (seq > 0 ? seq : 1));
    }
}

static bool ane_session_eval_qkv_prefix(int seq) {
    if (!g_session.matmul6Active || !g_session.kernel6) {
        return true;
    }
    const int ic6 = g_session.matmul6Ic;
    const int oc6 = g_session.matmul6Oc;
    std::vector<float> hidden((size_t) ic6, 0.f);
    bool have_hidden = false;
    if (g_session.lastHidden && g_session.lastHiddenLen >= ic6) {
        for (int i = 0; i < ic6; ++i) {
            hidden[(size_t) i] = g_session.lastHidden[i];
        }
        have_hidden = true;
    }
    if (!have_hidden) {
        have_hidden = ane_session_read_matmul_activations(
            g_session.inSID, hidden.data(), ic6, ic6, g_session.matmulOc, seq);
    }
    if (!have_hidden) {
        return false;
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID6, hidden.data(), ic6, ic6, oc6, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel6)) {
        return false;
    }
    const size_t qkv_bytes = (size_t) oc6 * (size_t) seq * sizeof(float);
    std::vector<float> qkv((size_t) oc6 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel6, 0, qkv.data(), qkv_bytes);
    ane_session_stash_qkv_from_buf(qkv.data(), oc6, seq);
    return true;
}

static bool ane_session_eval_matmul_chain5(int seq) {
    if (!ane_session_eval_qkv_prefix(seq)) {
        return false;
    }
    const int oc1 = g_session.matmulOc;
    const int ic2 = g_session.matmul2Ic;
    const int oc2 = g_session.matmul2Oc;
    const int ic3 = g_session.matmul3Ic;
    const int oc3 = g_session.matmul3Oc;
    const int ic4 = g_session.matmul4Ic;
    const int oc4 = g_session.matmul4Oc;
    const int ic5 = g_session.matmul5Ic;
    const int oc5 = g_session.matmul5Oc;
    const size_t gate_bytes = (size_t) oc1 * (size_t) seq * sizeof(float);
    const size_t up_bytes = (size_t) oc2 * (size_t) seq * sizeof(float);
    const size_t down_bytes = (size_t) oc3 * (size_t) seq * sizeof(float);
    const size_t attn_bytes = (size_t) oc4 * (size_t) seq * sizeof(float);

    if (!ane_bridge_eval(g_session.kernel)) {
        return false;
    }
    std::vector<float> gate((size_t) oc1 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel, 0, gate.data(), gate_bytes);

    std::vector<float> hidden((size_t) ic2, 0.f);
    bool have_hidden = false;
    if (g_session.lastHidden && g_session.lastHiddenLen >= ic2) {
        for (int i = 0; i < ic2; ++i) {
            hidden[(size_t) i] = g_session.lastHidden[i];
        }
        have_hidden = true;
    }
    if (!have_hidden) {
        have_hidden = ane_session_read_matmul_activations(
            g_session.inSID, hidden.data(), ic2, ic2, oc1, seq);
    }
    if (!have_hidden) {
        return false;
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID2, hidden.data(), ic2, ic2, oc2, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel2)) {
        return false;
    }
    std::vector<float> up((size_t) oc2 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel2, 0, up.data(), up_bytes);

    std::vector<float> swiglu((size_t) ic3);
    for (int i = 0; i < ic3; ++i) {
        double g_sum = 0.0;
        double u_sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            g_sum += (double) gate[(size_t) i * (size_t) seq + (size_t) s];
            u_sum += (double) up[(size_t) i * (size_t) seq + (size_t) s];
        }
        const float g_avg = (float) (g_sum / (double) (seq > 0 ? seq : 1));
        const float u_avg = (float) (u_sum / (double) (seq > 0 ? seq : 1));
        swiglu[(size_t) i] = ane_silu(g_avg) * u_avg;
    }
    if (g_session.matmul10Active && g_session.matmul3W && g_session.inSID3 != 0) {
        if (!ane_session_prime_matmul3_weights(g_session.inSID3, ic3, oc3, seq)) {
            return false;
        }
    }
    if (const char * tel = getenv("ZEROLLAMA_ANE_DRAFT_TELEMETRY"); tel && tel[0] && strcmp(tel, "0") != 0) {
        double sw_n = 0.0;
        for (float v : swiglu) {
            sw_n += (double) v * (double) v;
        }
        LOG_INF("%s: P3 blk.0 swiglu_n=%.6e ic=%d seq=%d\n",
                __func__, std::sqrt(sw_n), ic3, seq);
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID3, swiglu.data(), ic3, ic3, oc3, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel3)) {
        return false;
    }
    std::vector<float> down((size_t) oc3 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel3, 0, down.data(), down_bytes);
    if (const char * tel = getenv("ZEROLLAMA_ANE_DRAFT_TELEMETRY"); tel && tel[0] && strcmp(tel, "0") != 0) {
        double down_n = 0.0;
        for (float v : down) {
            down_n += (double) v * (double) v;
        }
        LOG_INF("%s: P3 blk.0 ffn_down out_n=%.6e ic=%d oc=%d seq=%d\n",
                __func__, std::sqrt(down_n), ic3, oc3, seq);
    }
    ane_session_stash_down_from_buf(down.data(), oc3, seq);

    if (!ane_session_pack_matmul2_activations(
            g_session.inSID4, g_session.lastDownHidden, oc3, ic4, oc4, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel4)) {
        return false;
    }
    std::vector<float> attn((size_t) oc4 * (size_t) seq);
    ane_bridge_read_output(g_session.kernel4, 0, attn.data(), attn_bytes);
    (void) attn;

    if (!ane_session_pack_matmul2_activations(
            g_session.inSID5, g_session.lastDownHidden, oc3, ic5, oc5, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel5)) {
        return false;
    }
    if (!g_session.matmul7Active || !g_session.kernel7) {
        ane_bridge_read_output(g_session.kernel5, 0, g_session.outBuf, g_session.outIoBytes);
        g_session.stepCount++;
        return true;
    }
    if (!g_session.lastDownHidden || g_session.lastDownHiddenLen < ic3) {
        return false;
    }
    const int ic7 = g_session.matmul7Ic;
    const int oc7 = g_session.matmul7Oc;
    const int ic9 = g_session.matmul9Ic;
    const int oc9 = g_session.matmul9Oc;
    const size_t gate7_bytes = (size_t) oc7 * (size_t) seq * sizeof(float);
    std::vector<float> gate7((size_t) oc7 * (size_t) seq);
    // P7/P8: host fp32 blk.1 gate+up — ANE fp16 ~0.58 B6 cos on lab qwen35 slice.
    if (g_session.matmul7W && g_session.matmul7WCount >= (size_t) ic7 * (size_t) oc7) {
        ane_host_matmul_seq(g_session.lastDownHidden, ic7, oc7, seq, g_session.matmul7W, gate7.data());
    } else {
        if (!ane_session_pack_matmul2_activations(
                g_session.inSID7, g_session.lastDownHidden, ic3, ic7, oc7, seq)) {
            return false;
        }
        if (!ane_bridge_eval(g_session.kernel7)) {
            return false;
        }
        ane_bridge_read_output(g_session.kernel7, 0, gate7.data(), gate7_bytes);
    }
    if (!g_session.matmul9Active || !g_session.kernel8) {
        if (g_session.outIoBytes >= gate7_bytes) {
            memcpy(g_session.outBuf, gate7.data(), gate7_bytes);
        }
        g_session.stepCount++;
        return true;
    }
    if (!g_session.lastDownHidden || g_session.lastDownHiddenLen < ic3) {
        return false;
    }
    const size_t up9_bytes = (size_t) oc9 * (size_t) seq * sizeof(float);
    std::vector<float> up9((size_t) oc9 * (size_t) seq);
    if (g_session.matmul9W && g_session.matmul9WCount >= (size_t) ic9 * (size_t) oc9) {
        ane_host_matmul_seq(g_session.lastDownHidden, ic9, oc9, seq, g_session.matmul9W, up9.data());
    } else {
        if (!ane_session_pack_matmul2_activations(
                g_session.inSID8, g_session.lastDownHidden, ic3, ic9, oc9, seq)) {
            return false;
        }
        if (!ane_bridge_eval(g_session.kernel8)) {
            return false;
        }
        ane_bridge_read_output(g_session.kernel8, 0, up9.data(), up9_bytes);
    }

    std::vector<float> swiglu1((size_t) oc9);
    for (int o = 0; o < oc9; ++o) {
        double g_sum = 0.0;
        double u_sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            g_sum += (double) gate7[(size_t) o * (size_t) seq + (size_t) s];
            u_sum += (double) up9[(size_t) o * (size_t) seq + (size_t) s];
        }
        const float g_avg = (float) (g_sum / (double) (seq > 0 ? seq : 1));
        const float u_avg = (float) (u_sum / (double) (seq > 0 ? seq : 1));
        swiglu1[(size_t) o] = ane_silu(g_avg) * u_avg;
    }
    if (const char * tel = getenv("ZEROLLAMA_ANE_DRAFT_TELEMETRY"); tel && tel[0] && strcmp(tel, "0") != 0) {
        double sw_n = 0.0;
        for (float v : swiglu1) {
            sw_n += (double) v * (double) v;
        }
        LOG_INF("%s: P9 blk.1 swiglu1_n=%.6e ic=%d seq=%d\n",
                __func__, std::sqrt(sw_n), oc9, seq);
    }
    if (!g_session.matmul10Active) {
        for (int o = 0; o < oc9; ++o) {
            for (int s = 0; s < seq; ++s) {
                g_session.outBuf[(size_t) o * (size_t) seq + (size_t) s] = swiglu1[(size_t) o];
            }
        }
        g_session.stepCount++;
        return true;
    }
    const int oc10 = g_session.matmul10Oc;
    const int ic10 = g_session.matmul10Ic;
    if (!g_session.matmul10W || g_session.matmul10WCount < (size_t) ic10 * (size_t) oc10) {
        return false;
    }
    // blk.1 SwiGLU activations are ~1e-3 vs ~3 for blk.0; ANE fp16 matmul underflows to zero.
    ane_host_matmul_seq(swiglu1.data(), ic10, oc10, seq, g_session.matmul10W, g_session.outBuf);
    double out_n = 0.0;
    const size_t nout = g_session.outIoBytes / sizeof(float);
    for (size_t i = 0; i < nout; ++i) {
        out_n += (double) g_session.outBuf[i] * (double) g_session.outBuf[i];
    }
    if (const char * tel = getenv("ZEROLLAMA_ANE_DRAFT_TELEMETRY"); tel && tel[0] && strcmp(tel, "0") != 0) {
        double sw_n = 0.0;
        for (float v : swiglu1) {
            sw_n += (double) v * (double) v;
        }
        LOG_INF("%s: P9 eval host_fp32 swiglu1_n=%.6e out_n=%.6e ic=%d oc=%d seq=%d\n",
                __func__, std::sqrt(sw_n), std::sqrt(out_n), ic10, oc10, seq);
    }
    ane_session_stash_down_from_buf(g_session.outBuf, oc10, seq);
    g_session.stepCount++;
    return true;
}

static void ane_apply_rms_gamma(float * h, int n, const float * gamma) {
    if (!h || n <= 0) {
        return;
    }
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_sq += (double) h[i] * (double) h[i];
    }
    const float inv_rms = 1.0f / sqrtf((float) (sum_sq / (double) n) + 1e-6f);
    for (int i = 0; i < n; ++i) {
        const float g = (gamma && i < n) ? gamma[i] : 1.0f;
        h[i] *= inv_rms * g;
    }
}

static bool ane_session_load_hidden_norm_gamma(int n) {
    const char * path = std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE");
    if (!path || !path[0] || n <= 0) {
        return true;
    }
    if (g_session.hiddenNormGamma && g_session.hiddenNormGammaLen == n) {
        return true;
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    const size_t blob_expected = 64 + 64 + (size_t) n * 2;
    const size_t f32_expected = (size_t) n * sizeof(float);
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::rewind(f);
    if (sz < 0) {
        std::fclose(f);
        return false;
    }
    if (g_session.hiddenNormGamma) {
        free(g_session.hiddenNormGamma);
        g_session.hiddenNormGamma = NULL;
    }
    g_session.hiddenNormGamma = (float *) calloc((size_t) n, sizeof(float));
    if (!g_session.hiddenNormGamma) {
        std::fclose(f);
        return false;
    }
    g_session.hiddenNormGammaLen = n;
    if ((size_t) sz == f32_expected) {
        if (std::fread(g_session.hiddenNormGamma, sizeof(float), (size_t) n, f) != (size_t) n) {
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        return true;
    }
    if ((size_t) sz != blob_expected) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(blob_expected);
    if (std::fread(buf.data(), 1, blob_expected, f) != blob_expected) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    const uint8_t * fp16 = buf.data() + 128;
    for (int i = 0; i < n; ++i) {
        uint16_t bits = (uint16_t) fp16[i * 2] | ((uint16_t) fp16[i * 2 + 1] << 8);
        g_session.hiddenNormGamma[i] = ane_fp16_to_f32(bits);
    }
    return true;
}

static bool ane_session_init_dflash_chain11(int seq) {
    const char * w2path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
    if (!w2path || !w2path[0] || !g_session.matmulDynamic) {
        return false;
    }
    const int ic2 = g_session.matmulOc;
    int oc2 = g_session.ch;
    if (const char * oc2_env = std::getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2")) {
        const int v = (int) strtol(oc2_env, NULL, 10);
        if (v > 0) {
            oc2 = v;
        }
    }
    if (ic2 <= 0 || oc2 <= 0) {
        return false;
    }
    if (!ane_session_load_hidden_norm_gamma(ic2)) {
        LOG_WRN("%s: hidden_norm gamma load failed n=%d\n", __func__, ic2);
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w2path, ic2, oc2, &g_session.matmul2W, &wcount)) {
        LOG_WRN("%s: matmul11 attn_q load failed path=%s ic=%d oc=%d\n", __func__, w2path, ic2, oc2);
        return false;
    }
    g_session.matmul2WCount = wcount;
    g_session.matmul2Ic = ic2;
    g_session.matmul2Oc = oc2;
    const int spIn2 = seq + oc2;
    g_session.ioBytes2 = (size_t) ic2 * (size_t) spIn2 * sizeof(float);
    g_session.outIoBytes = (size_t) oc2 * (size_t) seq * sizeof(float);
    g_session.matmul2Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel2, ic2, oc2, seq, g_session.ioBytes2, g_session.outIoBytes)) {
        LOG_WRN("%s: matmul11 attn_q compile failed ic=%d oc=%d seq=%d\n", __func__, ic2, oc2, seq);
        return false;
    }
    g_session.inSID2 = ane_bridge_input_surface_id(g_session.kernel2, 0);
    if (g_session.inSID2 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul2_weights(g_session.inSID2, ic2, oc2, seq)) {
        LOG_WRN("%s: matmul11 attn_q prime failed ic=%d oc=%d\n", __func__, ic2, oc2);
        return false;
    }
    g_session.matmul2Active = true;
    if (g_session.outBuf) {
        free(g_session.outBuf);
    }
    g_session.outBuf = (float *) calloc((size_t) oc2 * (size_t) seq, sizeof(float));
    if (!g_session.outBuf) {
        return false;
    }
    LOG_INF("%s: P10 dflash chain11 dflash_fc→hidden_norm→attn_q ic=%d→oc=%d seq=%d\n",
            __func__, ic2, oc2, seq);
    return true;
}

static bool ane_session_eval_dflash_chain11(int seq) {
    if (!g_session.kernel || !g_session.kernel2) {
        return false;
    }
    const int oc_fc = g_session.matmulOc;
    const int oc_q  = g_session.matmul2Oc;
    if (!ane_bridge_eval(g_session.kernel)) {
        return false;
    }
    const size_t fc_bytes = (size_t) oc_fc * (size_t) seq * sizeof(float);
    std::vector<float> fc_sp((size_t) oc_fc * (size_t) seq);
    ane_bridge_read_output(g_session.kernel, 0, fc_sp.data(), fc_bytes);
    std::vector<float> fc((size_t) oc_fc);
    for (int i = 0; i < oc_fc; ++i) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) fc_sp[(size_t) i * (size_t) seq + (size_t) s];
        }
        fc[(size_t) i] = (float) (sum / (double) (seq > 0 ? seq : 1));
    }
    ane_apply_rms_gamma(fc.data(), oc_fc, g_session.hiddenNormGamma);
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID2, fc.data(), oc_fc, oc_fc, oc_q, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel2)) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel2, 0, g_session.outBuf, g_session.outIoBytes);
    g_session.stepCount += 2;
    return true;
}

static bool ane_session_init_dflash_kv_proj(
        int seq, const char * wpath, const char * oc_env, const char * label,
        ANEKernelHandle ** kernel_out, uint32_t * sid_out,
        float ** w_out, size_t * wcount_out,
        int * ic_out, int * oc_out, size_t * iobytes_out,
        bool * active_out, bool * dynamic_out) {
    if (!wpath || !wpath[0] || !g_session.matmulDynamic) {
        return false;
    }
    const int ic = g_session.matmulOc;
    int oc = g_session.matmul2Oc;
    if (const char * oc_e = std::getenv(oc_env)) {
        const int v = (int) strtol(oc_e, NULL, 10);
        if (v > 0) {
            oc = v;
        }
    }
    if (ic <= 0 || oc <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(wpath, ic, oc, w_out, &wcount)) {
        LOG_WRN("%s: %s load failed path=%s ic=%d oc=%d\n", __func__, label, wpath, ic, oc);
        return false;
    }
    *wcount_out = wcount;
    *ic_out = ic;
    *oc_out = oc;
    const int spIn = seq + oc;
    *iobytes_out = (size_t) ic * (size_t) spIn * sizeof(float);
    const size_t outBytes = (size_t) oc * (size_t) seq * sizeof(float);
    *dynamic_out = true;
    if (!ane_compile_dynamic_matmul_kernel(kernel_out, ic, oc, seq, *iobytes_out, outBytes)) {
        LOG_WRN("%s: %s compile failed ic=%d oc=%d seq=%d\n", __func__, label, ic, oc, seq);
        return false;
    }
    *sid_out = ane_bridge_input_surface_id(*kernel_out, 0);
    if (*sid_out == 0) {
        return false;
    }
    const bool primed = (kernel_out == &g_session.kernel3)
        ? ane_session_prime_matmul3_weights(*sid_out, ic, oc, seq)
        : ane_session_prime_matmul4_weights(*sid_out, ic, oc, seq);
    if (!primed) {
        LOG_WRN("%s: %s prime failed ic=%d oc=%d\n", __func__, label, ic, oc);
        return false;
    }
    *active_out = true;
    LOG_INF("%s: P11 dflash chain12 %s ic=%d→oc=%d seq=%d\n", __func__, label, ic, oc, seq);
    return true;
}

static bool ane_session_init_dflash_chain12(int seq) {
    if (!ane_session_init_dflash_chain11(seq)) {
        return false;
    }
    const char * w3path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
    const char * w4path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
    if (!w3path || !w3path[0] || !w4path || !w4path[0]) {
        return false;
    }
    if (!ane_session_init_dflash_kv_proj(
            seq, w3path, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", "attn_k",
            &g_session.kernel3, &g_session.inSID3,
            &g_session.matmul3W, &g_session.matmul3WCount,
            &g_session.matmul3Ic, &g_session.matmul3Oc, &g_session.ioBytes3,
            &g_session.matmul3Active, &g_session.matmul3Dynamic)) {
        return false;
    }
    if (!ane_session_init_dflash_kv_proj(
            seq, w4path, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", "attn_v",
            &g_session.kernel4, &g_session.inSID4,
            &g_session.matmul4W, &g_session.matmul4WCount,
            &g_session.matmul4Ic, &g_session.matmul4Oc, &g_session.ioBytes4,
            &g_session.matmul4Active, &g_session.matmul4Dynamic)) {
        return false;
    }
    g_session.outIoBytes = (size_t) g_session.matmul2Oc * (size_t) seq * sizeof(float);
    if (g_session.outBuf) {
        free(g_session.outBuf);
    }
    g_session.outBuf = (float *) calloc((size_t) g_session.matmul2Oc * (size_t) seq, sizeof(float));
    if (!g_session.outBuf) {
        return false;
    }
    g_session.dflashChain12Active = true;
    return true;
}

static bool ane_session_eval_dflash_chain12(int seq) {
    if (!g_session.kernel || !g_session.kernel2 || !g_session.kernel3 || !g_session.kernel4) {
        return false;
    }
    const int oc_fc = g_session.matmulOc;
    const int oc_q  = g_session.matmul2Oc;
    const int oc_k  = g_session.matmul3Oc;
    const int oc_v  = g_session.matmul4Oc;
    std::vector<float> norm_fc((size_t) oc_fc);
    if (g_session.dflashFcHost && g_session.dflashFcHostLen >= oc_fc) {
        for (int i = 0; i < oc_fc; ++i) {
            norm_fc[(size_t) i] = g_session.dflashFcHost[i];
        }
        free(g_session.dflashFcHost);
        g_session.dflashFcHost = NULL;
        g_session.dflashFcHostLen = 0;
    } else {
        if (!ane_bridge_eval(g_session.kernel)) {
            return false;
        }
        const size_t fc_bytes = (size_t) oc_fc * (size_t) seq * sizeof(float);
        std::vector<float> fc_sp((size_t) oc_fc * (size_t) seq);
        ane_bridge_read_output(g_session.kernel, 0, fc_sp.data(), fc_bytes);
        for (int i = 0; i < oc_fc; ++i) {
            double sum = 0.0;
            for (int s = 0; s < seq; ++s) {
                sum += (double) fc_sp[(size_t) i * (size_t) seq + (size_t) s];
            }
            norm_fc[(size_t) i] = (float) (sum / (double) (seq > 0 ? seq : 1));
        }
    }
    ane_apply_rms_gamma(norm_fc.data(), oc_fc, g_session.hiddenNormGamma);

    if (!ane_session_pack_matmul2_activations(
            g_session.inSID2, norm_fc.data(), oc_fc, oc_fc, oc_q, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel2)) {
        return false;
    }
    if (g_session.lastDflashQ && g_session.lastDflashAttnLen >= oc_q) {
        const size_t q_bytes = (size_t) oc_q * (size_t) seq * sizeof(float);
        std::vector<float> q_sp((size_t) oc_q * (size_t) seq);
        ane_bridge_read_output(g_session.kernel2, 0, q_sp.data(), q_bytes);
        for (int o = 0; o < oc_q; ++o) {
            double sum = 0.0;
            for (int s = 0; s < seq; ++s) {
                sum += (double) q_sp[(size_t) o * (size_t) seq + (size_t) s];
            }
            g_session.lastDflashQ[o] = (float) (sum / (double) (seq > 0 ? seq : 1));
        }
    }

    if (!ane_session_pack_matmul2_activations(
            g_session.inSID3, norm_fc.data(), oc_fc, oc_fc, oc_k, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel3)) {
        return false;
    }
    if (g_session.lastDflashK && g_session.lastDflashAttnLen >= oc_k) {
        const size_t k_bytes = (size_t) oc_k * (size_t) seq * sizeof(float);
        std::vector<float> k_sp((size_t) oc_k * (size_t) seq);
        ane_bridge_read_output(g_session.kernel3, 0, k_sp.data(), k_bytes);
        for (int o = 0; o < oc_k; ++o) {
            double sum = 0.0;
            for (int s = 0; s < seq; ++s) {
                sum += (double) k_sp[(size_t) o * (size_t) seq + (size_t) s];
            }
            g_session.lastDflashK[o] = (float) (sum / (double) (seq > 0 ? seq : 1));
        }
    }

    if (!ane_session_pack_matmul2_activations(
            g_session.inSID4, norm_fc.data(), oc_fc, oc_fc, oc_v, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel4)) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel4, 0, g_session.outBuf, g_session.outIoBytes);
    if (g_session.lastDflashV && g_session.lastDflashAttnLen >= oc_v) {
        for (int o = 0; o < oc_v; ++o) {
            double sum = 0.0;
            for (int s = 0; s < seq; ++s) {
                sum += (double) g_session.outBuf[(size_t) o * (size_t) seq + (size_t) s];
            }
            g_session.lastDflashV[o] = (float) (sum / (double) (seq > 0 ? seq : 1));
        }
    }
    g_session.stepCount += 4;
    return true;
}

static bool ane_session_init_dflash_chain13(int seq) {
    if (!ane_session_init_dflash_chain12(seq)) {
        return false;
    }
    const int oc_q = g_session.matmul2Oc > 0 ? g_session.matmul2Oc : g_session.matmul4Oc;
    const int oc_kv = g_session.matmul3Oc > 0 ? g_session.matmul3Oc : oc_q;
    if (oc_q <= 0) {
        return false;
    }
    g_session.lastDflashAttnLen = oc_q;
    g_session.lastDflashQ = (float *) calloc((size_t) oc_q, sizeof(float));
    g_session.lastDflashK = (float *) calloc((size_t) oc_kv, sizeof(float));
    g_session.lastDflashV = (float *) calloc((size_t) oc_kv, sizeof(float));
    if (!g_session.lastDflashQ || !g_session.lastDflashK || !g_session.lastDflashV) {
        return false;
    }
    g_session.dflashChain13Active = true;
    return true;
}

static bool ane_session_init_dflash_wo(int seq) {
    const char * w5path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
    if (!w5path || !w5path[0] || !g_session.matmulDynamic) {
        return false;
    }
    const int ic5 = g_session.matmul2Oc > 0 ? g_session.matmul2Oc : g_session.matmul4Oc;
    int oc5 = g_session.matmulOc;
    if (const char * oc5_env = std::getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5")) {
        const int v = (int) strtol(oc5_env, NULL, 10);
        if (v > 0) {
            oc5 = v;
        }
    }
    if (ic5 <= 0 || oc5 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w5path, ic5, oc5, &g_session.matmul5W, &wcount)) {
        LOG_WRN("%s: dflash attn_wo load failed path=%s ic=%d oc=%d\n", __func__, w5path, ic5, oc5);
        return false;
    }
    g_session.matmul5WCount = wcount;
    g_session.matmul5Ic = ic5;
    g_session.matmul5Oc = oc5;
    const int spIn5 = seq + oc5;
    g_session.ioBytes5 = (size_t) ic5 * (size_t) spIn5 * sizeof(float);
    const size_t woOutBytes = (size_t) oc5 * (size_t) seq * sizeof(float);
    g_session.matmul5Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel5, ic5, oc5, seq, g_session.ioBytes5, woOutBytes)) {
        LOG_WRN("%s: dflash attn_wo compile failed ic=%d oc=%d seq=%d\n", __func__, ic5, oc5, seq);
        return false;
    }
    g_session.inSID5 = ane_bridge_input_surface_id(g_session.kernel5, 0);
    if (g_session.inSID5 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul5_weights(g_session.inSID5, ic5, oc5, seq)) {
        LOG_WRN("%s: dflash attn_wo prime failed ic=%d oc=%d\n", __func__, ic5, oc5);
        return false;
    }
    g_session.matmul5Active = true;
    if (woOutBytes > g_session.outIoBytes) {
        if (g_session.outBuf) {
            free(g_session.outBuf);
        }
        g_session.outBuf = (float *) calloc((size_t) oc5 * (size_t) seq, sizeof(float));
        if (!g_session.outBuf) {
            return false;
        }
    }
    LOG_INF("%s: P13 dflash chain14 attn_wo ic=%d→oc=%d seq=%d\n", __func__, ic5, oc5, seq);
    return true;
}

static bool ane_session_init_dflash_chain14(int seq) {
    if (!ane_session_init_dflash_chain13(seq)) {
        return false;
    }
    if (!ane_session_init_dflash_wo(seq)) {
        return false;
    }
    g_session.dflashChain14Active = true;
    return true;
}

static bool ane_session_init_dflash_ffn_gate(int seq) {
    const char * w6path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
    if (!w6path || !w6path[0] || !g_session.matmulDynamic) {
        return false;
    }
    const int ic6 = g_session.matmul5Oc > 0 ? g_session.matmul5Oc : g_session.matmulOc;
    int oc6 = g_session.ch;
    if (const char * oc6_env = std::getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6")) {
        const int v = (int) strtol(oc6_env, NULL, 10);
        if (v > 0) {
            oc6 = v;
        }
    }
    if (ic6 <= 0 || oc6 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w6path, ic6, oc6, &g_session.matmul6W, &wcount)) {
        LOG_WRN("%s: dflash ffn_gate load failed path=%s ic=%d oc=%d\n", __func__, w6path, ic6, oc6);
        return false;
    }
    g_session.matmul6WCount = wcount;
    g_session.matmul6Ic = ic6;
    g_session.matmul6Oc = oc6;
    const int spIn6 = seq + oc6;
    g_session.ioBytes6 = (size_t) ic6 * (size_t) spIn6 * sizeof(float);
    const size_t gateOutBytes = (size_t) oc6 * (size_t) seq * sizeof(float);
    g_session.matmul6Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel6, ic6, oc6, seq, g_session.ioBytes6, gateOutBytes)) {
        LOG_WRN("%s: dflash ffn_gate compile failed ic=%d oc=%d seq=%d\n", __func__, ic6, oc6, seq);
        return false;
    }
    g_session.inSID6 = ane_bridge_input_surface_id(g_session.kernel6, 0);
    if (g_session.inSID6 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul6_weights(g_session.inSID6, ic6, oc6, seq)) {
        LOG_WRN("%s: dflash ffn_gate prime failed ic=%d oc=%d\n", __func__, ic6, oc6);
        return false;
    }
    g_session.matmul6Active = true;
    LOG_INF("%s: P14 dflash chain15 ffn_gate ic=%d→oc=%d seq=%d\n", __func__, ic6, oc6, seq);
    return true;
}

static bool ane_session_load_attn_post_norm_gamma(int n) {
    const char * path = std::getenv("ZEROLLAMA_ANE_DRAFT_ATTN_POST_NORM_FILE");
    if (!path || !path[0] || n <= 0) {
        return false;
    }
    if (g_session.attnPostNormGamma && g_session.attnPostNormGammaLen == n) {
        return true;
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    const size_t blob_expected = 64 + 64 + (size_t) n * 2;
    const size_t f32_expected = (size_t) n * sizeof(float);
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::rewind(f);
    if (sz < 0) {
        std::fclose(f);
        return false;
    }
    if (g_session.attnPostNormGamma) {
        free(g_session.attnPostNormGamma);
        g_session.attnPostNormGamma = NULL;
    }
    g_session.attnPostNormGamma = (float *) calloc((size_t) n, sizeof(float));
    if (!g_session.attnPostNormGamma) {
        std::fclose(f);
        return false;
    }
    g_session.attnPostNormGammaLen = n;
    if ((size_t) sz == f32_expected) {
        if (std::fread(g_session.attnPostNormGamma, sizeof(float), (size_t) n, f) != (size_t) n) {
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        LOG_INF("%s: P14 dflash attn_post_norm gamma n=%d path=%s\n", __func__, n, path);
        return true;
    }
    if ((size_t) sz != blob_expected) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(blob_expected);
    if (std::fread(buf.data(), 1, blob_expected, f) != blob_expected) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    const uint8_t * fp16 = buf.data() + 128;
    for (int i = 0; i < n; ++i) {
        uint16_t bits = (uint16_t) fp16[i * 2] | ((uint16_t) fp16[i * 2 + 1] << 8);
        g_session.attnPostNormGamma[i] = ane_fp16_to_f32(bits);
    }
    LOG_INF("%s: P14 dflash attn_post_norm gamma n=%d path=%s\n", __func__, n, path);
    return true;
}

static bool ane_session_init_dflash_chain15(int seq) {
    if (!ane_session_init_dflash_chain14(seq)) {
        return false;
    }
    const int n_embd = g_session.matmul5Oc > 0 ? g_session.matmul5Oc : g_session.matmulOc;
    if (n_embd <= 0 || !ane_session_load_attn_post_norm_gamma(n_embd)) {
        LOG_WRN("%s: P14 dflash chain15 attn_post_norm load failed n=%d\n", __func__, n_embd);
        return false;
    }
    if (!ane_session_init_dflash_ffn_gate(seq)) {
        return false;
    }
    g_session.dflashChain15Active = true;
    return true;
}

static bool ane_session_init_dflash_ffn_up(int seq) {
    const char * w7path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
    if (!w7path || !w7path[0] || !g_session.matmulDynamic) {
        return false;
    }
    const int ic7 = g_session.matmul5Oc > 0 ? g_session.matmul5Oc : g_session.matmulOc;
    int oc7 = g_session.matmul6Oc > 0 ? g_session.matmul6Oc : g_session.ch;
    if (const char * oc7_env = std::getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC7")) {
        const int v = (int) strtol(oc7_env, NULL, 10);
        if (v > 0) {
            oc7 = v;
        }
    }
    if (ic7 <= 0 || oc7 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w7path, ic7, oc7, &g_session.matmul7W, &wcount)) {
        LOG_WRN("%s: dflash ffn_up load failed path=%s ic=%d oc=%d\n", __func__, w7path, ic7, oc7);
        return false;
    }
    g_session.matmul7WCount = wcount;
    g_session.matmul7Ic = ic7;
    g_session.matmul7Oc = oc7;
    const int spIn7 = seq + oc7;
    g_session.ioBytes7 = (size_t) ic7 * (size_t) spIn7 * sizeof(float);
    const size_t upOutBytes = (size_t) oc7 * (size_t) seq * sizeof(float);
    g_session.matmul7Dynamic = true;
    if (!ane_compile_dynamic_matmul_kernel(
            &g_session.kernel7, ic7, oc7, seq, g_session.ioBytes7, upOutBytes)) {
        LOG_WRN("%s: dflash ffn_up compile failed ic=%d oc=%d seq=%d\n", __func__, ic7, oc7, seq);
        return false;
    }
    g_session.inSID7 = ane_bridge_input_surface_id(g_session.kernel7, 0);
    if (g_session.inSID7 == 0) {
        return false;
    }
    if (!ane_session_prime_matmul7_weights(g_session.inSID7, ic7, oc7, seq)) {
        LOG_WRN("%s: dflash ffn_up prime failed ic=%d oc=%d\n", __func__, ic7, oc7);
        return false;
    }
    g_session.matmul7Active = true;
    LOG_INF("%s: P15 dflash chain16 ffn_up ic=%d→oc=%d seq=%d\n", __func__, ic7, oc7, seq);
    return true;
}

static bool ane_session_init_dflash_ffn_down_weights(void) {
    const char * w8path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
    if (!w8path || !w8path[0] || !g_session.matmul6Active) {
        return false;
    }
    const int ic10 = g_session.matmul6Oc;
    int oc10 = g_session.matmul5Oc > 0 ? g_session.matmul5Oc : g_session.matmulOc;
    if (const char * oc8_env = std::getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_OC8")) {
        const int v = (int) strtol(oc8_env, NULL, 10);
        if (v > 0) {
            oc10 = v;
        }
    }
    if (ic10 <= 0 || oc10 <= 0) {
        return false;
    }
    size_t wcount = 0;
    if (!ane_load_matmul_weight_matrix(w8path, ic10, oc10, &g_session.matmul10W, &wcount)) {
        LOG_WRN("%s: dflash ffn_down load failed path=%s ic=%d oc=%d\n", __func__, w8path, ic10, oc10);
        return false;
    }
    g_session.matmul10WCount = wcount;
    g_session.matmul10Ic = ic10;
    g_session.matmul10Oc = oc10;
    g_session.matmul10Active = true;
    LOG_INF("%s: P15 dflash chain16 ffn_down host_fp32 ic=%d→oc=%d\n", __func__, ic10, oc10);
    return true;
}

static bool ane_session_load_output_norm_gamma(int n) {
    const char * path = std::getenv("ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE");
    if (!path || !path[0] || n <= 0) {
        return false;
    }
    if (g_session.outputNormGamma && g_session.outputNormGammaLen == n) {
        return true;
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    const size_t f32_expected = (size_t) n * sizeof(float);
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::rewind(f);
    if (sz < 0 || (size_t) sz < f32_expected) {
        std::fclose(f);
        return false;
    }
    if (g_session.outputNormGamma) {
        free(g_session.outputNormGamma);
        g_session.outputNormGamma = NULL;
    }
    g_session.outputNormGamma = (float *) calloc((size_t) n, sizeof(float));
    if (!g_session.outputNormGamma) {
        std::fclose(f);
        return false;
    }
    g_session.outputNormGammaLen = n;
    if (std::fread(g_session.outputNormGamma, sizeof(float), (size_t) n, f) != (size_t) n) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    LOG_INF("%s: P16 dflash output_norm gamma n=%d path=%s\n", __func__, n, path);
    return true;
}

static bool ane_session_init_dflash_chain16(int seq) {
    if (!ane_session_init_dflash_chain15(seq)) {
        return false;
    }
    if (!ane_session_init_dflash_ffn_up(seq)) {
        return false;
    }
    if (!ane_session_init_dflash_ffn_down_weights()) {
        return false;
    }
    const int wo_len = g_session.matmul5Oc > 0 ? g_session.matmul5Oc : g_session.matmulOc;
    if (wo_len > 0) {
        g_session.lastDflashWoHidden = (float *) calloc((size_t) wo_len, sizeof(float));
        if (!g_session.lastDflashWoHidden) {
            return false;
        }
        g_session.lastDflashWoHiddenLen = wo_len;
    }
    g_session.dflashChain16Active = true;
    return true;
}

static bool ane_session_init_dflash_chain17(int seq) {
    if (!ane_session_init_dflash_chain16(seq)) {
        return false;
    }
    const int n_embd = g_session.matmul10Oc > 0 ? g_session.matmul10Oc : g_session.matmul5Oc;
    if (n_embd <= 0 || !ane_session_load_output_norm_gamma(n_embd)) {
        LOG_WRN("%s: P16 dflash chain17 output_norm load failed n=%d\n", __func__, n_embd);
        return false;
    }
    g_session.dflashChain17Active = true;
    return true;
}

static bool ane_session_eval(void) {
    if (g_session.dflashChain13Active || g_session.dflashChain12Active) {
        return ane_session_eval_dflash_chain12(g_session.sp);
    }
    if (g_session.dflashChain11Active) {
        return ane_session_eval_dflash_chain11(g_session.sp);
    }
    if (g_session.matmulActive) {
        if (g_session.matmul5Active && g_session.kernel5) {
            return ane_session_eval_matmul_chain5(g_session.sp);
        }
        if (g_session.matmul4Active && g_session.kernel4) {
            return ane_session_eval_matmul_chain4(g_session.sp);
        }
        if (g_session.matmul3Active && g_session.kernel3) {
            return ane_session_eval_matmul_chain3(g_session.sp);
        }
        if (!ane_bridge_eval(g_session.kernel)) {
            return false;
        }
        const int oc1 = g_session.matmulOc;
        const int seq = g_session.sp;
        const size_t gate_bytes = (size_t) oc1 * (size_t) seq * sizeof(float);
        if (!g_session.matmul2Active || !g_session.kernel2) {
            ane_bridge_read_output(g_session.kernel, 0, g_session.outBuf, g_session.outIoBytes);
            g_session.stepCount++;
            return true;
        }
        std::vector<float> gate((size_t) oc1 * (size_t) seq);
        ane_bridge_read_output(g_session.kernel, 0, gate.data(), gate_bytes);
        std::vector<float> act2((size_t) g_session.matmul2Ic);
        for (int i = 0; i < g_session.matmul2Ic; ++i) {
            double sum = 0.0;
            for (int s = 0; s < seq; ++s) {
                sum += (double) gate[(size_t) i * (size_t) seq + (size_t) s];
            }
            act2[(size_t) i] = ane_silu((float) (sum / (double) (seq > 0 ? seq : 1)));
        }
        if (!ane_session_pack_matmul2_activations(
                g_session.inSID2, act2.data(), g_session.matmul2Ic,
                g_session.matmul2Ic, g_session.matmul2Oc, seq)) {
            return false;
        }
        if (!ane_bridge_eval(g_session.kernel2)) {
            return false;
        }
        ane_bridge_read_output(g_session.kernel2, 0, g_session.outBuf, g_session.outIoBytes);
        g_session.stepCount++;
        return true;
    }
    if (!ane_bridge_eval(g_session.kernel)) {
        return false;
    }
    if (!g_session.kernel2) {
        ane_bridge_read_output(g_session.kernel, 0, g_session.outBuf, g_session.ioBytes);
        g_session.stepCount++;
        return true;
    }
    float * mid = (float *) malloc(g_session.ioBytes);
    if (!mid) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel, 0, mid, g_session.ioBytes);
    if (!ane_session_write_surface(g_session.inSID2, g_session.ioBytes, mid)) {
        free(mid);
        return false;
    }
    free(mid);
    if (!ane_bridge_eval(g_session.kernel2)) {
        return false;
    }
    if (!g_session.kernel3) {
        ane_bridge_read_output(g_session.kernel2, 0, g_session.outBuf, g_session.ioBytes);
        g_session.stepCount++;
        return true;
    }
    mid = (float *) malloc(g_session.ioBytes);
    if (!mid) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel2, 0, mid, g_session.ioBytes);
    if (!ane_session_write_surface(g_session.inSID3, g_session.ioBytes, mid)) {
        free(mid);
        return false;
    }
    free(mid);
    if (!ane_bridge_eval(g_session.kernel3)) {
        return false;
    }
    if (!g_session.kernel4) {
        ane_bridge_read_output(g_session.kernel3, 0, g_session.outBuf, g_session.ioBytes);
        g_session.stepCount++;
        return true;
    }
    mid = (float *) malloc(g_session.ioBytes);
    if (!mid) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel3, 0, mid, g_session.ioBytes);
    if (!ane_session_write_surface(g_session.inSID4, g_session.ioBytes, mid)) {
        free(mid);
        return false;
    }
    free(mid);
    if (!ane_bridge_eval(g_session.kernel4)) {
        return false;
    }
    if (!g_session.kernel5) {
        ane_bridge_read_output(g_session.kernel4, 0, g_session.outBuf, g_session.ioBytes);
        g_session.stepCount++;
        return true;
    }
    mid = (float *) malloc(g_session.ioBytes);
    if (!mid) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel4, 0, mid, g_session.ioBytes);
    if (!ane_session_write_surface(g_session.inSID5, g_session.ioBytes, mid)) {
        free(mid);
        return false;
    }
    free(mid);
    if (!ane_bridge_eval(g_session.kernel5)) {
        return false;
    }
    if (!g_session.kernel6) {
        ane_bridge_read_output(g_session.kernel5, 0, g_session.outBuf, g_session.ioBytes);
        g_session.stepCount++;
        return true;
    }
    mid = (float *) malloc(g_session.ioBytes);
    if (!mid) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel5, 0, mid, g_session.ioBytes);
    if (!ane_session_write_surface(g_session.inSID6, g_session.ioBytes, mid)) {
        free(mid);
        return false;
    }
    free(mid);
    if (!ane_bridge_eval(g_session.kernel6)) {
        return false;
    }
    if (!g_session.kernel7) {
        ane_bridge_read_output(g_session.kernel6, 0, g_session.outBuf, g_session.ioBytes);
        g_session.stepCount++;
        return true;
    }
    mid = (float *) malloc(g_session.ioBytes);
    if (!mid) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel6, 0, mid, g_session.ioBytes);
    if (!ane_session_write_surface(g_session.inSID7, g_session.ioBytes, mid)) {
        free(mid);
        return false;
    }
    free(mid);
    if (!ane_bridge_eval(g_session.kernel7)) {
        return false;
    }
    if (!g_session.kernel8) {
        ane_bridge_read_output(g_session.kernel7, 0, g_session.outBuf, g_session.ioBytes);
        g_session.stepCount++;
        return true;
    }
    mid = (float *) malloc(g_session.ioBytes);
    if (!mid) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel7, 0, mid, g_session.ioBytes);
    if (!ane_session_write_surface(g_session.inSID8, g_session.ioBytes, mid)) {
        free(mid);
        return false;
    }
    free(mid);
    if (!ane_bridge_eval(g_session.kernel8)) {
        return false;
    }
    ane_bridge_read_output(g_session.kernel8, 0, g_session.outBuf, g_session.ioBytes);
    g_session.stepCount++;
    return true;
}

int ane_draft_session_channels(void) {
    return g_session.ch;
}

int ane_draft_session_spatial(void) {
    return g_session.sp;
}

bool ane_draft_session_eval(void) {
    @autoreleasepool {
        if (!ane_draft_session_ready()) {
            return false;
        }
        return ane_session_eval();
    }
}

static dispatch_queue_t g_ane_eval_queue = NULL;
static dispatch_group_t g_ane_eval_group = NULL;
static std::atomic<bool> g_ane_eval_last_ok { false };

static void ane_eval_async_init(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        g_ane_eval_group = dispatch_group_create();
        g_ane_eval_queue = dispatch_queue_create("zerollama.ane.draft.eval", DISPATCH_QUEUE_SERIAL);
    });
}

void ane_draft_session_eval_sync(void) {
    @autoreleasepool {
        if (!g_ane_eval_group) {
            return;
        }
        dispatch_group_wait(g_ane_eval_group, DISPATCH_TIME_FOREVER);
    }
}

bool ane_draft_session_eval_async(ane_draft_eval_async_fn on_done) {
    @autoreleasepool {
        if (!ane_draft_session_ready()) {
            return false;
        }
        ane_eval_async_init();
        dispatch_group_enter(g_ane_eval_group);
        dispatch_group_async(g_ane_eval_group, g_ane_eval_queue, ^{
            @autoreleasepool {
                const bool ok = ane_session_eval();
                g_ane_eval_last_ok.store(ok);
                if (on_done) {
                    on_done(ok);
                }
                dispatch_group_leave(g_ane_eval_group);
            }
        });
        return true;
    }
}

bool ane_draft_session_eval_async_enabled(void) {
    const char * e = getenv("ZEROLLAMA_ANE_DRAFT_EVAL_ASYNC");
    if (e && e[0]) {
        return strcmp(e, "0") != 0 && strcasecmp(e, "false") != 0;
    }
    return g_session.matmulActive;
}

size_t ane_draft_session_read_output(float * dst, size_t dst_floats) {
    if (!g_session.outBuf) {
        return 0;
    }
    const size_t bytes = g_session.matmulActive ? g_session.outIoBytes : g_session.ioBytes;
    if (bytes == 0) {
        return 0;
    }
    const size_t n = bytes / sizeof(float);
    const size_t copy = dst_floats < n ? dst_floats : n;
    if (dst && copy > 0) {
        memcpy(dst, g_session.outBuf, copy * sizeof(float));
    }
    return copy * sizeof(float);
}

int ane_draft_session_step_count(void) {
    return g_session.stepCount;
}

int ane_draft_session_conv_depth_cap(void) {
    return ane_conv_depth_cap();
}

int ane_draft_session_active_conv_count(void) {
    if (!g_session.kernel) {
        return 0;
    }
    if (g_session.matmulActive) {
        return 1;
    }
    int n = 1;
    if (g_session.conv2Active && g_session.kernel2 != NULL) { n++; }
    if (g_session.conv3Active && g_session.kernel3 != NULL) { n++; }
    if (g_session.conv4Active && g_session.kernel4 != NULL) { n++; }
    if (g_session.conv5Active && g_session.kernel5 != NULL) { n++; }
    if (g_session.conv6Active && g_session.kernel6 != NULL) { n++; }
    if (g_session.conv7Active && g_session.kernel7 != NULL) { n++; }
    if (g_session.conv8Active && g_session.kernel8 != NULL) { n++; }
    return n;
}

int ane_draft_session_output_channels(void) {
    if (g_session.dflashChain17Active && g_session.matmul10Oc > 0) {
        return g_session.matmul10Oc;
    }
    if (g_session.dflashChain16Active && g_session.matmul10Active && g_session.matmul10Oc > 0) {
        return g_session.matmul10Oc;
    }
    if (g_session.dflashChain15Active && g_session.matmul6Active && g_session.matmul6Oc > 0) {
        return g_session.matmul6Oc;
    }
    if (g_session.dflashChain14Active && g_session.matmul5Active && g_session.matmul5Oc > 0) {
        return g_session.matmul5Oc;
    }
    if ((g_session.dflashChain13Active || g_session.dflashChain12Active) &&
        g_session.matmul4Active && g_session.matmul4Oc > 0) {
        return g_session.matmul4Oc;
    }
    if (g_session.dflashChain11Active && g_session.matmul2Active && g_session.matmul2Oc > 0) {
        return g_session.matmul2Oc;
    }
    if (g_session.dflashFcActive && g_session.matmulActive && g_session.matmulOc > 0) {
        return g_session.matmulOc;
    }
    if (g_session.matmul10Active && g_session.matmul10Oc > 0) {
        return g_session.matmul10Oc;
    }
    if (g_session.matmul9Active && g_session.matmul9Oc > 0) {
        return g_session.matmul9Oc;
    }
    if (g_session.matmul7Active && g_session.matmul7Oc > 0) {
        return g_session.matmul7Oc;
    }
    if (g_session.matmul5Active && g_session.matmul5Oc > 0) {
        return g_session.matmul5Oc;
    }
    if (g_session.matmul4Active && g_session.matmul4Oc > 0) {
        return g_session.matmul4Oc;
    }
    if (g_session.matmul3Active && g_session.matmul3Oc > 0) {
        return g_session.matmul3Oc;
    }
    if (g_session.matmul2Active && g_session.matmul2Oc > 0) {
        return g_session.matmul2Oc;
    }
    if (g_session.matmulActive && g_session.matmulOc > 0) {
        return g_session.matmulOc;
    }
    return g_session.ch;
}

int ane_draft_session_matmul_chain_depth(void) {
    if (!g_session.matmulActive) {
        return 0;
    }
    if (g_session.dflashChain17Active) {
        return 17;
    }
    if (g_session.dflashChain16Active) {
        return 16;
    }
    if (g_session.dflashChain15Active) {
        return 15;
    }
    if (g_session.dflashChain14Active) {
        return 14;
    }
    if (g_session.dflashChain13Active) {
        return 13;
    }
    if (g_session.dflashChain12Active) {
        return 12;
    }
    if (g_session.dflashChain11Active) {
        return 11;
    }
    if (g_session.dflashFcActive) {
        return 8;
    }
    if (g_session.matmul10Active) {
        return 10;
    }
    if (g_session.matmul9Active) {
        return 9;
    }
    if (g_session.matmul7Active) {
        return 7;
    }
    if (g_session.matmul6Active) {
        return 6;
    }
    if (g_session.matmul5Active) {
        return 5;
    }
    if (g_session.matmul4Active) {
        return 4;
    }
    if (g_session.matmul3Active) {
        return 3;
    }
    return g_session.matmul2Active ? 2 : 1;
}

bool ane_draft_session_dflash_fc_active(void) {
    return g_session.matmulActive && g_session.dflashFcActive;
}

bool ane_draft_session_dflash_chain11_active(void) {
    return g_session.matmulActive && g_session.dflashChain11Active;
}

bool ane_draft_session_dflash_chain12_active(void) {
    return g_session.matmulActive && g_session.dflashChain12Active;
}

bool ane_draft_session_dflash_chain13_active(void) {
    return g_session.matmulActive && g_session.dflashChain13Active;
}

bool ane_draft_session_dflash_chain14_active(void) {
    return g_session.matmulActive && g_session.dflashChain14Active;
}

bool ane_draft_session_dflash_chain15_active(void) {
    return g_session.matmulActive && g_session.dflashChain15Active;
}

bool ane_draft_session_dflash_chain16_active(void) {
    return g_session.matmulActive && g_session.dflashChain16Active;
}

bool ane_draft_session_dflash_chain17_active(void) {
    return g_session.matmulActive && g_session.dflashChain17Active;
}

bool ane_draft_session_eval_dflash_attn_wo(void) {
    if ((!g_session.dflashChain14Active && !g_session.dflashChain15Active && !g_session.dflashChain16Active && !g_session.dflashChain17Active) || !g_session.kernel5 || !g_session.outBuf) {
        return false;
    }
    const int ic5 = g_session.matmul5Ic;
    const int oc5 = g_session.matmul5Oc;
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    if (ic5 <= 0 || oc5 <= 0) {
        return false;
    }
    std::vector<float> attn((size_t) ic5);
    for (int o = 0; o < ic5; ++o) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) g_session.outBuf[(size_t) o * (size_t) seq + (size_t) s];
        }
        attn[(size_t) o] = (float) (sum / (double) seq);
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID5, attn.data(), ic5, ic5, oc5, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel5)) {
        return false;
    }
    const size_t wo_bytes = (size_t) oc5 * (size_t) seq * sizeof(float);
    ane_bridge_read_output(g_session.kernel5, 0, g_session.outBuf, wo_bytes);
    g_session.outIoBytes = wo_bytes;
    g_session.stepCount++;
    return true;
}

bool ane_draft_session_eval_dflash_ffn_gate(void) {
    if ((!g_session.dflashChain15Active && !g_session.dflashChain16Active && !g_session.dflashChain17Active) || !g_session.kernel6 || !g_session.outBuf) {
        return false;
    }
    const int ic6 = g_session.matmul6Ic;
    const int oc6 = g_session.matmul6Oc;
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    if (ic6 <= 0 || oc6 <= 0) {
        return false;
    }
    std::vector<float> hidden((size_t) ic6);
    for (int i = 0; i < ic6; ++i) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) g_session.outBuf[(size_t) i * (size_t) seq + (size_t) s];
        }
        hidden[(size_t) i] = (float) (sum / (double) seq);
    }
    if ((g_session.dflashChain16Active || g_session.dflashChain17Active) && g_session.lastDflashWoHidden &&
        g_session.lastDflashWoHiddenLen >= ic6) {
        memcpy(g_session.lastDflashWoHidden, hidden.data(), (size_t) ic6 * sizeof(float));
    }
    if (!ane_session_pack_matmul2_activations(
            g_session.inSID6, hidden.data(), ic6, ic6, oc6, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel6)) {
        return false;
    }
    const size_t gate_bytes = (size_t) oc6 * (size_t) seq * sizeof(float);
    ane_bridge_read_output(g_session.kernel6, 0, g_session.outBuf, gate_bytes);
    g_session.outIoBytes = gate_bytes;
    g_session.stepCount++;
    return true;
}

bool ane_draft_session_eval_dflash_ffn_up_swiglu_down(void) {
    if (!g_session.dflashChain16Active || !g_session.kernel7 || !g_session.outBuf ||
        !g_session.matmul10Active || !g_session.matmul10W) {
        return false;
    }
    const int oc_gate = g_session.matmul6Oc;
    const int ic_up = g_session.matmul7Ic;
    const int oc_up = g_session.matmul7Oc;
    const int ic_down = g_session.matmul10Ic;
    const int oc_down = g_session.matmul10Oc;
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    if (oc_gate <= 0 || ic_up <= 0 || oc_up <= 0 || ic_down <= 0 || oc_down <= 0 ||
        !g_session.lastDflashWoHidden || g_session.lastDflashWoHiddenLen < ic_up) {
        return false;
    }
    const size_t gate_bytes = (size_t) oc_gate * (size_t) seq * sizeof(float);
    std::vector<float> gate((size_t) oc_gate * (size_t) seq);
    memcpy(gate.data(), g_session.outBuf, gate_bytes);

    if (!ane_session_pack_matmul2_activations(
            g_session.inSID7, g_session.lastDflashWoHidden, ic_up, ic_up, oc_up, seq)) {
        return false;
    }
    if (!ane_bridge_eval(g_session.kernel7)) {
        return false;
    }
    const size_t up_bytes = (size_t) oc_up * (size_t) seq * sizeof(float);
    std::vector<float> up((size_t) oc_up * (size_t) seq);
    ane_bridge_read_output(g_session.kernel7, 0, up.data(), up_bytes);

    std::vector<float> swiglu((size_t) oc_gate);
    for (int i = 0; i < oc_gate; ++i) {
        double g_sum = 0.0;
        double u_sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            g_sum += (double) gate[(size_t) i * (size_t) seq + (size_t) s];
            u_sum += (double) up[(size_t) i * (size_t) seq + (size_t) s];
        }
        const float g_avg = (float) (g_sum / (double) (seq > 0 ? seq : 1));
        const float u_avg = (float) (u_sum / (double) (seq > 0 ? seq : 1));
        swiglu[(size_t) i] = ane_silu(g_avg) * u_avg;
    }

    const size_t down_bytes = (size_t) oc_down * (size_t) seq * sizeof(float);
    if (down_bytes > g_session.outIoBytes) {
        if (g_session.outBuf) {
            free(g_session.outBuf);
        }
        g_session.outBuf = (float *) calloc((size_t) oc_down * (size_t) seq, sizeof(float));
        if (!g_session.outBuf) {
            return false;
        }
    }
    ane_host_matmul_seq(swiglu.data(), ic_down, oc_down, seq, g_session.matmul10W, g_session.outBuf);
    g_session.outIoBytes = down_bytes;
    g_session.stepCount++;
    return true;
}

bool ane_draft_session_set_dflash_fc_host(const float * fc, int n) {
    if (!fc || n <= 0) {
        return false;
    }
    if (!g_session.dflashFcHost || g_session.dflashFcHostLen < n) {
        float * next = (float *) realloc(g_session.dflashFcHost, (size_t) n * sizeof(float));
        if (!next) {
            return false;
        }
        g_session.dflashFcHost = next;
    }
    std::memcpy(g_session.dflashFcHost, fc, (size_t) n * sizeof(float));
    g_session.dflashFcHostLen = n;
    return true;
}

void ane_draft_session_clear_dflash_fc_host(void) {
    if (g_session.dflashFcHost) {
        free(g_session.dflashFcHost);
        g_session.dflashFcHost = NULL;
    }
    g_session.dflashFcHostLen = 0;
}

bool ane_draft_session_eval_dflash_output_norm(void) {
    if (!g_session.dflashChain17Active || !g_session.outBuf ||
        !g_session.outputNormGamma || g_session.outputNormGammaLen <= 0) {
        return false;
    }
    const int n_embd = g_session.outputNormGammaLen;
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    if (n_embd <= 0) {
        return false;
    }
    std::vector<float> hidden((size_t) n_embd);
    for (int i = 0; i < n_embd; ++i) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) g_session.outBuf[(size_t) i * (size_t) seq + (size_t) s];
        }
        hidden[(size_t) i] = (float) (sum / (double) (seq > 0 ? seq : 1));
    }
    ane_apply_rms_gamma(hidden.data(), n_embd, g_session.outputNormGamma);
    for (int i = 0; i < n_embd; ++i) {
        for (int s = 0; s < seq; ++s) {
            g_session.outBuf[(size_t) i * (size_t) seq + (size_t) s] = hidden[(size_t) i];
        }
    }
    g_session.stepCount++;
    return true;
}

bool ane_draft_session_eval_dflash_attn_post_norm(void) {
    if ((!g_session.dflashChain15Active && !g_session.dflashChain16Active && !g_session.dflashChain17Active) ||
        !g_session.outBuf || !g_session.attnPostNormGamma || g_session.attnPostNormGammaLen <= 0) {
        return false;
    }
    const int n_embd = g_session.attnPostNormGammaLen;
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    if (n_embd <= 0) {
        return false;
    }
    std::vector<float> hidden((size_t) n_embd);
    for (int i = 0; i < n_embd; ++i) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) g_session.outBuf[(size_t) i * (size_t) seq + (size_t) s];
        }
        hidden[(size_t) i] = (float) (sum / (double) (seq > 0 ? seq : 1));
    }
    ane_apply_rms_gamma(hidden.data(), n_embd, g_session.attnPostNormGamma);
    for (int i = 0; i < n_embd; ++i) {
        for (int s = 0; s < seq; ++s) {
            g_session.outBuf[(size_t) i * (size_t) seq + (size_t) s] = hidden[(size_t) i];
        }
    }
    return true;
}

bool ane_draft_session_write_dflash_attn_out(const float * src, int n) {
    if (!src || n <= 0 || !g_session.outBuf) {
        return false;
    }
    const int oc = g_session.matmul2Oc > 0 ? g_session.matmul2Oc : g_session.matmul4Oc;
    if (oc <= 0) {
        return false;
    }
    const int copy = n < oc ? n : oc;
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    for (int o = 0; o < copy; ++o) {
        for (int s = 0; s < seq; ++s) {
            g_session.outBuf[(size_t) o * (size_t) seq + (size_t) s] = src[o];
        }
    }
    return true;
}

static int ane_session_output_row_dim(void) {
    if (g_session.outIoBytes == 0) {
        return 0;
    }
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    return (int) (g_session.outIoBytes / ((size_t) seq * sizeof(float)));
}

bool ane_draft_session_snapshot_output_row(float * row, int n) {
    if (!row || n <= 0 || !g_session.outBuf) {
        return false;
    }
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    const int oc = ane_session_output_row_dim();
    if (oc <= 0) {
        return false;
    }
    const int copy = n < oc ? n : oc;
    for (int o = 0; o < copy; ++o) {
        double sum = 0.0;
        for (int s = 0; s < seq; ++s) {
            sum += (double) g_session.outBuf[(size_t) o * (size_t) seq + (size_t) s];
        }
        row[o] = (float) (sum / (double) (seq > 0 ? seq : 1));
    }
    return true;
}

bool ane_draft_session_add_output_row(const float * delta, int n) {
    if (!delta || n <= 0 || !g_session.outBuf) {
        return false;
    }
    const int seq = g_session.sp > 0 ? g_session.sp : 1;
    const int oc = ane_session_output_row_dim();
    if (oc <= 0) {
        return false;
    }
    const int copy = n < oc ? n : oc;
    for (int o = 0; o < copy; ++o) {
        for (int s = 0; s < seq; ++s) {
            g_session.outBuf[(size_t) o * (size_t) seq + (size_t) s] += delta[o];
        }
    }
    return true;
}

bool ane_draft_session_read_dflash_qkv(float * q, float * k, float * v, int n) {
    if (n <= 0 || g_session.lastDflashAttnLen <= 0) {
        return false;
    }
    const int copy = n < g_session.lastDflashAttnLen ? n : g_session.lastDflashAttnLen;
    if (q && g_session.lastDflashQ) {
        memcpy(q, g_session.lastDflashQ, (size_t) copy * sizeof(float));
    }
    if (k && g_session.lastDflashK) {
        memcpy(k, g_session.lastDflashK, (size_t) copy * sizeof(float));
    }
    if (v && g_session.lastDflashV) {
        memcpy(v, g_session.lastDflashV, (size_t) copy * sizeof(float));
    }
    return g_session.lastDflashQ && g_session.lastDflashK && g_session.lastDflashV;
}

int ane_draft_session_matmul_ffn_embd(void) {
    if (g_session.dflashChain16Active && g_session.matmul10Active && g_session.matmul10Oc > 0) {
        return g_session.matmul10Oc;
    }
    if ((g_session.dflashChain15Active || g_session.dflashChain14Active || g_session.dflashChain13Active || g_session.dflashChain12Active || g_session.dflashChain11Active || g_session.dflashFcActive) && g_session.matmulOc > 0) {
        return g_session.matmulOc;
    }
    if (g_session.matmul3Active && g_session.matmul3Oc > 0) {
        return g_session.matmul3Oc;
    }
    if (g_session.matmulActive && g_session.ch > 0) {
        return g_session.ch;
    }
    return 0;
}

int ane_draft_session_matmul9_oc(void) {
    if (g_session.matmul9Active && g_session.matmul9Oc > 0) {
        return g_session.matmul9Oc;
    }
    return 0;
}

size_t ane_draft_session_read_qkv_prefix(float * dst, size_t dst_floats) {
    if (!g_session.lastQkvHidden || g_session.lastQkvHiddenLen <= 0) {
        return 0;
    }
    const size_t n = (size_t) g_session.lastQkvHiddenLen;
    const size_t copy = dst_floats < n ? dst_floats : n;
    if (dst && copy > 0) {
        memcpy(dst, g_session.lastQkvHidden, copy * sizeof(float));
    }
    return copy * sizeof(float);
}

size_t ane_draft_session_read_ffn_down(float * dst, size_t dst_floats) {
    if (!g_session.lastDownHidden || g_session.lastDownHiddenLen <= 0) {
        return 0;
    }
    const size_t n = (size_t) g_session.lastDownHiddenLen;
    const size_t copy = dst_floats < n ? dst_floats : n;
    if (dst && copy > 0) {
        memcpy(dst, g_session.lastDownHidden, copy * sizeof(float));
    }
    return copy * sizeof(float);
}

bool ane_draft_session_matmul_active(void) {
    return g_session.matmulActive && g_session.kernel != NULL;
}

bool ane_draft_session_matmul_dynamic(void) {
    return g_session.matmulActive && g_session.matmulDynamic;
}

bool ane_draft_session_pack_matmul_activations(float * dst, const float * hidden, int hidden_len) {
    if (!dst || !g_session.matmulDynamic || !g_session.matmulWeightsPrimed) {
        return false;
    }
    const int ic = g_session.ch;
    const int seq = g_session.sp;
    const int oc = g_session.matmulOc > 0 ? g_session.matmulOc : ic;
    const int spIn = seq + oc;
    if (ic <= 0 || seq <= 0) {
        return false;
    }
    if (hidden && hidden_len > 0) {
        if (g_session.lastHiddenLen != hidden_len) {
            free(g_session.lastHidden);
            g_session.lastHidden = (float *) malloc((size_t) hidden_len * sizeof(float));
            g_session.lastHiddenLen = hidden_len;
        }
        if (g_session.lastHidden) {
            memcpy(g_session.lastHidden, hidden, (size_t) hidden_len * sizeof(float));
        }
    }
    for (int c = 0; c < ic; ++c) {
        const float v = (hidden && c < hidden_len) ? hidden[c] : 0.f;
        for (int s = 0; s < seq; ++s) {
            dst[(size_t) c * (size_t) spIn + (size_t) s] = v;
        }
    }
    return true;
}

bool ane_draft_session_using_conv2(void) {
    return (g_session.conv2Active && g_session.kernel2 != NULL) ||
           (g_session.conv3Active && g_session.kernel3 != NULL) ||
           (g_session.conv4Active && g_session.kernel4 != NULL) ||
           (g_session.conv5Active && g_session.kernel5 != NULL) ||
           (g_session.conv6Active && g_session.kernel6 != NULL) ||
           (g_session.conv7Active && g_session.kernel7 != NULL) ||
           (g_session.conv8Active && g_session.kernel8 != NULL);
}

bool ane_draft_session_step_once(float fill_val) {
    @autoreleasepool {
        if (!ane_draft_session_ready()) {
            return false;
        }
        if (!ane_session_map_fill(fill_val)) {
            return false;
        }
        return ane_session_eval();
    }
}

void ane_draft_session_shutdown(void) {
    @autoreleasepool {
        ane_session_clear(&g_session);
    }
}
