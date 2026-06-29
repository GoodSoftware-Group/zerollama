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

#include <cstdlib>
#include <stdlib.h>
#include <string.h>

static mach_timebase_info_data_t g_tb;
static bool                      g_tb_init = false;

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
    uint32_t          inSID;
    uint32_t          inSID2;
    uint32_t          inSID3;
    uint32_t          inSID4;
    uint32_t          inSID5;
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
    if (st->outBuf) {
        free(st->outBuf);
        st->outBuf = NULL;
    }
    st->inSID   = 0;
    st->inSID2  = 0;
    st->inSID3  = 0;
    st->inSID4  = 0;
    st->inSID5  = 0;
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

        const BOOL use_gamma = NO; // B3: gamma applied in ggml handoff pack (ANE mul broadcast unstable in MIL)
        const char * weight2_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * weight3_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * weight4_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        const char * weight5_path = getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        const BOOL want_conv2 = weight2_path && weight2_path[0];
        const BOOL want_conv3 = weight3_path && weight3_path[0];
        const BOOL want_conv4 = weight4_path && weight4_path[0];
        const BOOL want_conv5 = weight5_path && weight5_path[0];
        NSData * wb = nil;
        NSData * wb2 = nil;
        NSData * wb3 = nil;
        NSData * wb4 = nil;
        NSData * wb5 = nil;
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
        (void) use_gamma;
        (void) gamma_path;
        if (!wb || (want_conv2 && !wb2) || (want_conv3 && !wb3) || (want_conv4 && !wb4) || (want_conv5 && !wb5)) {
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

static bool ane_session_eval(void) {
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
    ane_bridge_read_output(g_session.kernel5, 0, g_session.outBuf, g_session.ioBytes);
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

size_t ane_draft_session_read_output(float * dst, size_t dst_floats) {
    if (!g_session.outBuf || g_session.ioBytes == 0) {
        return 0;
    }
    const size_t n = (size_t) g_session.ch * (size_t) g_session.sp;
    const size_t copy = dst_floats < n ? dst_floats : n;
    if (dst && copy > 0) {
        memcpy(dst, g_session.outBuf, copy * sizeof(float));
    }
    return copy * sizeof(float);
}

int ane_draft_session_step_count(void) {
    return g_session.stepCount;
}

bool ane_draft_session_using_conv2(void) {
    return (g_session.conv2Active && g_session.kernel2 != NULL) ||
           (g_session.conv3Active && g_session.kernel3 != NULL) ||
           (g_session.conv4Active && g_session.kernel4 != NULL) ||
           (g_session.conv5Active && g_session.kernel5 != NULL);
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
