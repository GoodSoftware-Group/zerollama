// ANE prefill proxy bench — dynamic matmul y = x @ W at prefill-like IC×OC×SEQ.
// Variants isolate overhead that caused ANE to lose expert-up vs MPS:
//   baseline     — fp32 I/O + cast_in/out + packed act|W (legacy)
//   fp16         — fp16 I/O, packed act|W (maderix mil_dynamic)
//   fp16-blob    — fp16 acts only + weight BLOBFILE, same layout sandwich
//   fp16-native  — fp16 [1,1,SEQ,IC] @ blob [1,1,IC,OC] (minimal MIL)
//   fp16-conv    — fp16 [1,IC,1,SEQ] + 1×1 conv weight blob [OC,IC,1,1]
//   fp16-dyn     — fp16 acts + runtime W input (MoE: --experts N swaps W, compile once)
//   int8-conv    — fp16 acts + int8 weight BLOBFILE via constexpr_affine_dequantize + 1×1 conv
//
// JSON: ok, mode, variant, ic, oc, seq, eval_ms, write_ms, read_ms, gflop, tflops, …

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ane_bridge.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

static mach_timebase_info_data_t g_tb;

static double ticksToMs(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static void emitJSON(BOOL ok, const char *err, NSDictionary *fields) {
    NSMutableString *json = [NSMutableString stringWithFormat:@"{\"ok\":%@", ok ? @"true" : @"false"];
    for (NSString *key in fields) {
        id val = fields[key];
        if ([val isKindOfClass:[NSNumber class]]) {
            [json appendFormat:@",\"%@\":%@", key, val];
        } else if ([val isKindOfClass:[NSString class]]) {
            NSString *s = [(NSString *)val stringByReplacingOccurrencesOfString:@"\\" withString:@"\\\\"];
            s = [s stringByReplacingOccurrencesOfString:@"\"" withString:@"\\\""];
            [json appendFormat:@",\"%@\":\"%@\"", key, s];
        }
    }
    if (err && err[0]) {
        NSString *escaped = [[NSString stringWithUTF8String:err]
            stringByReplacingOccurrencesOfString:@"\\" withString:@"\\\\"];
        escaped = [escaped stringByReplacingOccurrencesOfString:@"\"" withString:@"\\\""];
        [json appendFormat:@",\"error\":\"%@\"", escaped];
    }
    [json appendString:@"}\n"];
    printf("%s", [json UTF8String]);
}

static void appendDynMatmul(NSMutableString *m, const char *prefix,
                            int ic, int oc, int seq, int actSpOff, int wSpOff,
                            const char *inputVar) {
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

static NSString *genBaselineMIL(int ic, int oc, int seq) {
    int spIn = seq + oc;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ic, spIn];
    [m appendString:
        @"        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n"
        @"        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to_fp16, x=x)[name=string(\"cast_in\")];\n", ic, spIn];
    appendDynMatmul(m, "mm", ic, oc, seq, 0, seq, "x16");
    [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to_fp32, x=mm_y)[name=string(\"cast_out\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static NSString *genFP16PackedMIL(int ic, int oc, int seq) {
    int spIn = seq + oc;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, spIn];
    appendDynMatmul(m, "mm", ic, oc, seq, 0, seq, "x");
    [m appendString:@"    } -> (mm_y);\n}\n"];
    return m;
}

static NSString *genFP16BlobMIL(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> W = const()[name=string(\"W\"), "
        @"val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n",
        ic, oc, ic, oc];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=x)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n",
        seq, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static NSString *genFP16NativeMIL(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, 1, %d, %d]> x) {\n", seq, ic];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> W = const()[name=string(\"W\"), "
        @"val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n",
        ic, oc, ic, oc];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> y = matmul(transpose_x=bF,transpose_y=bF,x=x,y=W)[name=string(\"y\")];\n",
        seq, oc];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static NSString *genFP16ConvMIL(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendString:
        @"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        @"        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        @"        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        @"        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        @"        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [m appendFormat:
        @"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        @"val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n",
        oc, ic, oc, ic];
    [m appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"y\")];\n",
        oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// MoE-friendly: compile once, stream expert weights as input 1.
static NSString *genFP16DynMIL(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [1, 1, %d, %d]> W) {\n",
        ic, seq, ic, oc];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=x)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n",
        seq, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// int8 weights dequantized at compile time (maderix ane_int8_bench pattern) + 1×1 conv.
// Scale must be a MIL fp16 literal; %a hex float is safest for non-power-of-two scales.
static NSString *genInt8ConvMIL(int ic, int oc, int seq, float scale) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendString:
        @"        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"];
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize()"
        @"[axis = int32(0), name = string(\"W\"), "
        @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        @"(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64))), "
        @"scale = fp16(%a), zero_point = int8(0)];\n",
        oc, ic, oc, ic, (double)(_Float16)scale];
    [m appendFormat:
        @"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = W, x = x)[name = string(\"y\")];\n"
        @"    } -> (y);\n"
        @"}\n",
        oc, seq];
    return m;
}

typedef NS_ENUM(NSInteger, PrefillVariant) {
    PrefillVariantBaseline = 0,
    PrefillVariantFP16,
    PrefillVariantFP16Blob,
    PrefillVariantFP16Native,
    PrefillVariantFP16Conv,
    PrefillVariantFP16Dyn,
    PrefillVariantInt8Conv,
};

static PrefillVariant parseVariant(const char *s) {
    if (!s || !s[0] || strcmp(s, "baseline") == 0) return PrefillVariantBaseline;
    if (strcmp(s, "fp16") == 0) return PrefillVariantFP16;
    if (strcmp(s, "fp16-blob") == 0) return PrefillVariantFP16Blob;
    if (strcmp(s, "fp16-native") == 0) return PrefillVariantFP16Native;
    if (strcmp(s, "fp16-conv") == 0) return PrefillVariantFP16Conv;
    if (strcmp(s, "fp16-dyn") == 0) return PrefillVariantFP16Dyn;
    if (strcmp(s, "int8-conv") == 0) return PrefillVariantInt8Conv;
    return (PrefillVariant)-1;
}

static const char *variantName(PrefillVariant v) {
    switch (v) {
        case PrefillVariantBaseline: return "baseline";
        case PrefillVariantFP16: return "fp16";
        case PrefillVariantFP16Blob: return "fp16-blob";
        case PrefillVariantFP16Native: return "fp16-native";
        case PrefillVariantFP16Conv: return "fp16-conv";
        case PrefillVariantFP16Dyn: return "fp16-dyn";
        case PrefillVariantInt8Conv: return "int8-conv";
    }
    return "unknown";
}

static void fillPackedFP32(float *buf, int ic, int oc, int seq) {
    int spIn = seq + oc;
    for (int c = 0; c < ic; c++) {
        for (int s = 0; s < seq; s++) buf[c * spIn + s] = 0.1f;
        for (int w = 0; w < oc; w++) buf[c * spIn + seq + w] = 0.05f;
    }
}

static void fillPackedFP16(_Float16 *buf, int ic, int oc, int seq) {
    int spIn = seq + oc;
    for (int c = 0; c < ic; c++) {
        for (int s = 0; s < seq; s++) buf[c * spIn + s] = (_Float16)0.1f;
        for (int w = 0; w < oc; w++) buf[c * spIn + seq + w] = (_Float16)0.05f;
    }
}

// Acts/weights must keep mul products ≥ ~1e-4: ANE fp16 flushes denormals (0.001×0.01→0).
static void fillActsFP16(_Float16 *buf, int ic, int seq) {
    for (int i = 0; i < ic * seq; i++) buf[i] = (_Float16)0.1f;
}

static void fillNativeFP16(_Float16 *buf, int seq, int ic) {
    for (int i = 0; i < seq * ic; i++) buf[i] = (_Float16)0.1f;
}

static void fillWeightFP16(_Float16 *buf, int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; i++) buf[i] = (_Float16)scale;
}

static float *makeWeightFP32(int rows, int cols) {
    float *w = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    if (!w) return NULL;
    for (int i = 0; i < rows * cols; i++) w[i] = 0.05f;
    return w;
}

static BOOL outputFiniteFP32(const float *buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(buf[i])) return NO;
    }
    return YES;
}

static BOOL outputFiniteFP16(const _Float16 *buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((float)buf[i])) return NO;
    }
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int ic = 256;
        int oc = 256;
        int seq = 512;
        int warmup = 3;
        int iters = 40;
        int experts = 1;
        PrefillVariant variant = PrefillVariantBaseline;
        BOOL phases = NO;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 15;
            } else if (strcmp(argv[i], "--ic") == 0 && i + 1 < argc) {
                ic = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--oc") == 0 && i + 1 < argc) {
                oc = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
                seq = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
                variant = parseVariant(argv[++i]);
            } else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
                experts = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--phases") == 0) {
                phases = YES;
            }
        }

        if (variant < 0) {
            emitJSON(NO, "unknown --variant (baseline|fp16|fp16-blob|fp16-native|fp16-conv|fp16-dyn|int8-conv)", @{
                @"mode": @"prefill_matmul",
            });
            return 1;
        }
        if (ic <= 0 || oc <= 0 || seq <= 0 || experts <= 0) {
            emitJSON(NO, "ic/oc/seq/experts must be positive", @{@"mode": @"prefill_matmul"});
            return 1;
        }
        if (experts > 1 && variant != PrefillVariantFP16Dyn) {
            emitJSON(NO, "--experts > 1 requires --variant fp16-dyn", @{@"mode": @"prefill_matmul"});
            return 1;
        }
        if (ane_bridge_init() != 0) {
            emitJSON(NO, "ane_bridge_init failed", @{@"mode": @"prefill_matmul"});
            return 1;
        }

        NSString *mil = nil;
        size_t inBytes = 0;
        size_t wBytes = 0;
        size_t outBytes = 0;
        BOOL useBlob = NO;
        BOOL useInt8Blob = NO;
        BOOL useDynW = NO;
        BOOL outFP16 = YES;
        int blobRows = ic, blobCols = oc;
        int nInputs = 1;
        size_t inputSizes[2] = {0, 0};
        float int8Scale = 0;

        switch (variant) {
            case PrefillVariantBaseline: {
                int spIn = seq + oc;
                inBytes = (size_t)ic * (size_t)spIn * sizeof(float);
                outBytes = (size_t)oc * (size_t)seq * sizeof(float);
                outFP16 = NO;
                mil = genBaselineMIL(ic, oc, seq);
                break;
            }
            case PrefillVariantFP16: {
                int spIn = seq + oc;
                inBytes = (size_t)ic * (size_t)spIn * sizeof(_Float16);
                outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);
                mil = genFP16PackedMIL(ic, oc, seq);
                break;
            }
            case PrefillVariantFP16Blob: {
                inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
                outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);
                useBlob = YES;
                mil = genFP16BlobMIL(ic, oc, seq);
                break;
            }
            case PrefillVariantFP16Native: {
                inBytes = (size_t)seq * (size_t)ic * sizeof(_Float16);
                outBytes = (size_t)seq * (size_t)oc * sizeof(_Float16);
                useBlob = YES;
                mil = genFP16NativeMIL(ic, oc, seq);
                break;
            }
            case PrefillVariantFP16Conv: {
                inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
                outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);
                useBlob = YES;
                blobRows = oc;
                blobCols = ic;
                mil = genFP16ConvMIL(ic, oc, seq);
                break;
            }
            case PrefillVariantFP16Dyn: {
                inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
                wBytes = (size_t)ic * (size_t)oc * sizeof(_Float16);
                outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);
                useDynW = YES;
                nInputs = 2;
                mil = genFP16DynMIL(ic, oc, seq);
                break;
            }
            case PrefillVariantInt8Conv: {
                inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
                outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);
                useBlob = YES;
                useInt8Blob = YES;
                blobRows = oc;
                blobCols = ic;
                // MIL filled after quantize (needs scale).
                mil = nil;
                break;
            }
        }
        inputSizes[0] = inBytes;
        inputSizes[1] = wBytes;

        uint8_t *weightBlob = NULL;
        size_t weightLen = 0;
        float *wFP32 = NULL;
        if (useBlob) {
            wFP32 = makeWeightFP32(blobRows, blobCols);
            if (!wFP32) {
                emitJSON(NO, "weight allocation failed", @{@"mode": @"prefill_matmul"});
                return 1;
            }
            if (useInt8Blob) {
                weightBlob = ane_bridge_build_weight_blob_quantized(
                    wFP32, blobRows, blobCols, &int8Scale, &weightLen);
                mil = genInt8ConvMIL(ic, oc, seq, int8Scale);
            } else {
                weightBlob = ane_bridge_build_weight_blob(wFP32, blobRows, blobCols, &weightLen);
            }
            free(wFP32);
            wFP32 = NULL;
            if (!weightBlob || !mil) {
                free(weightBlob);
                emitJSON(NO, "weight blob / MIL build failed", @{
                    @"mode": @"prefill_matmul",
                    @"variant": [NSString stringWithUTF8String:variantName(variant)],
                });
                return 1;
            }
        }

        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        if (!milData) {
            free(weightBlob);
            emitJSON(NO, "MIL allocation failed", @{@"mode": @"prefill_matmul"});
            return 1;
        }

        uint64_t tCompile0 = mach_absolute_time();
        ANEKernelHandle *kernel = ane_bridge_compile(
            [milData bytes], [milData length],
            weightBlob, weightLen,
            nInputs, inputSizes, 1, &outBytes);
        double compileMs = ticksToMs(mach_absolute_time() - tCompile0);
        if (weightBlob) {
            free(weightBlob);
            weightBlob = NULL;
        }
        if (!kernel) {
            emitJSON(NO, "ane_bridge_compile failed", @{
                @"mode": @"prefill_matmul",
                @"variant": [NSString stringWithUTF8String:variantName(variant)],
                @"ic": @(ic),
                @"oc": @(oc),
                @"seq": @(seq),
                @"compile_ms": @(compileMs),
                @"compile_count": @(ane_bridge_get_compile_count()),
            });
            return 1;
        }

        void *inBuf = calloc(1, inBytes);
        void *wBuf = useDynW ? calloc(1, wBytes) : NULL;
        void *outBuf = calloc(1, outBytes);
        if (!inBuf || !outBuf || (useDynW && !wBuf)) {
            ane_bridge_free(kernel);
            free(inBuf);
            free(wBuf);
            free(outBuf);
            emitJSON(NO, "buffer allocation failed", @{@"mode": @"prefill_matmul"});
            return 1;
        }

        switch (variant) {
            case PrefillVariantBaseline:
                fillPackedFP32((float *)inBuf, ic, oc, seq);
                break;
            case PrefillVariantFP16:
                fillPackedFP16((_Float16 *)inBuf, ic, oc, seq);
                break;
            case PrefillVariantFP16Blob:
            case PrefillVariantFP16Conv:
            case PrefillVariantFP16Dyn:
            case PrefillVariantInt8Conv:
                fillActsFP16((_Float16 *)inBuf, ic, seq);
                break;
            case PrefillVariantFP16Native:
                fillNativeFP16((_Float16 *)inBuf, seq, ic);
                break;
        }
        if (useDynW) {
            fillWeightFP16((_Float16 *)wBuf, ic, oc, 0.05f);
        }

        uint64_t tWrite0 = mach_absolute_time();
        ane_bridge_write_input(kernel, 0, inBuf, inBytes);
        if (useDynW) {
            ane_bridge_write_input(kernel, 1, wBuf, wBytes);
        }
        double writeMsOnce = ticksToMs(mach_absolute_time() - tWrite0);

        for (int i = 0; i < warmup; i++) {
            if (useDynW && experts > 1) {
                fillWeightFP16((_Float16 *)wBuf, ic, oc, 0.05f + 0.001f * (float)(i % experts));
                ane_bridge_write_input(kernel, 1, wBuf, wBytes);
            }
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(inBuf);
                free(wBuf);
                free(outBuf);
                emitJSON(NO, "warmup eval failed", @{
                    @"mode": @"prefill_matmul",
                    @"variant": [NSString stringWithUTF8String:variantName(variant)],
                });
                return 1;
            }
        }

        double evalMs = 0;
        double writeWMs = 0;
        double expertsTotalMs = 0;
        if (useDynW && experts > 1) {
            // MoE proxy: compile once, stream top-k expert weights, eval each.
            const int rounds = MAX(3, iters / experts);
            uint64_t tAll = 0, tW = 0, tE = 0;
            for (int r = 0; r < rounds; r++) {
                for (int e = 0; e < experts; e++) {
                    fillWeightFP16((_Float16 *)wBuf, ic, oc, 0.05f + 0.001f * (float)e);
                    uint64_t a = mach_absolute_time();
                    ane_bridge_write_input(kernel, 1, wBuf, wBytes);
                    uint64_t b = mach_absolute_time();
                    if (!ane_bridge_eval(kernel)) {
                        ane_bridge_free(kernel);
                        free(inBuf);
                        free(wBuf);
                        free(outBuf);
                        emitJSON(NO, "expert eval failed", @{@"mode": @"prefill_matmul"});
                        return 1;
                    }
                    uint64_t c = mach_absolute_time();
                    tW += b - a;
                    tE += c - b;
                    tAll += c - a;
                }
            }
            int n = rounds * experts;
            writeWMs = ticksToMs(tW) / (double)n;
            evalMs = ticksToMs(tE) / (double)n;
            expertsTotalMs = ticksToMs(tAll) / (double)rounds; // one full top-k step
        } else {
            // Steady-state eval (weights already compiled / written; acts on surface).
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                if (!ane_bridge_eval(kernel)) {
                    ane_bridge_free(kernel);
                    free(inBuf);
                    free(wBuf);
                    free(outBuf);
                    emitJSON(NO, "eval failed", @{
                        @"mode": @"prefill_matmul",
                        @"variant": [NSString stringWithUTF8String:variantName(variant)],
                    });
                    return 1;
                }
            }
            evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;
        }

        double writeMs = writeMsOnce;
        double readMs = 0;
        if (phases && !(useDynW && experts > 1)) {
            const int phaseIters = MAX(5, iters / 3);
            uint64_t tw = 0, te = 0, tr = 0;
            for (int i = 0; i < phaseIters; i++) {
                uint64_t a = mach_absolute_time();
                ane_bridge_write_input(kernel, 0, inBuf, inBytes);
                if (useDynW) {
                    ane_bridge_write_input(kernel, 1, wBuf, wBytes);
                }
                uint64_t b = mach_absolute_time();
                if (!ane_bridge_eval(kernel)) {
                    ane_bridge_free(kernel);
                    free(inBuf);
                    free(wBuf);
                    free(outBuf);
                    emitJSON(NO, "phase eval failed", @{@"mode": @"prefill_matmul"});
                    return 1;
                }
                uint64_t c = mach_absolute_time();
                ane_bridge_read_output(kernel, 0, outBuf, outBytes);
                uint64_t d = mach_absolute_time();
                tw += b - a;
                te += c - b;
                tr += d - c;
            }
            writeMs = ticksToMs(tw) / (double)phaseIters;
            (void)te;
            readMs = ticksToMs(tr) / (double)phaseIters;
        } else {
            ane_bridge_read_output(kernel, 0, outBuf, outBytes);
        }

        BOOL finite = outFP16
            ? outputFiniteFP16((_Float16 *)outBuf, outBytes / sizeof(_Float16))
            : outputFiniteFP32((float *)outBuf, outBytes / sizeof(float));
        float aneMax = 0;
        if (finite) {
            if (outFP16) {
                const _Float16 *o = (_Float16 *)outBuf;
                size_t n = outBytes / sizeof(_Float16);
                for (size_t i = 0; i < n; i++) {
                    float a = fabsf((float)o[i]);
                    if (a > aneMax) aneMax = a;
                }
            } else {
                const float *o = (float *)outBuf;
                size_t n = outBytes / sizeof(float);
                for (size_t i = 0; i < n; i++) {
                    float a = fabsf(o[i]);
                    if (a > aneMax) aneMax = a;
                }
            }
        }

        int compileCount = ane_bridge_get_compile_count();
        ane_bridge_free(kernel);
        free(inBuf);
        free(wBuf);
        free(outBuf);

        if (!finite) {
            emitJSON(NO, "non-finite output", @{
                @"mode": @"prefill_matmul",
                @"variant": [NSString stringWithUTF8String:variantName(variant)],
            });
            return 1;
        }
        // Catch silent denormal-flush zeros (old W=1e-3×X=1e-2 benches were empty).
        if (aneMax == 0) {
            emitJSON(NO, "ANE output all zeros (check act/weight scale vs fp16 denormals)", @{
                @"mode": @"prefill_matmul",
                @"variant": [NSString stringWithUTF8String:variantName(variant)],
                @"ic": @(ic), @"oc": @(oc), @"seq": @(seq),
            });
            return 1;
        }

        double gflop = 2.0 * (double)ic * (double)oc * (double)seq / 1e9;
        double tflops = evalMs > 0 ? gflop / (evalMs / 1000.0) : 0;

        NSMutableDictionary *fields = [@{
            @"mode": @"prefill_matmul",
            @"variant": [NSString stringWithUTF8String:variantName(variant)],
            @"ic": @(ic),
            @"oc": @(oc),
            @"seq": @(seq),
            @"experts": @(experts),
            @"eval_ms": @(evalMs),
            @"write_ms": @(writeMs),
            @"compile_ms": @(compileMs),
            @"gflop": @(gflop),
            @"tflops": @(tflops),
            @"compile_count": @(compileCount),
            @"ane_max_abs": @(aneMax),
            @"source": @"maderix/mil_dynamic",
            @"note": [NSString stringWithFormat:@"variant=%s; compare to Metal FFN at same IC×OC×SEQ",
                variantName(variant)],
        } mutableCopy];
        if (phases) {
            fields[@"read_ms"] = @(readMs);
            fields[@"phases"] = @YES;
        }
        if (useDynW && experts > 1) {
            fields[@"write_w_ms"] = @(writeWMs);
            fields[@"experts_total_ms"] = @(expertsTotalMs);
            fields[@"note"] = [NSString stringWithFormat:
                @"fp16-dyn MoE proxy: compile-once, stream %d expert weights per step", experts];
        }
        if (useInt8Blob) {
            fields[@"int8_scale"] = @(int8Scale);
            fields[@"weight_blob_bytes"] = @(weightLen);
            fields[@"note"] = @"int8-conv: constexpr_affine_dequantize + 1x1 conv; fp16 acts";
        }

        emitJSON(YES, NULL, fields);
        return 0;
    }
}
