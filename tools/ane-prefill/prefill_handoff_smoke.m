// Metal → IOSurface → ANE prefill handoff smoke.
// Why: ggml Metal will write prompt activations into ANE-bound IOSurface before
// mil_dynamic matmul eval — validates producer fill + prefill kernel without ggml hook.
//
// Variants:
//   baseline   — fp32 packed act|W (legacy)
//   fp16-conv  — Metal fills fp16 acts only; W is compile-time BLOBFILE 1×1 conv
//   fp16-dyn   — Metal fills fp16 acts; W is second IOSurface (MoE-ready)
//
// JSON: ok, mode=metal_prefill_handoff, variant, ic, oc, seq, metal_fill_ms, eval_ms, total_ms

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ane_bridge.h"
#include "../ane-metal/ggml_iosurface_map.h"

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

typedef NS_ENUM(NSInteger, HandoffVariant) {
    HandoffBaseline = 0,
    HandoffFP16Conv,
    HandoffFP16Dyn,
};

static HandoffVariant parseVariant(const char *s) {
    if (!s || !s[0] || strcmp(s, "baseline") == 0) return HandoffBaseline;
    if (strcmp(s, "fp16-conv") == 0) return HandoffFP16Conv;
    if (strcmp(s, "fp16-dyn") == 0) return HandoffFP16Dyn;
    return (HandoffVariant)-1;
}

static const char *variantName(HandoffVariant v) {
    switch (v) {
        case HandoffBaseline: return "baseline";
        case HandoffFP16Conv: return "fp16-conv";
        case HandoffFP16Dyn: return "fp16-dyn";
    }
    return "unknown";
}

static void writeWeightsOnSurface(void *base, int ic, int oc, int seq) {
    int spIn = seq + oc;
    float *buf = (float *)base;
    for (int c = 0; c < ic; c++) {
        for (int w = 0; w < oc; w++) {
            buf[c * spIn + seq + w] = 0.05f;
        }
    }
}

static IOSurfaceRef surfaceFromID(uint32_t sid) {
    return IOSurfaceLookup(sid);
}

static id<MTLComputePipelineState> buildActFillPipeline(id<MTLDevice> device, BOOL fp16, NSError **err) {
    NSString *src = fp16
        ? @"#include <metal_stdlib>\n"
          "using namespace metal;\n"
          "struct Params { int ic; int seq; int spIn; float val; };\n"
          "kernel void fill_act(device half *buf [[buffer(0)]], constant Params &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {\n"
          "    if (gid.x >= (uint)p.ic || gid.y >= (uint)p.seq) return;\n"
          "    buf[gid.x * p.spIn + gid.y] = half(p.val);\n"
          "}\n"
        : @"#include <metal_stdlib>\n"
          "using namespace metal;\n"
          "struct Params { int ic; int seq; int spIn; float val; };\n"
          "kernel void fill_act(device float *buf [[buffer(0)]], constant Params &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {\n"
          "    if (gid.x >= (uint)p.ic || gid.y >= (uint)p.seq) return;\n"
          "    buf[gid.x * p.spIn + gid.y] = p.val;\n"
          "}\n";
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"fill_act"];
    if (!fn) return nil;
    return [device newComputePipelineStateWithFunction:fn error:err];
}

static BOOL metalFillActivations(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> pipeline,
                                 uint32_t surfaceID,
                                 size_t bytes,
                                 int ic,
                                 int seq,
                                 int spIn,
                                 float value,
                                 double *fillMs) {
    IOSurfaceRef surface = surfaceFromID(surfaceID);
    if (!surface) return NO;

    IOSurfaceLock(surface, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return NO;
    }

    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:base
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!buffer) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return NO;
    }

    struct { int ic; int seq; int spIn; float val; } params = { ic, seq, spIn, value };

    uint64_t t0 = mach_absolute_time();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buffer offset:0 atIndex:0];
    [enc setBytes:&params length:sizeof(params) atIndex:1];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)ic, (NSUInteger)seq, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(16, (NSUInteger)ic), MIN(16, (NSUInteger)seq), 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (fillMs) *fillMs = ticksToMs(mach_absolute_time() - t0);

    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    return cmd.status == MTLCommandBufferStatusCompleted;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int ic = 256;
        int oc = 256;
        int seq = 128;
        int iters = 30;
        HandoffVariant variant = HandoffBaseline;
        BOOL steady = NO; // fill once; time eval only (ggml zero-copy proxy)
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 10;
            } else if (strcmp(argv[i], "--ic") == 0 && i + 1 < argc) {
                ic = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--oc") == 0 && i + 1 < argc) {
                oc = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
                seq = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
                variant = parseVariant(argv[++i]);
            } else if (strcmp(argv[i], "--steady") == 0) {
                steady = YES;
            }
        }

        if (variant < 0) {
            emitJSON(NO, "unknown --variant (baseline|fp16-conv|fp16-dyn)", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }
        if (ic <= 0 || oc <= 0 || seq <= 0) {
            emitJSON(NO, "ic/oc/seq must be positive", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }
        if (ane_bridge_init() != 0) {
            emitJSON(NO, "ane_bridge_init failed", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }

        NSString *mil = nil;
        size_t inBytes = 0, wBytes = 0, outBytes = 0;
        int nInputs = 1;
        size_t inputSizes[2] = {0, 0};
        uint8_t *weightBlob = NULL;
        size_t weightLen = 0;
        BOOL fp16Fill = NO;
        int fillSpIn = seq;

        switch (variant) {
            case HandoffBaseline: {
                int spIn = seq + oc;
                fillSpIn = spIn;
                inBytes = (size_t)ic * (size_t)spIn * sizeof(float);
                outBytes = (size_t)oc * (size_t)seq * sizeof(float);
                mil = genBaselineMIL(ic, oc, seq);
                break;
            }
            case HandoffFP16Conv: {
                fp16Fill = YES;
                inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
                outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);
                mil = genFP16ConvMIL(ic, oc, seq);
                float *w = (float *)malloc((size_t)oc * (size_t)ic * sizeof(float));
                if (!w) {
                    emitJSON(NO, "weight alloc failed", @{@"mode": @"metal_prefill_handoff"});
                    return 1;
                }
                for (int i = 0; i < oc * ic; i++) w[i] = 0.05f;
                weightBlob = ane_bridge_build_weight_blob(w, oc, ic, &weightLen);
                free(w);
                if (!weightBlob) {
                    emitJSON(NO, "weight blob failed", @{@"mode": @"metal_prefill_handoff"});
                    return 1;
                }
                break;
            }
            case HandoffFP16Dyn: {
                fp16Fill = YES;
                inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
                wBytes = (size_t)ic * (size_t)oc * sizeof(_Float16);
                outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);
                nInputs = 2;
                mil = genFP16DynMIL(ic, oc, seq);
                break;
            }
        }
        inputSizes[0] = inBytes;
        inputSizes[1] = wBytes;

        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        if (!milData) {
            free(weightBlob);
            emitJSON(NO, "MIL allocation failed", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }

        ANEKernelHandle *kernel = ane_bridge_compile(
            [milData bytes], [milData length],
            weightBlob, weightLen,
            nInputs, inputSizes, 1, &outBytes);
        free(weightBlob);
        if (!kernel) {
            emitJSON(NO, "ane_bridge_compile failed", @{
                @"mode": @"metal_prefill_handoff",
                @"variant": [NSString stringWithUTF8String:variantName(variant)],
            });
            return 1;
        }

        uint32_t inSID = ane_bridge_input_surface_id(kernel, 0);
        if (inSID == 0) {
            ane_bridge_free(kernel);
            emitJSON(NO, "ane_bridge_input_surface_id unavailable — run scripts/ane_bridge_patch.sh",
                       @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }

        if (variant == HandoffBaseline) {
            IOSurfaceRef initSurf = surfaceFromID(inSID);
            if (!initSurf) {
                ane_bridge_free(kernel);
                emitJSON(NO, "IOSurfaceLookup failed", @{@"mode": @"metal_prefill_handoff"});
                return 1;
            }
            IOSurfaceLock(initSurf, 0, NULL);
            writeWeightsOnSurface(IOSurfaceGetBaseAddress(initSurf), ic, oc, seq);
            IOSurfaceUnlock(initSurf, 0, NULL);
            CFRelease(initSurf);
        } else if (variant == HandoffFP16Dyn) {
            _Float16 *wHost = (_Float16 *)calloc(1, wBytes);
            if (!wHost) {
                ane_bridge_free(kernel);
                emitJSON(NO, "w host alloc failed", @{@"mode": @"metal_prefill_handoff"});
                return 1;
            }
            for (size_t i = 0; i < wBytes / sizeof(_Float16); i++) wHost[i] = (_Float16)0.05f;
            ane_bridge_write_input(kernel, 1, wHost, wBytes);
            free(wHost);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            ane_bridge_free(kernel);
            emitJSON(NO, "MTLCreateSystemDefaultDevice failed", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }
        NSError *merr = nil;
        id<MTLComputePipelineState> pipeline = buildActFillPipeline(device, fp16Fill, &merr);
        if (!pipeline) {
            ane_bridge_free(kernel);
            emitJSON(NO, "Metal activation fill pipeline failed", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        void *outBuf = calloc(1, outBytes);
        if (!outBuf) {
            ane_bridge_free(kernel);
            emitJSON(NO, "output buffer allocation failed", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }

        // One producer fill (Metal already has acts in real ggml path).
        double setupFillMs = 0;
        if (!metalFillActivations(device, queue, pipeline, inSID, inBytes, ic, seq, fillSpIn, 0.1f, &setupFillMs)) {
            ane_bridge_free(kernel);
            free(outBuf);
            emitJSON(NO, "metal activation fill failed", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }

        // ggml-style map cost (once) — proves IOSurface → MTLBuffer without copy.
        double mapMs = 0;
        {
            IOSurfaceRef surf = surfaceFromID(inSID);
            if (!surf) {
                ane_bridge_free(kernel);
                free(outBuf);
                emitJSON(NO, "IOSurfaceLookup for map failed", @{@"mode": @"metal_prefill_handoff"});
                return 1;
            }
            IOSurfaceLock(surf, 0, NULL);
            void *base = IOSurfaceGetBaseAddress(surf);
            void *mappedBase = NULL;
            size_t mappedSize = 0;
            uint64_t tm0 = mach_absolute_time();
            id<MTLBuffer> mapped = ggml_map_iosurface_base(device, base, inBytes, &mappedBase, &mappedSize);
            mapMs = ticksToMs(mach_absolute_time() - tm0);
            IOSurfaceUnlock(surf, 0, NULL);
            CFRelease(surf);
            if (!mapped) {
                ane_bridge_free(kernel);
                free(outBuf);
                emitJSON(NO, "ggml_map_iosurface_base failed", @{@"mode": @"metal_prefill_handoff"});
                return 1;
            }
        }

        // Warmup eval.
        for (int i = 0; i < 3; i++) {
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(outBuf);
                emitJSON(NO, "warmup eval failed", @{@"mode": @"metal_prefill_handoff"});
                return 1;
            }
        }

        double metalMs = 0, evalMs = 0;
        if (steady) {
            // Zero-copy proxy: surface already populated; only ANE eval per iter.
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                if (!ane_bridge_eval(kernel)) {
                    ane_bridge_free(kernel);
                    free(outBuf);
                    emitJSON(NO, "ane_bridge_eval failed", @{@"mode": @"metal_prefill_handoff"});
                    return 1;
                }
            }
            evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;
            metalMs = 0;
        } else {
            for (int i = 0; i < iters; i++) {
                double fill = 0;
                if (!metalFillActivations(device, queue, pipeline, inSID, inBytes, ic, seq, fillSpIn, 0.1f, &fill)) {
                    ane_bridge_free(kernel);
                    free(outBuf);
                    emitJSON(NO, "metal activation fill failed", @{@"mode": @"metal_prefill_handoff"});
                    return 1;
                }
                metalMs += fill;

                uint64_t t0 = mach_absolute_time();
                if (!ane_bridge_eval(kernel)) {
                    ane_bridge_free(kernel);
                    free(outBuf);
                    emitJSON(NO, "ane_bridge_eval failed", @{@"mode": @"metal_prefill_handoff"});
                    return 1;
                }
                evalMs += ticksToMs(mach_absolute_time() - t0);
            }
            metalMs /= (double)iters;
            evalMs /= (double)iters;
        }

        ane_bridge_read_output(kernel, 0, outBuf, outBytes);
        BOOL finite = YES;
        if (fp16Fill) {
            _Float16 *o = (_Float16 *)outBuf;
            for (size_t i = 0; i < outBytes / sizeof(_Float16); i++) {
                if (!isfinite((float)o[i])) { finite = NO; break; }
            }
        } else {
            float *o = (float *)outBuf;
            for (size_t i = 0; i < outBytes / sizeof(float); i++) {
                if (!isfinite(o[i])) { finite = NO; break; }
            }
        }

        ane_bridge_free(kernel);
        free(outBuf);

        if (!finite) {
            emitJSON(NO, "non-finite output", @{@"mode": @"metal_prefill_handoff"});
            return 1;
        }

        double gflop = 2.0 * (double)ic * (double)oc * (double)seq / 1e9;
        double totalMs = steady ? evalMs : (metalMs + evalMs);
        NSString *note = steady
            ? @"steady: fill once + ggml map once; timed eval only (ggml zero-copy proxy)"
            : @"per-iter Metal fill+eval; fill tax dominates — use --steady for ggml-like path";
        emitJSON(YES, NULL, @{
            @"mode": @"metal_prefill_handoff",
            @"variant": [NSString stringWithUTF8String:variantName(variant)],
            @"steady": @(steady),
            @"ic": @(ic),
            @"oc": @(oc),
            @"seq": @(seq),
            @"surface_id": @(inSID),
            @"surface_bytes": @(inBytes),
            @"setup_fill_ms": @(setupFillMs),
            @"map_ms": @(mapMs),
            @"metal_fill_ms": @(metalMs),
            @"eval_ms": @(evalMs),
            @"total_ms": @(totalMs),
            @"gflop": @(gflop),
            @"source": @"zerollama/prefill-handoff",
            @"note": note,
        });
        return 0;
    }
}
