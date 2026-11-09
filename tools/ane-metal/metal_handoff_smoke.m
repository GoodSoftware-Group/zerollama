// Metal → IOSurface → ANE handoff smoke.
// Why: ggml Metal will produce activations into IOSurface-backed buffers; this
// validates zero-copy producer fill via newBufferWithIOSurface before ANE eval.
//
// JSON: ok, mode=metal_iosurface_handoff, metal_fill_ms, eval_ms, read_ms, surface_id

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#include "ane_bridge.h"

static mach_timebase_info_data_t g_tb;

static double ticksToMs(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static NSString *genConvMIL(int ch, int sp) {
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

static NSData *buildWeightBlob(int ch) {
    NSUInteger wsize = (NSUInteger)ch * (NSUInteger)ch * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = calloc(total, 1);
    if (!buf) {
        return nil;
    }
    buf[0] = 0x01;
    buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF;
    chunk[1] = 0xBE;
    chunk[2] = 0xAD;
    chunk[3] = 0xDE;
    chunk[4] = 0x01;
    chunk[10] = 0x08;
    uint16_t *fp16 = (uint16_t *)(chunk + 64);
    for (NSUInteger j = 0; j < wsize / 2; j++) {
        fp16[j] = 0x3400;
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
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

static IOSurfaceRef surfaceFromID(uint32_t sid) {
    return IOSurfaceLookup(sid);
}

static id<MTLComputePipelineState> buildFillPipeline(id<MTLDevice> device, NSError **err) {
    NSString *src = @""
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

static BOOL metalFillSurface(id<MTLDevice> device,
                           id<MTLCommandQueue> queue,
                           id<MTLComputePipelineState> pipeline,
                           uint32_t surfaceID,
                           size_t bytes,
                           float value,
                           double *fillMs) {
    IOSurfaceRef surface = surfaceFromID(surfaceID);
    if (!surface) {
        return NO;
    }

    IOSurfaceLock(surface, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return NO;
    }

    size_t nfloats = bytes / sizeof(float);
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:base
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!buffer) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return NO;
    }

    uint64_t t0 = mach_absolute_time();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buffer offset:0 atIndex:0];
    [enc setBytes:&value length:sizeof(value) atIndex:1];
    NSUInteger tg = MIN((NSUInteger)256, nfloats);
    [enc dispatchThreads:MTLSizeMake(nfloats, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (fillMs) {
        *fillMs = ticksToMs(mach_absolute_time() - t0);
    }

    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    return cmd.status == MTLCommandBufferStatusCompleted;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int ch = 64;
        int sp = 16;
        int iters = 30;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 10;
            } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
                ch = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--spatial") == 0 && i + 1 < argc) {
                sp = atoi(argv[++i]);
            }
        }

        if (ane_bridge_init() != 0) {
            emitJSON(NO, "ane_bridge_init failed", @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }

        NSString *mil = genConvMIL(ch, sp);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        NSData *wb = buildWeightBlob(ch);
        if (!milData || !wb) {
            emitJSON(NO, "MIL/weight allocation failed", @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }

        const char *weightName = "@model_path/weights/weight.bin";
        const uint8_t *weightData = (const uint8_t *)[wb bytes];
        size_t weightLen = [wb length];
        size_t ioBytes = (size_t)ch * (size_t)sp * sizeof(float);

        ANEKernelHandle *kernel = ane_bridge_compile_multi_weights(
            [milData bytes], [milData length],
            &weightName, &weightData, &weightLen, 1,
            1, &ioBytes, 1, &ioBytes);
        if (!kernel) {
            emitJSON(NO, "ane_bridge_compile failed", @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }

        uint32_t inSID = ane_bridge_input_surface_id(kernel, 0);
        if (inSID == 0) {
            ane_bridge_free(kernel);
            emitJSON(NO, "ane_bridge_input_surface_id unavailable — run scripts/ane/ane_bridge_patch.sh",
                       @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }

        float *outBuf = (float *)calloc(ch * sp, sizeof(float));
        if (!outBuf) {
            ane_bridge_free(kernel);
            emitJSON(NO, "output buffer allocation failed", @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            ane_bridge_free(kernel);
            free(outBuf);
            emitJSON(NO, "MTLCreateSystemDefaultDevice failed", @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }
        NSError *merr = nil;
        id<MTLComputePipelineState> pipeline = buildFillPipeline(device, &merr);
        if (!pipeline) {
            ane_bridge_free(kernel);
            free(outBuf);
            emitJSON(NO, "Metal fill pipeline compile failed", @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        double metalMs = 0, evalMs = 0, readMs = 0;
        for (int i = 0; i < iters; i++) {
            double fill = 0;
            if (!metalFillSurface(device, queue, pipeline, inSID, ioBytes, 0.01f, &fill)) {
                ane_bridge_free(kernel);
                free(outBuf);
                emitJSON(NO, "metal IOSurface fill failed", @{@"mode": @"metal_iosurface_handoff"});
                return 1;
            }
            metalMs += fill;

            uint64_t t0 = mach_absolute_time();
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(outBuf);
                emitJSON(NO, "ane_bridge_eval failed", @{@"mode": @"metal_iosurface_handoff"});
                return 1;
            }
            evalMs += ticksToMs(mach_absolute_time() - t0);

            t0 = mach_absolute_time();
            ane_bridge_read_output(kernel, 0, outBuf, ioBytes);
            readMs += ticksToMs(mach_absolute_time() - t0);
        }
        metalMs /= (double)iters;
        evalMs /= (double)iters;
        readMs /= (double)iters;

        BOOL finite = YES;
        for (int i = 0; i < ch * sp; i++) {
            if (!isfinite(outBuf[i])) {
                finite = NO;
                break;
            }
        }

        ane_bridge_free(kernel);
        free(outBuf);

        if (!finite) {
            emitJSON(NO, "non-finite output", @{@"mode": @"metal_iosurface_handoff"});
            return 1;
        }

        emitJSON(YES, NULL, @{
            @"mode": @"metal_iosurface_handoff",
            @"channels": @(ch),
            @"spatial": @(sp),
            @"surface_id": @(inSID),
            @"surface_bytes": @(ioBytes),
            @"metal_fill_ms": @(metalMs),
            @"eval_ms": @(evalMs),
            @"read_ms": @(readMs),
            @"total_ms": @(metalMs + evalMs + readMs),
            @"source": @"zerollama/ane-metal",
            @"note": @"Metal compute fill on IOSurface via newBufferWithBytesNoCopy; ggml hook is follow-on",
        });
        return 0;
    }
}
