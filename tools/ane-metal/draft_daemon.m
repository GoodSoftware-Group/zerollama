// Persistent ANE draft conv daemon — compile kernel once, reuse across eval sessions.
// Protocol: first stdout line is ready JSON; then one JSON request per stdin line.
//
// Requests: {"cmd":"ping"} | {"cmd":"eval","fill":0.01} | {"cmd":"bench","iters":30,"fill":0.01} | {"cmd":"quit"}
// Responses: JSON line with ok, event, timings; compile_count stays at 1 after ready.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#include "ane_bridge.h"
#include "ggml_iosurface_map.h"

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

static NSData *loadWeightBlobFromFile(const char *path, int ch) {
    if (!path || !path[0]) {
        return nil;
    }
    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data) {
        return nil;
    }
    NSUInteger expected = 64 + 64 + (NSUInteger)ch * (NSUInteger)ch * 2;
    if ([data length] != expected) {
        return nil;
    }
    return data;
}

static void emitLine(NSDictionary *obj) {
    NSError *err = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:obj options:0 error:&err];
    if (!data) {
        printf("{\"ok\":false,\"error\":\"json encode failed\"}\n");
        fflush(stdout);
        return;
    }
    NSMutableData *line = [data mutableCopy];
    [line appendBytes:"\n" length:1];
    fwrite(line.bytes, 1, line.length, stdout);
    fflush(stdout);
}

static void emitErr(NSString *msg) {
    emitLine(@{@"ok": @NO, @"error": msg ?: @"unknown error"});
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

typedef struct {
    ANEKernelHandle *kernel;
    uint32_t inSID;
    size_t ioBytes;
    int ch;
    int sp;
    float *outBuf;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    int evalCount;
    const char *weightSource;
} DraftDaemonState;

static BOOL runOneEval(DraftDaemonState *st, float fillVal, double *metalMs, double *evalMs, double *readMs) {
    double fill = 0;
    if (!metalFillSurface(st->device, st->queue, st->pipeline, st->inSID, st->ioBytes, fillVal, &fill)) {
        return NO;
    }
    if (metalMs) {
        *metalMs = fill;
    }

    uint64_t t0 = mach_absolute_time();
    if (!ane_bridge_eval(st->kernel)) {
        return NO;
    }
    if (evalMs) {
        *evalMs = ticksToMs(mach_absolute_time() - t0);
    }

    t0 = mach_absolute_time();
    ane_bridge_read_output(st->kernel, 0, st->outBuf, st->ioBytes);
    if (readMs) {
        *readMs = ticksToMs(mach_absolute_time() - t0);
    }

    st->evalCount++;
    return YES;
}

static BOOL initDaemon(DraftDaemonState *st, int ch, int sp, const char *weightPath, double *compileMs) {
    st->kernel = NULL;
    st->inSID = 0;
    st->ioBytes = 0;
    st->ch = ch;
    st->sp = sp;
    st->outBuf = NULL;
    st->device = nil;
    st->queue = nil;
    st->pipeline = nil;
    st->evalCount = 0;
    st->weightSource = "synthetic";

    if (ane_bridge_init() != 0) {
        return NO;
    }

    NSString *mil = genConvMIL(ch, sp);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSData *wb = nil;
    if (weightPath && weightPath[0]) {
        wb = loadWeightBlobFromFile(weightPath, ch);
        if (!wb) {
            return NO;
        }
        st->weightSource = "sidecar_extract";
    } else {
        wb = buildWeightBlob(ch);
    }
    if (!milData || !wb) {
        return NO;
    }

    const char *weightName = "@model_path/weights/weight.bin";
    const uint8_t *weightData = (const uint8_t *)[wb bytes];
    size_t weightLen = [wb length];
    st->ioBytes = (size_t)ch * (size_t)sp * sizeof(float);

    uint64_t t0 = mach_absolute_time();
    st->kernel = ane_bridge_compile_multi_weights(
        [milData bytes], [milData length],
        &weightName, &weightData, &weightLen, 1,
        1, &st->ioBytes, 1, &st->ioBytes);
    if (compileMs) {
        *compileMs = ticksToMs(mach_absolute_time() - t0);
    }
    if (!st->kernel) {
        return NO;
    }

    st->inSID = ane_bridge_input_surface_id(st->kernel, 0);
    if (st->inSID == 0) {
        return NO;
    }

    st->outBuf = (float *)calloc(ch * sp, sizeof(float));
    if (!st->outBuf) {
        return NO;
    }

    st->device = MTLCreateSystemDefaultDevice();
    if (!st->device) {
        return NO;
    }
    NSError *merr = nil;
    st->pipeline = buildFillPipeline(st->device, &merr);
    if (!st->pipeline) {
        return NO;
    }
    st->queue = [st->device newCommandQueue];
    return YES;
}

static void freeDaemon(DraftDaemonState *st) {
    if (st->kernel) {
        ane_bridge_free(st->kernel);
        st->kernel = NULL;
    }
    if (st->outBuf) {
        free(st->outBuf);
        st->outBuf = NULL;
    }
}

static NSDictionary *readyPayload(DraftDaemonState *st, double compileMs) {
    return @{
        @"ok": @YES,
        @"event": @"ready",
        @"mode": @"draft_daemon",
        @"channels": @(st->ch),
        @"spatial": @(st->sp),
        @"surface_id": @(st->inSID),
        @"surface_bytes": @(st->ioBytes),
        @"compile_ms": @(compileMs),
        @"compile_count": @(ane_bridge_get_compile_count()),
        @"weight_source": @(st->weightSource ?: "synthetic"),
        @"source": @"zerollama/ane-metal",
        @"note": @"compile-once session; parent fills IOSurface via Metal then eval",
    };
}

static BOOL metalFillMappedBuffer(id<MTLDevice> device,
                                  id<MTLCommandQueue> queue,
                                  id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> buffer,
                                  size_t tensorBytes,
                                  size_t tensorOffset,
                                  float value,
                                  double *fillMs) {
    if (tensorBytes == 0 || tensorBytes + tensorOffset > buffer.length) {
        return NO;
    }

    size_t nfloats = tensorBytes / sizeof(float);
    uint64_t t0 = mach_absolute_time();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buffer offset:tensorOffset atIndex:0];
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
    return cmd.status == MTLCommandBufferStatusCompleted;
}

static NSDictionary *runGGMLMapFill(DraftDaemonState *st, float fillVal) {
    IOSurfaceRef surface = ggml_map_surface_from_id(st->inSID);
    if (!surface) {
        return @{@"ok": @NO, @"error": @"IOSurfaceLookup failed", @"event": @"map_fill"};
    }

    IOSurfaceLock(surface, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return @{@"ok": @NO, @"error": @"IOSurfaceGetBaseAddress failed", @"event": @"map_fill"};
    }

    void *mappedBase = NULL;
    size_t mappedSize = 0;
    id<MTLBuffer> mapped = ggml_map_iosurface_base(st->device, base, st->ioBytes, &mappedBase, &mappedSize);
    if (!mapped) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return @{@"ok": @NO, @"error": @"ggml_metal_buffer_map path failed", @"event": @"map_fill"};
    }

    size_t pageOffset = (size_t)((char *)base - (char *)mappedBase);
    double fillMs = 0;
    BOOL ok = metalFillMappedBuffer(st->device, st->queue, st->pipeline, mapped, st->ioBytes, pageOffset, fillVal, &fillMs);

    IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);

    if (!ok) {
        return @{@"ok": @NO, @"error": @"Metal fill on ggml-mapped buffer failed", @"event": @"map_fill"};
    }

    return @{
        @"ok": @YES,
        @"event": @"map_fill",
        @"mode": @"ggml_iosurface_map",
        @"surface_id": @(st->inSID),
        @"surface_bytes": @(st->ioBytes),
        @"mapped_bytes": @(mappedSize),
        @"mapped_page_offset": @(pageOffset),
        @"metal_fill_ms": @(fillMs),
        @"ggml_map_ok": @YES,
        @"compile_count": @(ane_bridge_get_compile_count()),
        @"note": @"page-aligned newBufferWithBytesNoCopy — same path as ggml_metal_buffer_map",
    };
}

static BOOL runANEOnlyEval(DraftDaemonState *st, double *evalMs, double *readMs) {
    uint64_t t0 = mach_absolute_time();
    if (!ane_bridge_eval(st->kernel)) {
        return NO;
    }
    if (evalMs) {
        *evalMs = ticksToMs(mach_absolute_time() - t0);
    }

    t0 = mach_absolute_time();
    ane_bridge_read_output(st->kernel, 0, st->outBuf, st->ioBytes);
    if (readMs) {
        *readMs = ticksToMs(mach_absolute_time() - t0);
    }

    st->evalCount++;
    return YES;
}

static NSDictionary *handleCommand(DraftDaemonState *st, NSDictionary *req) {
    NSString *cmd = req[@"cmd"];
    if (![cmd isKindOfClass:[NSString class]]) {
        return @{@"ok": @NO, @"error": @"missing cmd"};
    }

    if ([cmd isEqualToString:@"ping"]) {
        return @{
            @"ok": @YES,
            @"event": @"pong",
            @"eval_count": @(st->evalCount),
            @"compile_count": @(ane_bridge_get_compile_count()),
            @"surface_id": @(st->inSID),
        };
    }

    if ([cmd isEqualToString:@"quit"]) {
        return @{@"ok": @YES, @"event": @"bye", @"eval_count": @(st->evalCount)};
    }

    float fillVal = 0.01f;
    if (req[@"fill"] != nil) {
        fillVal = [req[@"fill"] floatValue];
    }

    if ([cmd isEqualToString:@"map_fill"]) {
        return runGGMLMapFill(st, fillVal);
    }

    if ([cmd isEqualToString:@"eval_ane"]) {
        double evalMs = 0, readMs = 0;
        if (!runANEOnlyEval(st, &evalMs, &readMs)) {
            return @{@"ok": @NO, @"error": @"eval_ane failed"};
        }
        return @{
            @"ok": @YES,
            @"event": @"eval_ane",
            @"eval_ms": @(evalMs),
            @"read_ms": @(readMs),
            @"total_ms": @(evalMs + readMs),
            @"eval_count": @(st->evalCount),
            @"compile_count": @(ane_bridge_get_compile_count()),
            @"surface_id": @(st->inSID),
        };
    }

    if ([cmd isEqualToString:@"eval"]) {
        double metalMs = 0, evalMs = 0, readMs = 0;
        if (!runOneEval(st, fillVal, &metalMs, &evalMs, &readMs)) {
            return @{@"ok": @NO, @"error": @"eval failed"};
        }
        return @{
            @"ok": @YES,
            @"event": @"eval",
            @"metal_fill_ms": @(metalMs),
            @"eval_ms": @(evalMs),
            @"read_ms": @(readMs),
            @"total_ms": @(metalMs + evalMs + readMs),
            @"eval_count": @(st->evalCount),
            @"compile_count": @(ane_bridge_get_compile_count()),
            @"surface_id": @(st->inSID),
        };
    }

    if ([cmd isEqualToString:@"bench"]) {
        int iters = 30;
        if (req[@"iters"] != nil) {
            iters = [req[@"iters"] intValue];
        }
        if (iters < 1) {
            iters = 1;
        }
        double metalSum = 0, evalSum = 0, readSum = 0;
        for (int i = 0; i < iters; i++) {
            double metalMs = 0, evalMs = 0, readMs = 0;
            if (!runOneEval(st, fillVal, &metalMs, &evalMs, &readMs)) {
                return @{@"ok": @NO, @"error": @"bench eval failed"};
            }
            metalSum += metalMs;
            evalSum += evalMs;
            readSum += readMs;
        }
        double n = (double)iters;
        double metalAvg = metalSum / n;
        double evalAvg = evalSum / n;
        double readAvg = readSum / n;
        return @{
            @"ok": @YES,
            @"event": @"bench",
            @"iters": @(iters),
            @"metal_fill_ms": @(metalAvg),
            @"eval_ms": @(evalAvg),
            @"read_ms": @(readAvg),
            @"total_ms": @(metalAvg + evalAvg + readAvg),
            @"eval_count": @(st->evalCount),
            @"compile_count": @(ane_bridge_get_compile_count()),
            @"surface_id": @(st->inSID),
        };
    }

    return @{@"ok": @NO, @"error": [NSString stringWithFormat:@"unknown cmd: %@", cmd]};
}

static int runDaemonLoop(DraftDaemonState *st, double compileMs) {
    emitLine(readyPayload(st, compileMs));

    char line[8192];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }

        NSData *data = [NSData dataWithBytes:line length:len];
        NSError *err = nil;
        id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
        if (![obj isKindOfClass:[NSDictionary class]]) {
            emitErr(@"invalid json request");
            continue;
        }

        NSDictionary *resp = handleCommand(st, (NSDictionary *)obj);
        emitLine(resp);

        NSString *cmd = ((NSDictionary *)obj)[@"cmd"];
        if ([cmd isKindOfClass:[NSString class]] && [cmd isEqualToString:@"quit"]) {
            return 0;
        }
    }
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int ch = 64;
        int sp = 16;
        int iters = 30;
        BOOL benchOnce = NO;
        const char *weightPath = NULL;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 10;
            } else if (strcmp(argv[i], "--bench") == 0) {
                benchOnce = YES;
            } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
                ch = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--spatial") == 0 && i + 1 < argc) {
                sp = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
                iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--weight-file") == 0 && i + 1 < argc) {
                weightPath = argv[++i];
            }
        }

        DraftDaemonState st;
        double compileMs = 0;
        if (!initDaemon(&st, ch, sp, weightPath, &compileMs)) {
            emitErr(weightPath ? @"daemon init failed — check --weight-file size matches channels" : @"daemon init failed — run scripts/ane/ane_bridge_patch.sh");
            return 1;
        }

        int rc = 0;
        if (benchOnce) {
            NSDictionary *resp = handleCommand(&st, (@{@"cmd": @"bench", @"iters": @(iters)}));
            NSMutableDictionary *out = [resp mutableCopy];
            out[@"mode"] = @"draft_daemon";
            out[@"channels"] = @(ch);
            out[@"spatial"] = @(sp);
            out[@"compile_ms"] = @(compileMs);
            out[@"source"] = @"zerollama/ane-metal";
            emitLine(out);
            if (![resp[@"ok"] boolValue]) {
                rc = 1;
            }
        } else {
            rc = runDaemonLoop(&st, compileMs);
        }

        freeDaemon(&st);
        return rc;
    }
}
