// ggml_metal_buffer_map equivalent on ANE IOSurface — parent-side lab POC (same process as surface owner).
// Cross-process IOSurfaceLookup fails for ANE bridge surfaces; orchestration uses ane-draft-daemon map_fill cmd.
//
// JSON: ok, mode=ggml_iosurface_map, surface_id, mapped_bytes, metal_fill_ms, ggml_map_ok

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#include "../ane-metal/ggml_iosurface_map.h"
#include <unistd.h>

static mach_timebase_info_data_t g_tb;

static double ticksToMs(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
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

// Returns ggml-style mapped MTLBuffer or nil. Sets *mappedBase and *mappedSize.
static id<MTLBuffer> mapSurfaceBase(id<MTLDevice> device,
                                    void *ptr,
                                    size_t size,
                                    void **mappedBase,
                                    size_t *mappedSize) {
    return ggml_map_iosurface_base(device, ptr, size, mappedBase, mappedSize);
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

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        uint32_t surfaceID = 0;
        size_t bytes = 4096;
        float fillVal = 0.01f;
        size_t tensorOffset = 0;
        int iters = 1;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--surface-id") == 0 && i + 1 < argc) {
                surfaceID = (uint32_t)strtoul(argv[++i], NULL, 10);
            } else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
                bytes = (size_t)strtoul(argv[++i], NULL, 10);
            } else if (strcmp(argv[i], "--fill") == 0 && i + 1 < argc) {
                fillVal = (float)atof(argv[++i]);
            } else if (strcmp(argv[i], "--offset") == 0 && i + 1 < argc) {
                tensorOffset = (size_t)strtoul(argv[++i], NULL, 10);
            } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
                iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--quick") == 0) {
                iters = 3;
            }
        }

        if (surfaceID == 0) {
            emitLine(@{@"ok": @NO, @"error": @"--surface-id required", @"mode": @"ggml_iosurface_map"});
            return 1;
        }

        IOSurfaceRef surface = ggml_map_surface_from_id(surfaceID);
        if (!surface) {
            emitLine(@{@"ok": @NO, @"error": @"IOSurfaceLookup failed", @"mode": @"ggml_iosurface_map", @"surface_id": @(surfaceID)});
            return 1;
        }

        IOSurfaceLock(surface, 0, NULL);
        void *base = IOSurfaceGetBaseAddress(surface);
        if (!base) {
            IOSurfaceUnlock(surface, 0, NULL);
            CFRelease(surface);
            emitLine(@{@"ok": @NO, @"error": @"IOSurfaceGetBaseAddress failed", @"mode": @"ggml_iosurface_map"});
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            IOSurfaceUnlock(surface, 0, NULL);
            CFRelease(surface);
            emitLine(@{@"ok": @NO, @"error": @"MTLCreateSystemDefaultDevice failed", @"mode": @"ggml_iosurface_map"});
            return 1;
        }

        void *mappedBase = NULL;
        size_t mappedSize = 0;
        id<MTLBuffer> mapped = mapSurfaceBase(device, base, bytes, &mappedBase, &mappedSize);
        if (!mapped) {
            IOSurfaceUnlock(surface, 0, NULL);
            CFRelease(surface);
            emitLine(@{@"ok": @NO, @"error": @"newBufferWithBytesNoCopy failed (ggml_metal_buffer_map path)", @"mode": @"ggml_iosurface_map"});
            return 1;
        }

        NSError *merr = nil;
        id<MTLComputePipelineState> pipeline = buildFillPipeline(device, &merr);
        if (!pipeline) {
            IOSurfaceUnlock(surface, 0, NULL);
            CFRelease(surface);
            emitLine(@{@"ok": @NO, @"error": @"Metal fill pipeline compile failed", @"mode": @"ggml_iosurface_map"});
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        size_t pageOffset = (size_t)((char *)base - (char *)mappedBase);
        double fillMs = 0;
        for (int i = 0; i < iters; i++) {
            double one = 0;
            if (!metalFillMappedBuffer(device, queue, pipeline, mapped, bytes, pageOffset, fillVal, &one)) {
                IOSurfaceUnlock(surface, 0, NULL);
                CFRelease(surface);
                emitLine(@{@"ok": @NO, @"error": @"Metal fill on mapped buffer failed", @"mode": @"ggml_iosurface_map"});
                return 1;
            }
            fillMs += one;
        }
        fillMs /= (double)iters;

        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);

        emitLine(@{
            @"ok": @YES,
            @"mode": @"ggml_iosurface_map",
            @"surface_id": @(surfaceID),
            @"surface_bytes": @(bytes),
            @"mapped_bytes": @(mappedSize),
            @"mapped_page_offset": @((NSUInteger)((char *)base - (char *)mappedBase)),
            @"tensor_offset": @(tensorOffset),
            @"metal_fill_ms": @(fillMs),
            @"ggml_map_ok": @YES,
            @"iters": @(iters),
            @"source": @"zerollama/ane-ggml-map",
            @"note": @"page-aligned newBufferWithBytesNoCopy — same path as ggml_metal_buffer_map on IOSurface base",
        });
        return 0;
    }
}
