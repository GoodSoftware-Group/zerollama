// Metal prefill proxy bench — naive compute matmul C[SEQ×OC] = A[SEQ×IC] @ B[IC×OC].
// Pair with ane-prefill-bench at the same IC×OC×SEQ for apples-to-apples comparison.
//
// JSON: ok, mode=metal_prefill_matmul, ic, oc, seq, eval_ms, gflop, tflops

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

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

static id<MTLComputePipelineState> buildMatmulPipeline(id<MTLDevice> device, NSError **err) {
    NSString *src = @""
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Dims { uint ic; uint oc; uint seq; };\n"
        "kernel void matmul_prefill(\n"
        "    device const float *A [[buffer(0)]],\n"
        "    device const float *B [[buffer(1)]],\n"
        "    device float *C [[buffer(2)]],\n"
        "    constant Dims &d [[buffer(3)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "  uint s = gid.y;\n"
        "  uint o = gid.x;\n"
        "  if (s >= d.seq || o >= d.oc) return;\n"
        "  float sum = 0.0f;\n"
        "  for (uint k = 0; k < d.ic; k++) {\n"
        "    sum += A[s * d.ic + k] * B[k * d.oc + o];\n"
        "  }\n"
        "  C[s * d.oc + o] = sum;\n"
        "}\n";
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:err];
    if (!lib) {
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_prefill"];
    if (!fn) {
        return nil;
    }
    return [device newComputePipelineStateWithFunction:fn error:err];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int ic = 256;
        int oc = 256;
        int seq = 512;
        int warmup = 3;
        int iters = 40;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 15;
            } else if (strcmp(argv[i], "--ic") == 0 && i + 1 < argc) {
                ic = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--oc") == 0 && i + 1 < argc) {
                oc = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
                seq = atoi(argv[++i]);
            }
        }

        if (ic <= 0 || oc <= 0 || seq <= 0) {
            emitJSON(NO, "ic/oc/seq must be positive", @{@"mode": @"metal_prefill_matmul"});
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            emitJSON(NO, "MTLCreateSystemDefaultDevice failed", @{@"mode": @"metal_prefill_matmul"});
            return 1;
        }

        NSError *err = nil;
        id<MTLComputePipelineState> pipeline = buildMatmulPipeline(device, &err);
        if (!pipeline) {
            emitJSON(NO, "Metal pipeline compile failed", @{@"mode": @"metal_prefill_matmul"});
            return 1;
        }

        size_t aBytes = (size_t)seq * (size_t)ic * sizeof(float);
        size_t bBytes = (size_t)ic * (size_t)oc * sizeof(float);
        size_t cBytes = (size_t)seq * (size_t)oc * sizeof(float);

        id<MTLBuffer> bufA = [device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:bBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufD = [device newBufferWithLength:sizeof(uint32_t) * 3 options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC || !bufD) {
            emitJSON(NO, "Metal buffer allocation failed", @{@"mode": @"metal_prefill_matmul"});
            return 1;
        }

        float *A = (float *)bufA.contents;
        float *B = (float *)bufB.contents;
        for (size_t i = 0; i < (size_t)seq * (size_t)ic; i++) {
            A[i] = 0.01f;
        }
        for (size_t i = 0; i < (size_t)ic * (size_t)oc; i++) {
            B[i] = 0.001f;
        }
        uint32_t *dims = (uint32_t *)bufD.contents;
        dims[0] = (uint32_t)ic;
        dims[1] = (uint32_t)oc;
        dims[2] = (uint32_t)seq;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        struct {
            uint32_t ic, oc, seq;
        } dimStruct = {(uint32_t)ic, (uint32_t)oc, (uint32_t)seq};

        void (^encode)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBytes:&dimStruct length:sizeof(dimStruct) atIndex:3];
            MTLSize grid = MTLSizeMake((NSUInteger)oc, (NSUInteger)seq, 1);
            MTLSize tg = MTLSizeMake(MIN(16, (NSUInteger)oc), MIN(16, (NSUInteger)seq), 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        };

        for (int i = 0; i < warmup; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode(enc);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) {
                emitJSON(NO, "warmup failed", @{@"mode": @"metal_prefill_matmul"});
                return 1;
            }
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode(enc);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) {
                emitJSON(NO, "eval failed", @{@"mode": @"metal_prefill_matmul"});
                return 1;
            }
        }
        double evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;

        float *C = (float *)bufC.contents;
        BOOL finite = YES;
        for (size_t i = 0; i < (size_t)seq * (size_t)oc; i++) {
            if (!isfinite(C[i])) {
                finite = NO;
                break;
            }
        }
        if (!finite) {
            emitJSON(NO, "non-finite output", @{@"mode": @"metal_prefill_matmul"});
            return 1;
        }

        double gflop = 2.0 * (double)ic * (double)oc * (double)seq / 1e9;
        double tflops = evalMs > 0 ? gflop / (evalMs / 1000.0) : 0;

        emitJSON(YES, NULL, @{
            @"mode": @"metal_prefill_matmul",
            @"ic": @(ic),
            @"oc": @(oc),
            @"seq": @(seq),
            @"eval_ms": @(evalMs),
            @"gflop": @(gflop),
            @"tflops": @(tflops),
            @"source": @"zerollama/metal-prefill",
            @"note": @"naive Metal compute matmul; ggml uses tuned kernels — relative compare only",
        });
        return 0;
    }
}
