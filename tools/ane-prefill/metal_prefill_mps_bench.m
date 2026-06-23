// Metal Performance Shaders matmul prefill proxy — tuned GPU baseline vs naive shader.
// C[SEQ×OC] = A[SEQ×IC] @ B[IC×OC]
//
// JSON: ok, mode=metal_mps_prefill_matmul, ic, oc, seq, eval_ms, gflop, tflops

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
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
            emitJSON(NO, "ic/oc/seq must be positive", @{@"mode": @"metal_mps_prefill_matmul"});
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            emitJSON(NO, "MTLCreateSystemDefaultDevice failed", @{@"mode": @"metal_mps_prefill_matmul"});
            return 1;
        }

        NSUInteger rowBytesA = (NSUInteger)ic * sizeof(float);
        NSUInteger rowBytesB = (NSUInteger)oc * sizeof(float);
        NSUInteger rowBytesC = (NSUInteger)oc * sizeof(float);
        size_t bytesA = (size_t)seq * rowBytesA;
        size_t bytesB = (size_t)ic * rowBytesB;
        size_t bytesC = (size_t)seq * rowBytesC;

        id<MTLBuffer> bufA = [device newBufferWithLength:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:bytesB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) {
            emitJSON(NO, "buffer allocation failed", @{@"mode": @"metal_mps_prefill_matmul"});
            return 1;
        }

        float *A = (float *)bufA.contents;
        float *B = (float *)bufB.contents;
        for (int r = 0; r < seq; r++) {
            for (int c = 0; c < ic; c++) {
                A[r * ic + c] = 0.01f;
            }
        }
        for (int r = 0; r < ic; r++) {
            for (int c = 0; c < oc; c++) {
                B[r * oc + c] = 0.001f;
            }
        }

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:seq
                                                                         columns:ic
                                                                        rowBytes:rowBytesA
                                                                        dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:ic
                                                                         columns:oc
                                                                        rowBytes:rowBytesB
                                                                        dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:seq
                                                                         columns:oc
                                                                        rowBytes:rowBytesC
                                                                        dataType:MPSDataTypeFloat32];
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA offset:0 descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB offset:0 descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC offset:0 descriptor:descC];

        MPSMatrixMultiplication *mul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                           transposeLeft:NO
                                                                          transposeRight:NO
                                                                             resultRows:seq
                                                                          resultColumns:oc
                                                                        interiorColumns:ic
                                                                                  alpha:1.0
                                                                                   beta:0.0];
        if (!mul) {
            emitJSON(NO, "MPSMatrixMultiplication init failed", @{@"mode": @"metal_mps_prefill_matmul"});
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        void (^runOnce)(void) = ^{
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            [mul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            [cmd commit];
            [cmd waitUntilCompleted];
        };

        for (int i = 0; i < warmup; i++) {
            runOnce();
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            runOnce();
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
            emitJSON(NO, "non-finite output", @{@"mode": @"metal_mps_prefill_matmul"});
            return 1;
        }

        double gflop = 2.0 * (double)ic * (double)oc * (double)seq / 1e9;
        double tflops = evalMs > 0 ? gflop / (evalMs / 1000.0) : 0;

        emitJSON(YES, NULL, @{
            @"mode": @"metal_mps_prefill_matmul",
            @"ic": @(ic),
            @"oc": @(oc),
            @"seq": @(seq),
            @"eval_ms": @(evalMs),
            @"gflop": @(gflop),
            @"tflops": @(tflops),
            @"source": @"zerollama/metal-mps-prefill",
            @"note": @"MPSMatrixMultiplication — closer Metal baseline than naive shader; still not ggml",
        });
        return 0;
    }
}
