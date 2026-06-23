// ANE peak throughput bench — stacked conv matmul proxy via libane_bridge.
// Why conv not raw matmul MIL: maderix peak path (inmem_peak.m) uses conv stacks;
// reports TFLOPS comparable to published M4 Max numbers (~11–19 FP16 TOPS).
//
// JSON stdout: ok, mode, channels, spatial, depth, eval_ms, gflop, tflops, compile_count

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include "ane_bridge.h"
#include "mil_conv_peak.h"

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
        escaped = [escaped stringByReplacingOccurrencesOfString:@"\n" withString:@"\\n"];
        [json appendFormat:@",\"error\":\"%@\"", escaped];
    }
    [json appendString:@"}\n"];
    printf("%s", [json UTF8String]);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int ch = 512;
        int sp = 64;
        int depth = 32;
        int warmup = 3;
        int iters = 50;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                depth = 8;
                iters = 20;
            } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
                depth = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
                ch = atoi(argv[++i]);
            }
        }

        if (ane_bridge_init() != 0) {
            emitJSON(NO, "ane_bridge_init failed", @{@"mode": @"conv_peak"});
            return 1;
        }

        NSString *mil = aneGenConvStackMIL(ch, sp, depth);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        NSData *wb = aneBuildConvStackWeightBlob(ch, depth);
        if (!milData || !wb) {
            emitJSON(NO, "MIL/weight allocation failed", @{@"mode": @"conv_peak"});
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
            emitJSON(NO, "ane_bridge_compile failed", @{
                @"mode": @"conv_peak",
                @"compile_count": @(ane_bridge_get_compile_count()),
            });
            return 1;
        }

        float *inBuf = (float *)calloc(ch * sp, sizeof(float));
        float *outBuf = (float *)calloc(ch * sp, sizeof(float));
        if (!inBuf || !outBuf) {
            ane_bridge_free(kernel);
            emitJSON(NO, "IO buffer allocation failed", @{@"mode": @"conv_peak"});
            free(inBuf);
            free(outBuf);
            return 1;
        }
        for (int i = 0; i < ch * sp; i++) {
            inBuf[i] = 0.01f;
        }
        ane_bridge_write_input(kernel, 0, inBuf, ioBytes);

        for (int i = 0; i < warmup; i++) {
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(inBuf);
                free(outBuf);
                emitJSON(NO, "ane_bridge_eval warmup failed", @{@"mode": @"conv_peak"});
                return 1;
            }
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(inBuf);
                free(outBuf);
                emitJSON(NO, "ane_bridge_eval failed", @{@"mode": @"conv_peak"});
                return 1;
            }
        }
        double evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;

        ane_bridge_read_output(kernel, 0, outBuf, ioBytes);
        int compileCount = ane_bridge_get_compile_count();

        BOOL finite = YES;
        for (int i = 0; i < ch * sp; i++) {
            if (!isfinite(outBuf[i])) {
                finite = NO;
                break;
            }
        }

        ane_bridge_free(kernel);
        free(inBuf);
        free(outBuf);

        if (!finite) {
            emitJSON(NO, "non-finite output", @{
                @"mode": @"conv_peak",
                @"eval_ms": @(evalMs),
                @"compile_count": @(compileCount),
            });
            return 1;
        }

        double gflop = aneConvStackGFLOP(ch, sp, depth);
        double tflops = gflop / (evalMs / 1000.0);

        emitJSON(YES, NULL, @{
            @"mode": @"conv_peak",
            @"channels": @(ch),
            @"spatial": @(sp),
            @"depth": @(depth),
            @"eval_ms": @(evalMs),
            @"gflop": @(gflop),
            @"tflops": @(tflops),
            @"compile_count": @(compileCount),
            @"source": @"maderix/ane-bridge",
        });
        return 0;
    }
}
