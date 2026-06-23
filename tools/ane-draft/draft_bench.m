// ANE draft-step latency bench — small conv proxy for speculative draft heads.
// Why conv not dynamic matmul MIL: maderix dynamic matmul needs IOSurface weight
// staging; this measures ANE dispatch latency at draft-like tensor sizes until
// GGUF drafter wiring lands.
//
// JSON stdout: ok, mode=draft_conv, channels, spatial, eval_ms, compile_count

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include "ane_bridge.h"

static mach_timebase_info_data_t g_tb;

static double ticksToMs(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static NSString *genDraftConvMIL(int ch, int sp) {
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

static NSData *buildDraftWeightBlob(int ch) {
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

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        // ~2B-param draft head proxy dims (single token, moderate channel width).
        int ch = 64;
        int sp = 16;
        int iters = 100;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 30;
            } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
                ch = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--spatial") == 0 && i + 1 < argc) {
                sp = atoi(argv[++i]);
            }
        }

        if (ane_bridge_init() != 0) {
            emitJSON(NO, "ane_bridge_init failed", @{@"mode": @"draft_conv"});
            return 1;
        }

        NSString *mil = genDraftConvMIL(ch, sp);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        NSData *wb = buildDraftWeightBlob(ch);
        if (!milData || !wb) {
            emitJSON(NO, "MIL/weight allocation failed", @{@"mode": @"draft_conv"});
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
                @"mode": @"draft_conv",
                @"compile_count": @(ane_bridge_get_compile_count()),
            });
            return 1;
        }

        float *inBuf = (float *)calloc(ch * sp, sizeof(float));
        float *outBuf = (float *)calloc(ch * sp, sizeof(float));
        if (!inBuf || !outBuf) {
            ane_bridge_free(kernel);
            emitJSON(NO, "IO buffer allocation failed", @{@"mode": @"draft_conv"});
            free(inBuf);
            free(outBuf);
            return 1;
        }
        for (int i = 0; i < ch * sp; i++) {
            inBuf[i] = 0.01f;
        }
        ane_bridge_write_input(kernel, 0, inBuf, ioBytes);

        for (int i = 0; i < 5; i++) {
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(inBuf);
                free(outBuf);
                emitJSON(NO, "warmup eval failed", @{@"mode": @"draft_conv"});
                return 1;
            }
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(inBuf);
                free(outBuf);
                emitJSON(NO, "eval failed", @{@"mode": @"draft_conv"});
                return 1;
            }
        }
        double evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;
        int compileCount = ane_bridge_get_compile_count();

        ane_bridge_free(kernel);
        free(inBuf);
        free(outBuf);

        emitJSON(YES, NULL, @{
            @"mode": @"draft_conv",
            @"channels": @(ch),
            @"spatial": @(sp),
            @"eval_ms": @(evalMs),
            @"compile_count": @(compileCount),
            @"source": @"maderix/ane-bridge",
            @"note": @"draft-step proxy; GGUF drafter + IOSurface handoff is follow-on",
        });
        return 0;
    }
}
