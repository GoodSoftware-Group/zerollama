// ANE prefill proxy bench — dynamic matmul y = x @ W at [IC × SEQ] (maderix mil_dynamic.h shape).
// Why: prefill batches prompt tokens into the spatial dimension; this measures ANE throughput
// at prefill-like geometry before any ggml hook.
//
// JSON: ok, mode=prefill_matmul, ic, oc, seq, eval_ms, gflop, tflops, compile_count

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include "ane_bridge.h"

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

static NSString *genPrefillMatmulMIL(int ic, int oc, int seq) {
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

static void fillInputFP32(float *buf, int ic, int oc, int seq) {
    int spIn = seq + oc;
    for (int c = 0; c < ic; c++) {
        for (int s = 0; s < seq; s++) {
            buf[c * spIn + s] = 0.01f;
        }
        for (int w = 0; w < oc; w++) {
            buf[c * spIn + seq + w] = 0.001f;
        }
    }
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
            emitJSON(NO, "ic/oc/seq must be positive", @{@"mode": @"prefill_matmul"});
            return 1;
        }

        if (ane_bridge_init() != 0) {
            emitJSON(NO, "ane_bridge_init failed", @{@"mode": @"prefill_matmul"});
            return 1;
        }

        int spIn = seq + oc;
        size_t inBytes = (size_t)ic * (size_t)spIn * sizeof(float);
        size_t outBytes = (size_t)oc * (size_t)seq * sizeof(float);

        NSString *mil = genPrefillMatmulMIL(ic, oc, seq);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        if (!milData) {
            emitJSON(NO, "MIL allocation failed", @{@"mode": @"prefill_matmul"});
            return 1;
        }

        ANEKernelHandle *kernel = ane_bridge_compile(
            [milData bytes], [milData length],
            NULL, 0,
            1, &inBytes, 1, &outBytes);
        if (!kernel) {
            emitJSON(NO, "ane_bridge_compile failed", @{
                @"mode": @"prefill_matmul",
                @"ic": @(ic),
                @"oc": @(oc),
                @"seq": @(seq),
                @"compile_count": @(ane_bridge_get_compile_count()),
            });
            return 1;
        }

        float *inBuf = (float *)calloc(ic * spIn, sizeof(float));
        float *outBuf = (float *)calloc(oc * seq, sizeof(float));
        if (!inBuf || !outBuf) {
            ane_bridge_free(kernel);
            free(inBuf);
            free(outBuf);
            emitJSON(NO, "buffer allocation failed", @{@"mode": @"prefill_matmul"});
            return 1;
        }
        fillInputFP32(inBuf, ic, oc, seq);
        ane_bridge_write_input(kernel, 0, inBuf, inBytes);

        for (int i = 0; i < warmup; i++) {
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(inBuf);
                free(outBuf);
                emitJSON(NO, "warmup eval failed", @{@"mode": @"prefill_matmul"});
                return 1;
            }
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            if (!ane_bridge_eval(kernel)) {
                ane_bridge_free(kernel);
                free(inBuf);
                free(outBuf);
                emitJSON(NO, "eval failed", @{@"mode": @"prefill_matmul"});
                return 1;
            }
        }
        double evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;

        ane_bridge_read_output(kernel, 0, outBuf, outBytes);
        BOOL finite = YES;
        for (size_t i = 0; i < (size_t)oc * (size_t)seq; i++) {
            if (!isfinite(outBuf[i])) {
                finite = NO;
                break;
            }
        }

        int compileCount = ane_bridge_get_compile_count();
        ane_bridge_free(kernel);
        free(inBuf);
        free(outBuf);

        if (!finite) {
            emitJSON(NO, "non-finite output", @{@"mode": @"prefill_matmul"});
            return 1;
        }

        double gflop = 2.0 * (double)ic * (double)oc * (double)seq / 1e9;
        double tflops = evalMs > 0 ? gflop / (evalMs / 1000.0) : 0;

        emitJSON(YES, NULL, @{
            @"mode": @"prefill_matmul",
            @"ic": @(ic),
            @"oc": @(oc),
            @"seq": @(seq),
            @"eval_ms": @(evalMs),
            @"gflop": @(gflop),
            @"tflops": @(tflops),
            @"compile_count": @(compileCount),
            @"source": @"maderix/mil_dynamic",
            @"note": @"dynamic matmul prefill proxy; compare to Metal FFN at same IC×OC×SEQ",
        });
        return 0;
    }
}
