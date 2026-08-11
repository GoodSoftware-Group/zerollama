// In-process ANE SwiGLU smoke — compile-once session + steady eval + CPU golden.
// Why: lab gate for fused gate/up/SiLU/down before any ggml FFN intercept.
//
// JSON: ok, mode=ane_prefill_swiglu, eval_ms, golden_cosine, max_abs_err, …

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

#include "ane_prefill_swiglu_session.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static float silu(float x) {
    return x / (1.f + expf(-x));
}

// Wg/Wu [dim][hidden], Wd [hidden][dim], X/Y [dim][seq]
static void cpuSwiGLU(const float *Wg, const float *Wu, const float *Wd,
                      const float *X, float *Y, int dim, int hidden, int seq) {
    float *Hg = (float *)calloc((size_t)hidden * (size_t)seq, sizeof(float));
    float *Hu = (float *)calloc((size_t)hidden * (size_t)seq, sizeof(float));
    float *Sw = (float *)calloc((size_t)hidden * (size_t)seq, sizeof(float));
    if (!Hg || !Hu || !Sw) {
        free(Hg); free(Hu); free(Sw);
        return;
    }
    for (int h = 0; h < hidden; h++) {
        for (int s = 0; s < seq; s++) {
            float g = 0, u = 0;
            for (int i = 0; i < dim; i++) {
                float x = X[i * seq + s];
                g += Wg[i * hidden + h] * x;
                u += Wu[i * hidden + h] * x;
            }
            Hg[h * seq + s] = g;
            Hu[h * seq + s] = u;
            Sw[h * seq + s] = silu(g) * u;
        }
    }
    for (int o = 0; o < dim; o++) {
        for (int s = 0; s < seq; s++) {
            float acc = 0;
            for (int h = 0; h < hidden; h++) {
                acc += Wd[h * dim + o] * Sw[h * seq + s];
            }
            Y[o * seq + s] = acc;
        }
    }
    free(Hg); free(Hu); free(Sw);
}

static double cosineSim(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na <= 0 || nb <= 0) return 0;
    return dot / (sqrt(na) * sqrt(nb));
}

static float maxAbsErr(const float *a, const float *b, size_t n) {
    float m = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        // Expert-ish defaults: dim=512, hidden=256 (smaller than 2048 for fast compile).
        int dim = 512;
        int hidden = 256;
        int seq = 128;
        int iters = 40;
        int warmup = 3;
        BOOL parity = YES;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 15;
            } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
                dim = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
                hidden = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
                seq = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--no-parity") == 0) {
                parity = NO;
            }
        }

        if (dim <= 0 || hidden <= 0 || seq <= 0) {
            emitJSON(NO, "dim/hidden/seq must be positive", @{@"mode": @"ane_prefill_swiglu"});
            return 1;
        }

        // Scales keep mul products above fp16 denormal (~6e-5) and below overflow.
        float *Wg = (float *)malloc((size_t)dim * (size_t)hidden * sizeof(float));
        float *Wu = (float *)malloc((size_t)dim * (size_t)hidden * sizeof(float));
        float *Wd = (float *)malloc((size_t)hidden * (size_t)dim * sizeof(float));
        float *X = (float *)malloc((size_t)dim * (size_t)seq * sizeof(float));
        float *Ycpu = (float *)malloc((size_t)dim * (size_t)seq * sizeof(float));
        if (!Wg || !Wu || !Wd || !X || !Ycpu) {
            free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "host buffer alloc failed", @{@"mode": @"ane_prefill_swiglu"});
            return 1;
        }
        for (int i = 0; i < dim * hidden; i++) {
            Wg[i] = 0.05f * (float)((i * 17 + 3) % 97) / 97.0f;
            Wu[i] = 0.05f * (float)((i * 13 + 7) % 89) / 89.0f;
        }
        for (int i = 0; i < hidden * dim; i++) {
            Wd[i] = 0.05f * (float)((i * 11 + 5) % 83) / 83.0f;
        }
        for (int i = 0; i < dim * seq; i++) {
            X[i] = 0.1f * (float)((i * 19 + 2) % 79) / 79.0f;
        }
        if (parity) {
            cpuSwiGLU(Wg, Wu, Wd, X, Ycpu, dim, hidden, seq);
        }

        ANEPrefillSwiGLUSession *sess =
            ane_prefill_swiglu_session_create(dim, hidden, seq, Wg, Wu, Wd);
        if (!sess) {
            free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "ane_prefill_swiglu_session_create failed", @{
                @"mode": @"ane_prefill_swiglu",
                @"dim": @(dim), @"hidden": @(hidden), @"seq": @(seq),
            });
            return 1;
        }

        size_t inBytes = ane_prefill_swiglu_session_input_bytes(sess);
        size_t outBytes = ane_prefill_swiglu_session_output_bytes(sess);
        _Float16 *X16 = (_Float16 *)malloc(inBytes);
        if (!X16) {
            ane_prefill_swiglu_session_destroy(sess);
            free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "act fp16 alloc failed", @{@"mode": @"ane_prefill_swiglu"});
            return 1;
        }
        for (int i = 0; i < dim * seq; i++) {
            X16[i] = (_Float16)X[i];
        }
        if (!ane_prefill_swiglu_session_write_acts_fp16(sess, X16, inBytes)) {
            free(X16);
            ane_prefill_swiglu_session_destroy(sess);
            free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "write acts failed", @{@"mode": @"ane_prefill_swiglu"});
            return 1;
        }
        free(X16);

        for (int i = 0; i < warmup; i++) {
            if (!ane_prefill_swiglu_session_eval(sess)) {
                ane_prefill_swiglu_session_destroy(sess);
                free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
                emitJSON(NO, "warmup eval failed", @{@"mode": @"ane_prefill_swiglu"});
                return 1;
            }
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            if (!ane_prefill_swiglu_session_eval(sess)) {
                ane_prefill_swiglu_session_destroy(sess);
                free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
                emitJSON(NO, "eval failed", @{@"mode": @"ane_prefill_swiglu"});
                return 1;
            }
        }
        double evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;

        _Float16 *Y16 = (_Float16 *)malloc(outBytes);
        float *Yane = (float *)malloc((size_t)dim * (size_t)seq * sizeof(float));
        if (!Y16 || !Yane || !ane_prefill_swiglu_session_read_out_fp16(sess, Y16, outBytes)) {
            free(Y16); free(Yane);
            ane_prefill_swiglu_session_destroy(sess);
            free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "read output failed", @{@"mode": @"ane_prefill_swiglu"});
            return 1;
        }
        BOOL finite = YES;
        float aneMax = 0;
        for (int i = 0; i < dim * seq; i++) {
            float v = (float)Y16[i];
            if (!isfinite(v)) {
                finite = NO;
                break;
            }
            float a = fabsf(v);
            if (a > aneMax) aneMax = a;
            Yane[i] = v;
        }
        free(Y16);

        if (finite && aneMax == 0) {
            ane_prefill_swiglu_session_destroy(sess);
            free(Wg); free(Wu); free(Wd); free(X); free(Ycpu); free(Yane);
            emitJSON(NO, "ANE output all zeros", @{
                @"mode": @"ane_prefill_swiglu",
                @"dim": @(dim), @"hidden": @(hidden), @"seq": @(seq),
            });
            return 1;
        }

        double goldenCos = 0;
        float maxErr = 0;
        if (parity && finite) {
            goldenCos = cosineSim(Yane, Ycpu, (size_t)dim * (size_t)seq);
            maxErr = maxAbsErr(Yane, Ycpu, (size_t)dim * (size_t)seq);
        }

        double compileMs = ane_prefill_swiglu_session_compile_ms(sess);
        int evalCount = ane_prefill_swiglu_session_eval_count(sess);
        uint32_t sid = ane_prefill_swiglu_session_input_surface_id(sess);
        ane_prefill_swiglu_session_destroy(sess);
        free(Wg); free(Wu); free(Wd); free(X); free(Ycpu); free(Yane);

        if (!finite) {
            emitJSON(NO, "non-finite output", @{@"mode": @"ane_prefill_swiglu"});
            return 1;
        }
        if (parity && goldenCos < 0.999) {
            emitJSON(NO, "golden cosine below 0.999", @{
                @"mode": @"ane_prefill_swiglu",
                @"dim": @(dim), @"hidden": @(hidden), @"seq": @(seq),
                @"golden_cosine": @(goldenCos),
                @"max_abs_err": @(maxErr),
            });
            return 1;
        }

        // 3 matmuls: 2×(dim×hidden×seq) + hidden×dim×seq
        double gflop = (4.0 * (double)dim * (double)hidden * (double)seq
                        + 2.0 * (double)hidden * (double)dim * (double)seq) / 1e9;
        double tflops = evalMs > 0 ? gflop / (evalMs / 1000.0) : 0;
        NSMutableDictionary *fields = [@{
            @"mode": @"ane_prefill_swiglu",
            @"variant": @"swiglu-fused",
            @"dim": @(dim),
            @"hidden": @(hidden),
            @"seq": @(seq),
            @"surface_id": @(sid),
            @"surface_bytes": @(inBytes),
            @"compile_ms": @(compileMs),
            @"eval_ms": @(evalMs),
            @"gflop": @(gflop),
            @"tflops": @(tflops),
            @"eval_count": @(evalCount),
            @"kernel_reused": @YES,
            @"ane_max_abs": @(aneMax),
            @"source": @"zerollama/ane_prefill_swiglu_session",
            @"note": @"fused gate+up+silu*+down; multi BLOBFILE; CPU golden parity",
        } mutableCopy];
        if (parity) {
            fields[@"golden_cosine"] = @(goldenCos);
            fields[@"max_abs_err"] = @(maxErr);
            fields[@"parity"] = @YES;
        }
        emitJSON(YES, NULL, fields);
        return 0;
    }
}
