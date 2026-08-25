// Lab smoke: SwiGLU name helpers + fused host replace (ANE).
// Default: fp16-blob force replace.
// Opt-in best path: --int8 --w8a8 [--w8a8-x|--int8-in] (sets ZEROLLAMA_ANE_FFN_*).
#import <Foundation/Foundation.h>
#include "ane_ffn_force_replace.h"
#include "ane_ffn_force_pack.h"
#include "ane_ffn_layout_metal.h"
#include "ane_ffn_policy.h"
#include "ane_ffn_swiglu_fuse.h"

#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

static mach_timebase_info_data_t g_tb;

static double ticksToMs(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static void emit(BOOL ok, const char *err, NSDictionary *fields) {
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

static void fillSynth(float *dst, int n, float scale, int seedA, int seedB) {
    for (int i = 0; i < n; i++) {
        dst[i] = scale * (float)((i * seedA + seedB) % 97) / 97.0f;
        if (dst[i] < 0.01f) {
            dst[i] = 0.01f;
        }
    }
}

static float silu(float x) {
    return x / (1.f + expf(-x));
}

static void cpuMatmul(const float *W, const float *X, float *Y, int ic, int oc, int seq) {
    for (int o = 0; o < oc; o++) {
        for (int s = 0; s < seq; s++) {
            float acc = 0;
            for (int i = 0; i < ic; i++) {
                acc += W[o * ic + i] * X[i * seq + s];
            }
            Y[o * seq + s] = acc;
        }
    }
}

static void applyActInt8RoundTrip(float *hid, int n, float scale) {
    if (!(scale > 0)) return;
    for (int i = 0; i < n; i++) {
        float v = hid[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        int8_t q = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
        hid[i] = (float)q * scale;
    }
}

static float quantizeDequantInPlace(float *w, int n) {
    float maxAbs = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(w[i]);
        if (a > maxAbs) maxAbs = a;
    }
    float scale = maxAbs / 127.0f;
    if (!(scale > 0)) scale = 1.0f;
    for (int i = 0; i < n; i++) {
        float v = w[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        int8_t q = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
        w[i] = (float)q * scale;
    }
    return scale;
}

static float cpuHidMaxAbs(const float *Wg, const float *Wu, const float *X,
                          int ic, int hidden, int seq) {
    float *gate = (float *)malloc((size_t)hidden * (size_t)seq * sizeof(float));
    float *up = (float *)malloc((size_t)hidden * (size_t)seq * sizeof(float));
    if (!gate || !up) {
        free(gate); free(up);
        return 1.0f;
    }
    cpuMatmul(Wg, X, gate, ic, hidden, seq);
    cpuMatmul(Wu, X, up, ic, hidden, seq);
    float mx = 0;
    for (int i = 0; i < hidden * seq; i++) {
        float a = fabsf(silu(gate[i]) * up[i]);
        if (a > mx) mx = a;
    }
    free(gate); free(up);
    return mx > 0 ? mx : 1.0f;
}

// y = (silu(x@Wg)*(x@Wu))@Wd; optional hid int8 round-trip for W8A8 golden.
static void cpuSwiGLU(const float *Wg, const float *Wu, const float *Wd,
                      const float *X, float *Y, int ic, int hidden, int seq,
                      float act_scale) {
    float *gate = (float *)malloc((size_t)hidden * (size_t)seq * sizeof(float));
    float *up = (float *)malloc((size_t)hidden * (size_t)seq * sizeof(float));
    float *hid = (float *)malloc((size_t)hidden * (size_t)seq * sizeof(float));
    if (!gate || !up || !hid) {
        free(gate); free(up); free(hid);
        memset(Y, 0, (size_t)ic * (size_t)seq * sizeof(float));
        return;
    }
    cpuMatmul(Wg, X, gate, ic, hidden, seq);
    cpuMatmul(Wu, X, up, ic, hidden, seq);
    for (int i = 0; i < hidden * seq; i++) {
        hid[i] = silu(gate[i]) * up[i];
    }
    if (act_scale > 0) {
        applyActInt8RoundTrip(hid, hidden * seq, act_scale);
    }
    cpuMatmul(Wd, hid, Y, hidden, ic, seq);
    free(gate); free(up); free(hid);
}

static double cosineSim(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na <= 0 || nb <= 0) {
        return 0;
    }
    return dot / (sqrt(na) * sqrt(nb));
}

static void clear_env(void) {
    unsetenv("ZEROLLAMA_ANE_FFN");
    unsetenv("ZEROLLAMA_ANE_FFN_MODE");
    unsetenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE");
    unsetenv("ZEROLLAMA_ANE_FFN_SWIGLU");
    unsetenv("ZEROLLAMA_ANE_FFN_NAME");
    unsetenv("ZEROLLAMA_ANE_FFN_IC");
    unsetenv("ZEROLLAMA_ANE_FFN_OC");
    unsetenv("ZEROLLAMA_ANE_FFN_SEQ_MAX");
    unsetenv("ZEROLLAMA_ANE_FFN_LAB_PORT");
    unsetenv("OLLAMA_HOST");
    unsetenv("ZEROLLAMA_ANE_FFN_TELEMETRY");
    unsetenv("ZEROLLAMA_ANE_FFN_INT8");
    unsetenv("ZEROLLAMA_ANE_FFN_W8A8");
    unsetenv("ZEROLLAMA_ANE_FFN_W8A8_X");
    unsetenv("ZEROLLAMA_ANE_FFN_INT8_IN");
}

int main(int argc, const char *argv[]) {
    int ic = 256;
    int hidden = 128;
    int seq = 64;
    int iters = 20;
    int warmup = 2;
    BOOL wantInt8 = NO;
    BOOL w8a8 = NO;
    BOOL w8a8x = NO;
    BOOL int8In = NO;
    BOOL prepack = NO;
    BOOL ggmlPack = NO;
    BOOL metalLayout = NO;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ic") == 0 && i + 1 < argc) ic = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--int8") == 0) wantInt8 = YES;
        else if (strcmp(argv[i], "--w8a8") == 0) { w8a8 = YES; wantInt8 = YES; }
        else if (strcmp(argv[i], "--w8a8-x") == 0) { w8a8 = YES; w8a8x = YES; wantInt8 = YES; }
        else if (strcmp(argv[i], "--int8-in") == 0) {
            w8a8 = YES; w8a8x = YES; int8In = YES; wantInt8 = YES;
        } else if (strcmp(argv[i], "--prepack") == 0) {
            prepack = YES;
        } else if (strcmp(argv[i], "--ggml-pack") == 0) {
            ggmlPack = YES;
            prepack = YES;
        } else if (strcmp(argv[i], "--metal-layout") == 0) {
            metalLayout = YES;
            ggmlPack = YES;
            prepack = YES;
        }
    }

    @autoreleasepool {
        mach_timebase_info(&g_tb);

        if (!ane_ffn_name_is_ffn_up("blk.0.ffn_up_shexp.weight") ||
            ane_ffn_name_is_ffn_up("blk.0.ffn_up_exps.weight") ||
            !ane_ffn_name_is_ffn_gate("blk.0.ffn_gate.weight") ||
            !ane_ffn_name_is_ffn_down("blk.0.ffn_down_shexp.weight") ||
            ane_ffn_name_is_ffn_down("blk.0.ffn_down_exps.weight")) {
            emit(NO, "name helper failed", @{@"mode": @"ane_ffn_swiglu"});
            return 1;
        }

        clear_env();
        setenv("ZEROLLAMA_ANE_FFN", "1", 1);
        setenv("ZEROLLAMA_ANE_FFN_MODE", "force", 1);
        setenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE", "1", 1);
        setenv("ZEROLLAMA_ANE_FFN_SWIGLU", "1", 1);
        setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435", 1);
        setenv("OLLAMA_HOST", "127.0.0.1:11435", 1);
        setenv("ZEROLLAMA_ANE_FFN_TELEMETRY", "1", 1);
        if (wantInt8) setenv("ZEROLLAMA_ANE_FFN_INT8", "1", 1);
        if (w8a8) setenv("ZEROLLAMA_ANE_FFN_W8A8", "1", 1);
        if (w8a8x) setenv("ZEROLLAMA_ANE_FFN_W8A8_X", "1", 1);
        if (int8In) setenv("ZEROLLAMA_ANE_FFN_INT8_IN", "1", 1);
        ane_ffn_force_set_swiglu_replace(ane_ffn_force_replace_swiglu);

        float *WgF = (float *)malloc((size_t)hidden * ic * sizeof(float));
        float *WuF = (float *)malloc((size_t)hidden * ic * sizeof(float));
        float *WdF = (float *)malloc((size_t)ic * hidden * sizeof(float));
        float *X = (float *)malloc((size_t)ic * seq * sizeof(float));
        float *Yane = (float *)malloc((size_t)ic * seq * sizeof(float));
        float *Ycpu = (float *)malloc((size_t)ic * seq * sizeof(float));
        if (!WgF || !WuF || !WdF || !X || !Yane || !Ycpu) {
            emit(NO, "alloc failed", @{@"mode": @"ane_ffn_swiglu"});
            return 1;
        }
        fillSynth(WgF, hidden * ic, 0.08f, 17, 3);
        fillSynth(WuF, hidden * ic, 0.08f, 13, 7);
        fillSynth(WdF, ic * hidden, 0.08f, 19, 11);
        fillSynth(X, ic * seq, 0.1f, 11, 5);

        float actScale = 0;
        float *Xgolden = X;
        float *Xrt = NULL;
        if (wantInt8) {
            float *Gdq = (float *)malloc((size_t)hidden * ic * sizeof(float));
            float *Udq = (float *)malloc((size_t)hidden * ic * sizeof(float));
            float *Ddq = (float *)malloc((size_t)ic * hidden * sizeof(float));
            if (!Gdq || !Udq || !Ddq) {
                free(Gdq); free(Udq); free(Ddq);
                free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                emit(NO, "dequant alloc failed", @{@"mode": @"ane_ffn_swiglu"});
                return 1;
            }
            memcpy(Gdq, WgF, (size_t)hidden * ic * sizeof(float));
            memcpy(Udq, WuF, (size_t)hidden * ic * sizeof(float));
            memcpy(Ddq, WdF, (size_t)ic * hidden * sizeof(float));
            (void)quantizeDequantInPlace(Gdq, hidden * ic);
            (void)quantizeDequantInPlace(Udq, hidden * ic);
            (void)quantizeDequantInPlace(Ddq, ic * hidden);
            if (w8a8x || int8In) {
                float xmax = 0;
                for (int i = 0; i < ic * seq; i++) {
                    float a = fabsf(X[i]);
                    if (a > xmax) xmax = a;
                }
                float xScale = xmax / 127.0f;
                if (!(xScale > 0)) xScale = 1.0f;
                Xrt = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
                if (!Xrt) {
                    free(Gdq); free(Udq); free(Ddq);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "x roundtrip alloc failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
                memcpy(Xrt, X, (size_t)ic * (size_t)seq * sizeof(float));
                applyActInt8RoundTrip(Xrt, ic * seq, xScale);
                Xgolden = Xrt;
            }
            if (w8a8) {
                float hidMax = cpuHidMaxAbs(Gdq, Udq, Xgolden, ic, hidden, seq);
                actScale = hidMax / 127.0f;
                if (!(actScale > 0)) actScale = 1.0f;
            }
            cpuSwiGLU(Gdq, Udq, Ddq, Xgolden, Ycpu, ic, hidden, seq, actScale);
            free(Gdq); free(Udq); free(Ddq);
        } else {
            cpuSwiGLU(WgF, WuF, WdF, X, Ycpu, ic, hidden, seq, 0);
        }

        if (!ane_ffn_force_try_swiglu_host(ic, hidden, seq, WgF, WuF, WdF, X, Yane)) {
            free(Xrt);
            free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
            emit(NO, "swiglu host replace failed", @{
                @"mode": @"ane_ffn_swiglu",
                @"ic": @(ic), @"hidden": @(hidden), @"seq": @(seq),
            });
            return 1;
        }
        double cos = cosineSim(Ycpu, Yane, (size_t)ic * (size_t)seq);

        double hostMs = 0;
        double prepackMs = 0;
        double reevalMs = 0;
        double aneOnlyMs = 0;
        double ggmlI8Ms = 0;
        double ggmlF16Ms = 0;
        double metalMs = 0;
        float xScaleUsed = 0;

        if (prepack) {
            if (!int8In) {
                free(Xrt);
                free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                emit(NO, "--prepack requires --int8-in", @{@"mode": @"ane_ffn_swiglu"});
                return 1;
            }
            xScaleUsed = ane_ffn_force_swiglu_x_scale();
            if (!(xScaleUsed > 0)) {
                free(Xrt);
                free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                emit(NO, "x_scale unset after create", @{@"mode": @"ane_ffn_swiglu"});
                return 1;
            }
            size_t nAct = (size_t)ic * (size_t)seq;
            int8_t *Qin = (int8_t *)malloc(nAct);
            if (!Qin) {
                free(Xrt);
                free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                emit(NO, "qin alloc failed", @{@"mode": @"ane_ffn_swiglu"});
                return 1;
            }
            for (size_t i = 0; i < nAct; i++) {
                float v = X[i] / xScaleUsed;
                if (v > 127.0f) v = 127.0f;
                if (v < -128.0f) v = -128.0f;
                Qin[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
            }

            // Baseline: host path still quantizes each call (buffer-reuse only).
            for (int i = 0; i < warmup; i++) {
                (void)ane_ffn_force_try_swiglu_host(ic, hidden, seq, WgF, WuF, WdF, X, Yane);
            }
            uint64_t th = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                if (!ane_ffn_force_try_swiglu_host(ic, hidden, seq, WgF, WuF, WdF, X, Yane)) {
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "host timed replace failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            hostMs = ticksToMs(mach_absolute_time() - th) / (double)iters;

            for (int i = 0; i < warmup; i++) {
                if (!ane_ffn_force_replace_swiglu_int8(
                        ic, hidden, seq, WgF, WuF, WdF, Qin, Yane)) {
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "prepack warmup failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            uint64_t tp = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                if (!ane_ffn_force_replace_swiglu_int8(
                        ic, hidden, seq, WgF, WuF, WdF, Qin, Yane)) {
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "prepack timed replace failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            prepackMs = ticksToMs(mach_absolute_time() - tp) / (double)iters;
            cos = cosineSim(Ycpu, Yane, nAct);

            if (!ane_ffn_force_swiglu_write_int8(Qin, seq)) {
                free(Qin); free(Xrt);
                free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                emit(NO, "write_int8 failed", @{@"mode": @"ane_ffn_swiglu"});
                return 1;
            }
            for (int i = 0; i < warmup; i++) {
                if (!ane_ffn_force_swiglu_reeval_f32(Yane, seq)) {
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "reeval warmup failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            uint64_t te = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                if (!ane_ffn_force_swiglu_reeval_f32(Yane, seq)) {
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "reeval timed failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            reevalMs = ticksToMs(mach_absolute_time() - te) / (double)iters;

            for (int i = 0; i < warmup; i++) {
                if (!ane_ffn_force_swiglu_eval_only()) {
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "eval_only warmup failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            uint64_t ta = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                if (!ane_ffn_force_swiglu_eval_only()) {
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "eval_only timed failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            aneOnlyMs = ticksToMs(mach_absolute_time() - ta) / (double)iters;
            // One read for abs gate after eval-only loop.
            if (!ane_ffn_force_swiglu_reeval_f32(Yane, seq)) {
                free(Qin); free(Xrt);
                free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                emit(NO, "post eval_only read failed", @{@"mode": @"ane_ffn_swiglu"});
                return 1;
            }

            if (ggmlPack) {
                // Mimic tensors path: ggml [seq][ic] → pack → replace → unpack (fp16 dst).
                float *ggmlX = (float *)malloc(nAct * sizeof(float));
                _Float16 *Y16 = (_Float16 *)malloc(nAct * sizeof(_Float16));
                _Float16 *Ygg16 = (_Float16 *)malloc(nAct * sizeof(_Float16));
                int8_t *Xi8 = (int8_t *)malloc(nAct);
                _Float16 *X16 = (_Float16 *)malloc(nAct * sizeof(_Float16));
                if (!ggmlX || !Y16 || !Ygg16 || !Xi8 || !X16) {
                    free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                    free(Qin); free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "ggml-pack alloc failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
                for (int i = 0; i < ic; i++) {
                    for (int t = 0; t < seq; t++) {
                        ggmlX[t * ic + i] = X[i * seq + t];
                    }
                }

                for (int i = 0; i < warmup; i++) {
                    if (!ane_ffn_pack_acts_ggml_to_channel_i8(ggmlX, 0, ic, seq, xScaleUsed, Xi8) ||
                        !ane_ffn_force_replace_swiglu_int8_fp16(
                            ic, hidden, seq, WgF, WuF, WdF, Xi8, Y16) ||
                        !ane_ffn_unpack_dst_channel_f16_to_ggml(Y16, 1, ic, seq, Ygg16)) {
                        free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                        free(Qin); free(Xrt);
                        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                        emit(NO, "ggml i8 pack warmup failed", @{@"mode": @"ane_ffn_swiglu"});
                        return 1;
                    }
                }
                uint64_t tg = mach_absolute_time();
                for (int i = 0; i < iters; i++) {
                    if (!ane_ffn_pack_acts_ggml_to_channel_i8(ggmlX, 0, ic, seq, xScaleUsed, Xi8) ||
                        !ane_ffn_force_replace_swiglu_int8_fp16(
                            ic, hidden, seq, WgF, WuF, WdF, Xi8, Y16) ||
                        !ane_ffn_unpack_dst_channel_f16_to_ggml(Y16, 1, ic, seq, Ygg16)) {
                        free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                        free(Qin); free(Xrt);
                        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                        emit(NO, "ggml i8 pack timed failed", @{@"mode": @"ane_ffn_swiglu"});
                        return 1;
                    }
                }
                ggmlI8Ms = ticksToMs(mach_absolute_time() - tg) / (double)iters;
                // Cosine on channel Y16 vs golden (skip ggml layout for parity check).
                for (size_t i = 0; i < nAct; i++) {
                    Yane[i] = (float)Y16[i];
                }
                cos = cosineSim(Ycpu, Yane, nAct);

                // Metal layout before f16 host path — f16 replace can disturb surface timing.
                if (metalLayout) {
                    if (!ane_ffn_layout_metal_ready()) {
                        free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                        free(Qin); free(Xrt);
                        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                        emit(NO, "metal layout not ready", @{@"mode": @"ane_ffn_swiglu"});
                        return 1;
                    }
                    uint32_t inSid = ane_ffn_force_swiglu_input_surface_id();
                    uint32_t outSid = ane_ffn_force_swiglu_output_surface_id();
                    const int sessSeq = ane_ffn_force_swiglu_session_seq();
                    if (inSid == 0 || outSid == 0 || sessSeq <= 0 || seq > sessSeq) {
                        free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                        free(Qin); free(Xrt);
                        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                        emit(NO, "surface ids / sess seq missing", @{@"mode": @"ane_ffn_swiglu"});
                        return 1;
                    }
                    float *ggmlPad = ggmlX;
                    bool freePad = false;
                    if (sessSeq != seq) {
                        size_t nPad = (size_t)ic * (size_t)sessSeq;
                        ggmlPad = (float *)calloc(nPad, sizeof(float));
                        if (!ggmlPad) {
                            free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                            free(Qin); free(Xrt);
                            free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                            emit(NO, "metal pad alloc failed", @{@"mode": @"ane_ffn_swiglu"});
                            return 1;
                        }
                        for (int t = 0; t < seq; t++) {
                            memcpy(ggmlPad + (size_t)t * (size_t)ic,
                                   ggmlX + (size_t)t * (size_t)ic,
                                   (size_t)ic * sizeof(float));
                        }
                        freePad = true;
                    }
                    void *YggAligned = NULL;
                    size_t ybytes = (size_t)ic * (size_t)sessSeq * sizeof(_Float16);
                    if (posix_memalign(&YggAligned, 16384, ybytes) != 0 || !YggAligned) {
                        if (freePad) free(ggmlPad);
                        free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                        free(Qin); free(Xrt);
                        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                        emit(NO, "aligned ggml dst alloc failed", @{@"mode": @"ane_ffn_swiglu"});
                        return 1;
                    }
                    memset(YggAligned, 0, ybytes);

                    for (int i = 0; i < warmup; i++) {
                        if (!ane_ffn_layout_metal_pack_in_i8_f32(
                                inSid, ggmlPad, ic, sessSeq, xScaleUsed) ||
                            !ane_ffn_force_swiglu_eval_only() ||
                            !ane_ffn_layout_metal_unpack_out_f16(
                                outSid, ic, sessSeq, YggAligned)) {
                            if (freePad) free(ggmlPad);
                            free(YggAligned);
                            free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                            free(Qin); free(Xrt);
                            free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                            emit(NO, "metal layout warmup failed", @{@"mode": @"ane_ffn_swiglu"});
                            return 1;
                        }
                    }
                    uint64_t tm = mach_absolute_time();
                    for (int i = 0; i < iters; i++) {
                        if (!ane_ffn_layout_metal_pack_in_i8_f32(
                                inSid, ggmlPad, ic, sessSeq, xScaleUsed) ||
                            !ane_ffn_force_swiglu_eval_only() ||
                            !ane_ffn_layout_metal_unpack_out_f16(
                                outSid, ic, sessSeq, YggAligned)) {
                            if (freePad) free(ggmlPad);
                            free(YggAligned);
                            free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                            free(Qin); free(Xrt);
                            free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                            emit(NO, "metal layout timed failed", @{@"mode": @"ane_ffn_swiglu"});
                            return 1;
                        }
                    }
                    metalMs = ticksToMs(mach_absolute_time() - tm) / (double)iters;

                    const _Float16 *yg = (const _Float16 *)YggAligned;
                    for (int o = 0; o < ic; o++) {
                        for (int t = 0; t < seq; t++) {
                            Yane[o * seq + t] = (float)yg[t * ic + o];
                        }
                    }
                    cos = cosineSim(Ycpu, Yane, nAct);
                    if (freePad) free(ggmlPad);
                    free(YggAligned);
                }

                for (int i = 0; i < warmup; i++) {
                    if (!ane_ffn_pack_acts_ggml_to_channel_f16(ggmlX, 0, ic, seq, X16) ||
                        !ane_ffn_force_replace_swiglu_fp16(
                            ic, hidden, seq, WgF, WuF, WdF, X16, Y16) ||
                        !ane_ffn_unpack_dst_channel_f16_to_ggml(Y16, 1, ic, seq, Ygg16)) {
                        free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                        free(Qin); free(Xrt);
                        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                        emit(NO, "ggml f16 pack warmup failed", @{@"mode": @"ane_ffn_swiglu"});
                        return 1;
                    }
                }
                uint64_t tf = mach_absolute_time();
                for (int i = 0; i < iters; i++) {
                    if (!ane_ffn_pack_acts_ggml_to_channel_f16(ggmlX, 0, ic, seq, X16) ||
                        !ane_ffn_force_replace_swiglu_fp16(
                            ic, hidden, seq, WgF, WuF, WdF, X16, Y16) ||
                        !ane_ffn_unpack_dst_channel_f16_to_ggml(Y16, 1, ic, seq, Ygg16)) {
                        free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
                        free(Qin); free(Xrt);
                        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                        emit(NO, "ggml f16 pack timed failed", @{@"mode": @"ane_ffn_swiglu"});
                        return 1;
                    }
                }
                ggmlF16Ms = ticksToMs(mach_absolute_time() - tf) / (double)iters;
                free(ggmlX); free(Y16); free(Ygg16); free(Xi8); free(X16);
            }
            free(Qin);
        } else {
            for (int i = 0; i < warmup; i++) {
                if (!ane_ffn_force_try_swiglu_host(ic, hidden, seq, WgF, WuF, WdF, X, Yane)) {
                    free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "warmup replace failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                if (!ane_ffn_force_try_swiglu_host(ic, hidden, seq, WgF, WuF, WdF, X, Yane)) {
                    free(Xrt);
                    free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
                    emit(NO, "timed replace failed", @{@"mode": @"ane_ffn_swiglu"});
                    return 1;
                }
            }
            hostMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;
        }

        free(Xrt);
        free(WgF); free(WuF); free(WdF); free(X); free(Yane); free(Ycpu);
        clear_env();

        NSString *variant = @"fp16-swiglu";
        if (metalLayout) variant = @"int8-swiglu-w8a8-int8in-metal-layout";
        else if (ggmlPack) variant = @"int8-swiglu-w8a8-int8in-ggml-pack";
        else if (prepack) variant = @"int8-swiglu-w8a8-int8in-prepack";
        else if (int8In) variant = @"int8-swiglu-w8a8-int8in";
        else if (w8a8x) variant = @"int8-swiglu-w8a8-x";
        else if (w8a8) variant = @"int8-swiglu-w8a8";
        else if (wantInt8) variant = @"int8-swiglu";

        double cosMin = wantInt8 ? 0.99 : 0.999;
        double reportMs = metalLayout ? metalMs : (ggmlPack ? ggmlI8Ms : (prepack ? prepackMs : hostMs));
        if (cos < cosMin) {
            emit(NO, "cosine below threshold", @{
                @"mode": @"ane_ffn_swiglu",
                @"variant": variant,
                @"cosine": @(cos),
                @"eval_ms": @(reportMs),
            });
            return 1;
        }
        NSMutableDictionary *fields = [@{
            @"mode": @"ane_ffn_swiglu",
            @"variant": variant,
            @"ic": @(ic), @"hidden": @(hidden), @"seq": @(seq),
            @"cosine": @(cos),
            @"eval_ms": @(reportMs),
            @"host_ms": @(hostMs),
            @"n_fuse": @4,
            @"force_replaced_count": @(ane_ffn_force_replaced_count()),
            @"note": @"--metal-layout = Metal pack→eval→Metal unpack (skip host transpose)",
        } mutableCopy];
        if (prepack) {
            fields[@"prepack_ms"] = @(prepackMs);
            fields[@"reeval_ms"] = @(reevalMs);
            fields[@"ane_only_ms"] = @(aneOnlyMs);
            fields[@"x_scale"] = @(xScaleUsed);
        }
        if (ggmlPack) {
            fields[@"ggml_i8_ms"] = @(ggmlI8Ms);
            fields[@"ggml_f16_ms"] = @(ggmlF16Ms);
        }
        if (metalLayout) {
            fields[@"metal_ms"] = @(metalMs);
        }
        emit(YES, NULL, fields);
        return 0;
    }
}
