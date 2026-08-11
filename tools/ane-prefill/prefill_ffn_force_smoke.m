// Lab smoke: policy force + registered ANE host replace returns true and matches CPU.
// JSON: ok, mode=ane_ffn_force, cosine, force_replaced_count, …

#import <Foundation/Foundation.h>
#include "ane_ffn_force_replace.h"
#include "ane_ffn_policy.h"
#include "ane_ffn_force_pack.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static void clear_ffn_env(void) {
    unsetenv("ZEROLLAMA_ANE_FFN");
    unsetenv("ZEROLLAMA_ANE_FFN_MODE");
    unsetenv("ZEROLLAMA_ANE_FFN_IC");
    unsetenv("ZEROLLAMA_ANE_FFN_OC");
    unsetenv("ZEROLLAMA_ANE_FFN_SEQ_MAX");
    unsetenv("ZEROLLAMA_ANE_FFN_LAB_PORT");
    unsetenv("ZEROLLAMA_ANE_FFN_PORT");
    unsetenv("ZEROLLAMA_ANE_FFN_TELEMETRY");
    unsetenv("ZEROLLAMA_ANE_FFN_NAME");
    unsetenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE");
    unsetenv("OLLAMA_HOST");
}

int main(int argc, const char *argv[]) {
    int ic = 512;
    int oc = 256;
    int seq = 64;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ic") == 0 && i + 1 < argc) {
            ic = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--oc") == 0 && i + 1 < argc) {
            oc = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            seq = atoi(argv[++i]);
        }
    }

    @autoreleasepool {
        clear_ffn_env();

        // Pack roundtrip: ggml acts [seq][ic] ↔ channel [ic][seq]
        {
            const int pic = 4, pseq = 3, poc = 2;
            float ggml_x[12] = {
                /*t0*/ 1, 2, 3, 4,
                /*t1*/ 5, 6, 7, 8,
                /*t2*/ 9, 10, 11, 12,
            };
            float chan[12];
            float back[12];
            if (!ane_ffn_pack_acts_ggml_to_channel(ggml_x, 0, pic, pseq, chan)) {
                emit(NO, "pack acts failed", @{@"mode": @"ane_ffn_force"});
                return 1;
            }
            // channel: X[i*seq+t] — i=0 → 1,5,9
            if (fabsf(chan[0] - 1.f) > 1e-6f || fabsf(chan[1] - 5.f) > 1e-6f || fabsf(chan[2] - 9.f) > 1e-6f) {
                emit(NO, "pack acts layout wrong", @{@"mode": @"ane_ffn_force"});
                return 1;
            }
            float ychan[6] = { /*o0*/ 1, 2, 3, /*o1*/ 4, 5, 6 };
            if (!ane_ffn_unpack_dst_channel_to_ggml(ychan, 0, poc, pseq, back)) {
                emit(NO, "unpack dst failed", @{@"mode": @"ane_ffn_force"});
                return 1;
            }
            // ggml Y[t*oc+o]: t0 → 1,4
            if (fabsf(back[0] - 1.f) > 1e-6f || fabsf(back[1] - 4.f) > 1e-6f) {
                emit(NO, "unpack dst layout wrong", @{@"mode": @"ane_ffn_force"});
                return 1;
            }
        }

        setenv("ZEROLLAMA_ANE_FFN", "1", 1);
        setenv("ZEROLLAMA_ANE_FFN_MODE", "force", 1);
        setenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE", "1", 1);
        setenv("ZEROLLAMA_ANE_FFN_IC", [[NSString stringWithFormat:@"%d", ic] UTF8String], 1);
        setenv("ZEROLLAMA_ANE_FFN_OC", [[NSString stringWithFormat:@"%d", oc] UTF8String], 1);
        setenv("ZEROLLAMA_ANE_FFN_SEQ_MAX", [[NSString stringWithFormat:@"%d", seq] UTF8String], 1);
        setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435", 1);
        setenv("OLLAMA_HOST", "127.0.0.1:11435", 1);
        setenv("ZEROLLAMA_ANE_FFN_TELEMETRY", "1", 1);

        // Without replace: dims-only and host both defer / fail-closed.
        if (ane_ffn_force_try_mul_mat(ic, oc, seq, NULL)) {
            emit(NO, "dims-only force returned true without replace", @{
                @"mode": @"ane_ffn_force",
            });
            return 1;
        }
        float *dummyY = (float *)calloc((size_t)oc * (size_t)seq, sizeof(float));
        float *dummyW = (float *)calloc((size_t)oc * (size_t)ic, sizeof(float));
        float *dummyX = (float *)calloc((size_t)ic * (size_t)seq, sizeof(float));
        if (!dummyY || !dummyW || !dummyX) {
            free(dummyY); free(dummyW); free(dummyX);
            emit(NO, "alloc failed", @{@"mode": @"ane_ffn_force"});
            return 1;
        }
        if (ane_ffn_force_try_mul_mat_host(ic, oc, seq, dummyW, dummyX, dummyY)) {
            free(dummyY); free(dummyW); free(dummyX);
            emit(NO, "host force returned true without registered replace", @{
                @"mode": @"ane_ffn_force",
            });
            return 1;
        }

        // Refuse production even with replace registered.
        ane_ffn_force_set_host_replace(ane_ffn_force_replace_mul_mat);
        setenv("OLLAMA_HOST", "127.0.0.1:11434", 1);
        if (ane_ffn_force_try_mul_mat_host(ic, oc, seq, dummyW, dummyX, dummyY)) {
            free(dummyY); free(dummyW); free(dummyX);
            emit(NO, "force allowed on production port 11434", @{@"mode": @"ane_ffn_force"});
            return 1;
        }
        setenv("OLLAMA_HOST", "127.0.0.1:11435", 1);
        free(dummyY); free(dummyW); free(dummyX);

        float *W = (float *)malloc((size_t)oc * (size_t)ic * sizeof(float));
        float *X = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
        float *Yane = (float *)malloc((size_t)oc * (size_t)seq * sizeof(float));
        float *Ycpu = (float *)malloc((size_t)oc * (size_t)seq * sizeof(float));
        if (!W || !X || !Yane || !Ycpu) {
            free(W); free(X); free(Yane); free(Ycpu);
            emit(NO, "alloc failed", @{@"mode": @"ane_ffn_force"});
            return 1;
        }
        fillSynth(W, oc * ic, 0.1f, 17, 3);
        fillSynth(X, ic * seq, 0.1f, 13, 7);
        cpuMatmul(W, X, Ycpu, ic, oc, seq);

        uint64_t before = ane_ffn_force_replaced_count();
        if (!ane_ffn_force_try_mul_mat_host(ic, oc, seq, W, X, Yane)) {
            free(W); free(X); free(Yane); free(Ycpu);
            emit(NO, "host force replace returned false", @{
                @"mode": @"ane_ffn_force",
                @"ic": @(ic), @"oc": @(oc), @"seq": @(seq),
            });
            return 1;
        }
        if (ane_ffn_force_replaced_count() != before + 1) {
            free(W); free(X); free(Yane); free(Ycpu);
            emit(NO, "force_replaced_count did not increment", @{@"mode": @"ane_ffn_force"});
            return 1;
        }

        // Second call should hit session cache (same weight pointer).
        if (!ane_ffn_force_try_mul_mat_host(ic, oc, seq, W, X, Yane)) {
            free(W); free(X); free(Yane); free(Ycpu);
            emit(NO, "cached host force replace failed", @{@"mode": @"ane_ffn_force"});
            return 1;
        }

        double cos = cosineSim(Ycpu, Yane, (size_t)oc * (size_t)seq);
        free(W); free(X); free(Yane); free(Ycpu);

        // Autoload via dylib (same as Metal lab path).
        {
            clear_ffn_env();
            setenv("ZEROLLAMA_ANE_FFN", "1", 1);
            setenv("ZEROLLAMA_ANE_FFN_MODE", "force", 1);
            setenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE", "1", 1);
            setenv("ZEROLLAMA_ANE_FFN_IC", [[NSString stringWithFormat:@"%d", ic] UTF8String], 1);
            setenv("ZEROLLAMA_ANE_FFN_OC", [[NSString stringWithFormat:@"%d", oc] UTF8String], 1);
            setenv("ZEROLLAMA_ANE_FFN_SEQ_MAX", [[NSString stringWithFormat:@"%d", seq] UTF8String], 1);
            setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435", 1);
            setenv("OLLAMA_HOST", "127.0.0.1:11435", 1);
            NSString *dylib = [[[NSString stringWithUTF8String:argv[0]] stringByDeletingLastPathComponent]
                stringByAppendingPathComponent:@"libane_ffn_force.dylib"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:dylib]) {
                setenv("ZEROLLAMA_ANE_FFN_REPLACE_DYLIB", [dylib UTF8String], 1);
                ane_ffn_force_set_host_replace(NULL);
                float *W2 = (float *)malloc((size_t)oc * (size_t)ic * sizeof(float));
                float *X2 = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
                float *Y2 = (float *)malloc((size_t)oc * (size_t)seq * sizeof(float));
                if (W2 && X2 && Y2) {
                    fillSynth(W2, oc * ic, 0.1f, 17, 3);
                    fillSynth(X2, ic * seq, 0.1f, 13, 7);
                    if (!ane_ffn_force_try_mul_mat_host(ic, oc, seq, W2, X2, Y2)) {
                        free(W2); free(X2); free(Y2);
                        emit(NO, "dlopen host replace failed", @{
                            @"mode": @"ane_ffn_force",
                            @"dylib": dylib,
                        });
                        return 1;
                    }
                }
                free(W2); free(X2); free(Y2);
            }
        }

        clear_ffn_env();

        if (cos < 0.999) {
            emit(NO, "cosine below threshold", @{
                @"mode": @"ane_ffn_force",
                @"cosine": @(cos),
                @"ic": @(ic), @"oc": @(oc), @"seq": @(seq),
            });
            return 1;
        }
        emit(YES, NULL, @{
            @"mode": @"ane_ffn_force",
            @"ic": @(ic), @"oc": @(oc), @"seq": @(seq),
            @"cosine": @(cos),
            @"force_replaced_count": @(ane_ffn_force_replaced_count()),
            @"force_deferred_count": @(ane_ffn_force_deferred_count()),
            @"note": @"pack+host replace+dlopen; dims-only Metal deferred; FORCE_HOST for tensor pack",
        });
        return 0;
    }
}
