// In-process ANE FFN-slice smoke — compile-once session + ggml map + steady eval + CPU parity.
// Modes:
//   default              fp16-blob matmul
//   --int8               int8-conv matmul
//   --swiglu             fp16 fused SwiGLU (3 conv)
//   --swiglu --int8      int8-weight SwiGLU
//   --swiglu --fuse-gu   fp16 gate∥up fused + slice
//   --swiglu --int8 --fuse-gu [--w8a8]  int8 fused gate∥up (+ optional hid quant)
//
// JSON: ok, mode, variant, eval_ms, golden_cosine, …

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include "ane_prefill_session.h"
#include "ane_ffn_policy.h"
#include "../ane-metal/ggml_iosurface_map.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

// Channel-major X[ic,seq], W[oc,ic] → Y[oc,seq]
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

static float silu(float x) {
    return x / (1.f + expf(-x));
}

// Apply same global int8 round-trip as MIL quantize/dequantize (scale * round(x/scale)).
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

// y = (silu(x@Wg) * (x@Wu)) @ Wd — channel-major layouts matching session.
// If act_scale > 0, apply int8 round-trip on hid before down (W8A8 golden).
static void cpuSwiGLU(const float *Wg, const float *Wu, const float *Wd,
                      const float *X, float *Y,
                      int ic, int hidden, int seq, float act_scale) {
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
    // Wd is [ic × hidden]; treat as OC=ic, IC=hidden
    cpuMatmul(Wd, hid, Y, hidden, ic, seq);
    free(gate); free(up); free(hid);
}

// Max |silu(gate)*up| for act-scale calibration.
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
        float h = silu(gate[i]) * up[i];
        float a = fabsf(h);
        if (a > mx) mx = a;
    }
    free(gate); free(up);
    return mx > 0 ? mx : 1.0f;
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

static void fillSynth(float *dst, int n, float scale, int seedA, int seedB) {
    for (int i = 0; i < n; i++) {
        dst[i] = scale * (float)((i * seedA + seedB) % 97) / 97.0f;
        if (dst[i] < 0.01f) {
            dst[i] = 0.01f;
        }
    }
}

// Mirror ane_bridge_build_weight_blob_quantized: global symmetric scale, round-to-nearest.
static float quantizeDequantInPlace(float *w, int n) {
    float maxAbs = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(w[i]);
        if (a > maxAbs) maxAbs = a;
    }
    float scale = maxAbs / 127.0f;
    if (scale == 0) scale = 1.0f;
    for (int i = 0; i < n; i++) {
        float v = w[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        int8_t q = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
        w[i] = (float)q * scale;
    }
    return scale;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int ic = 2048;
        int oc = 512; // matmul OC / SwiGLU hidden
        int seq = 512;
        int iters = 40;
        int warmup = 3;
        BOOL parity = YES;
        BOOL swiglu = NO;
        BOOL int8 = NO;
        BOOL fuseGU = NO;
        BOOL w8a8 = NO;
        BOOL w8a8x = NO;
        BOOL int8In = NO;
        BOOL autoTile = NO;
        int sp0 = 0, sp1 = 0;
        float xScaleKept = 0;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) {
                iters = 15;
            } else if (strcmp(argv[i], "--ic") == 0 && i + 1 < argc) {
                ic = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--oc") == 0 && i + 1 < argc) {
                oc = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
                seq = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--no-parity") == 0) {
                parity = NO;
            } else if (strcmp(argv[i], "--swiglu") == 0) {
                swiglu = YES;
            } else if (strcmp(argv[i], "--int8") == 0) {
                int8 = YES;
            } else if (strcmp(argv[i], "--fuse-gu") == 0) {
                fuseGU = YES;
            } else if (strcmp(argv[i], "--w8a8") == 0) {
                w8a8 = YES;
            } else if (strcmp(argv[i], "--w8a8-x") == 0) {
                w8a8 = YES;
                w8a8x = YES;
            } else if (strcmp(argv[i], "--int8-in") == 0) {
                int8In = YES;
                w8a8 = YES;
                w8a8x = YES; // int8 input implies x dequant path
            } else if (strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
                // "HxW" or "auto"
                const char *t = argv[++i];
                if (strcmp(t, "auto") == 0) {
                    autoTile = YES;
                } else {
                    int a = 0, b = 0;
                    if (sscanf(t, "%dx%d", &a, &b) == 2 && a > 0 && b > 0) {
                        sp0 = a; sp1 = b;
                    }
                }
            }
        }

        if (ic <= 0 || oc <= 0 || seq <= 0) {
            emitJSON(NO, "ic/oc/seq must be positive", @{@"mode": @"ane_prefill_ffn_slice"});
            return 1;
        }
        if ((fuseGU || w8a8) && !swiglu) {
            emitJSON(NO, "--fuse-gu/--w8a8 require --swiglu", @{@"mode": @"ane_prefill_ffn_slice"});
            return 1;
        }
        if (w8a8 && !int8) {
            emitJSON(NO, "--w8a8/--int8-in require --int8 (weight path)", @{@"mode": @"ane_prefill_ffn_slice"});
            return 1;
        }
        if (int8In && fuseGU) {
            emitJSON(NO, "--int8-in not supported with --fuse-gu yet", @{@"mode": @"ane_prefill_ffn_slice"});
            return 1;
        }
        if (autoTile || (w8a8x && sp0 == 0)) {
            ane_prefill_session_pick_tile(seq, &sp0, &sp1);
        }
        if (sp0 == 0) {
            sp0 = 1;
            sp1 = seq;
        }
        if (sp0 * sp1 != seq) {
            emitJSON(NO, "--tile HxW must satisfy H*W==seq", @{
                @"mode": @"ane_prefill_ffn_slice", @"seq": @(seq), @"sp0": @(sp0), @"sp1": @(sp1),
            });
            return 1;
        }

        const char *mode = swiglu
            ? (int8
                ? (fuseGU
                    ? (w8a8 ? "ane_prefill_swiglu_int8_fused_w8a8" : "ane_prefill_swiglu_int8_fused")
                    : (w8a8 ? "ane_prefill_swiglu_int8_w8a8" : "ane_prefill_swiglu_int8"))
                : (fuseGU ? "ane_prefill_swiglu_fused_gu" : "ane_prefill_swiglu"))
            : (int8 ? "ane_prefill_int8" : "ane_prefill_ffn_slice");
        int hidden = oc;
        int outCh = swiglu ? ic : oc;
        float int8Scale = 0;

        float *X = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
        float *Ycpu = (float *)malloc((size_t)outCh * (size_t)seq * sizeof(float));
        if (!X || !Ycpu) {
            free(X); free(Ycpu);
            emitJSON(NO, "host buffer alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
            return 1;
        }
        fillSynth(X, ic * seq, 0.1f, 13, 7);

        ANEPrefillSession *sess = NULL;
        float *W = NULL;
        float *Wg = NULL;
        float *Wu = NULL;
        float *Wd = NULL;

        if (swiglu) {
            size_t nGate = (size_t)hidden * (size_t)ic;
            size_t nDown = (size_t)ic * (size_t)hidden;
            Wg = (float *)malloc(nGate * sizeof(float));
            Wu = (float *)malloc(nGate * sizeof(float));
            Wd = (float *)malloc(nDown * sizeof(float));
            if (!Wg || !Wu || !Wd) {
                free(X); free(Ycpu); free(Wg); free(Wu); free(Wd);
                emitJSON(NO, "weight alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                return 1;
            }
            fillSynth(Wg, (int)nGate, 0.08f, 17, 3);
            fillSynth(Wu, (int)nGate, 0.08f, 13, 7);
            fillSynth(Wd, (int)nDown, 0.08f, 19, 11);
            if (int8) {
                // Golden = CPU on dequantized weights (match BLOBFILE quant).
                float *Gdq = (float *)malloc(nGate * sizeof(float));
                float *Udq = (float *)malloc(nGate * sizeof(float));
                float *Ddq = (float *)malloc(nDown * sizeof(float));
                if (!Gdq || !Udq || !Ddq) {
                    free(Gdq); free(Udq); free(Ddq);
                    free(X); free(Ycpu); free(Wg); free(Wu); free(Wd);
                    emitJSON(NO, "dequant weight alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                    return 1;
                }
                memcpy(Ddq, Wd, nDown * sizeof(float));
                (void)quantizeDequantInPlace(Ddq, (int)nDown);
                if (fuseGU) {
                    // One global scale over concat(Wg|Wu) — matches create_swiglu_int8_fused.
                    float *GUdq = (float *)malloc(2 * nGate * sizeof(float));
                    if (!GUdq) {
                        free(Gdq); free(Udq); free(Ddq);
                        free(X); free(Ycpu); free(Wg); free(Wu); free(Wd);
                        emitJSON(NO, "fused dequant alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                        return 1;
                    }
                    memcpy(GUdq, Wg, nGate * sizeof(float));
                    memcpy(GUdq + nGate, Wu, nGate * sizeof(float));
                    int8Scale = quantizeDequantInPlace(GUdq, (int)(2 * nGate));
                    memcpy(Gdq, GUdq, nGate * sizeof(float));
                    memcpy(Udq, GUdq + nGate, nGate * sizeof(float));
                    free(GUdq);
                } else {
                    memcpy(Gdq, Wg, nGate * sizeof(float));
                    memcpy(Udq, Wu, nGate * sizeof(float));
                    int8Scale = quantizeDequantInPlace(Gdq, (int)nGate);
                    (void)quantizeDequantInPlace(Udq, (int)nGate);
                }
                float actScale = 0;
                float xScale = 0;
                if (w8a8) {
                    // Calibrate hid scale on dequant W; optionally also x.
                    float *Xcal = X;
                    float *Xrt = NULL;
                    if (w8a8x) {
                        float xmax = 0;
                        for (int i = 0; i < ic * seq; i++) {
                            float a = fabsf(X[i]);
                            if (a > xmax) xmax = a;
                        }
                        xScale = xmax / 127.0f;
                        if (!(xScale > 0)) xScale = 1.0f;
                        xScaleKept = xScale;
                        Xrt = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
                        if (!Xrt) {
                            free(Gdq); free(Udq); free(Ddq);
                            free(X); free(Ycpu); free(Wg); free(Wu); free(Wd);
                            emitJSON(NO, "x roundtrip alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                            return 1;
                        }
                        memcpy(Xrt, X, (size_t)ic * (size_t)seq * sizeof(float));
                        applyActInt8RoundTrip(Xrt, ic * seq, xScale);
                        Xcal = Xrt;
                    }
                    float hidMax = cpuHidMaxAbs(Gdq, Udq, Xcal, ic, hidden, seq);
                    actScale = hidMax / 127.0f;
                    if (!(actScale > 0)) actScale = 1.0f;
                    if (parity) {
                        cpuSwiGLU(Gdq, Udq, Ddq, Xcal, Ycpu, ic, hidden, seq, actScale);
                    }
                    free(Xrt);
                } else if (parity) {
                    cpuSwiGLU(Gdq, Udq, Ddq, X, Ycpu, ic, hidden, seq, 0);
                }
                free(Gdq); free(Udq); free(Ddq);
                if (fuseGU) {
                    sess = ane_prefill_session_create_swiglu_int8_fused(
                        ic, hidden, seq, Wg, Wu, Wd, w8a8, actScale);
                } else if (w8a8) {
                    sess = ane_prefill_session_create_swiglu_int8_w8a8(
                        ic, hidden, seq, Wg, Wu, Wd, actScale, w8a8x ? xScale : 0, sp0, sp1, int8In);
                } else {
                    sess = ane_prefill_session_create_swiglu_int8(ic, hidden, seq, Wg, Wu, Wd);
                }
                if (sess) {
                    int8Scale = ane_prefill_session_int8_scale(sess);
                }
            } else {
                if (parity) {
                    cpuSwiGLU(Wg, Wu, Wd, X, Ycpu, ic, hidden, seq, 0);
                }
                if (fuseGU) {
                    sess = ane_prefill_session_create_swiglu_fused_gu(ic, hidden, seq, Wg, Wu, Wd);
                } else {
                    sess = ane_prefill_session_create_swiglu(ic, hidden, seq, Wg, Wu, Wd);
                }
            }
        } else {
            W = (float *)malloc((size_t)oc * (size_t)ic * sizeof(float));
            if (!W) {
                free(X); free(Ycpu);
                emitJSON(NO, "weight alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                return 1;
            }
            fillSynth(W, oc * ic, 0.1f, 17, 3);
            if (int8) {
                // Golden uses dequantized weights (same quant as ANE BLOBFILE).
                float *Wdq = (float *)malloc((size_t)oc * (size_t)ic * sizeof(float));
                if (!Wdq) {
                    free(W); free(X); free(Ycpu);
                    emitJSON(NO, "dequant weight alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                    return 1;
                }
                memcpy(Wdq, W, (size_t)oc * (size_t)ic * sizeof(float));
                int8Scale = quantizeDequantInPlace(Wdq, oc * ic);
                if (parity) {
                    cpuMatmul(Wdq, X, Ycpu, ic, oc, seq);
                }
                free(Wdq);
                sess = ane_prefill_session_create_int8(ic, oc, seq, W);
                if (sess) {
                    int8Scale = ane_prefill_session_int8_scale(sess);
                }
            } else {
                if (parity) {
                    cpuMatmul(W, X, Ycpu, ic, oc, seq);
                }
                sess = ane_prefill_session_create(ic, oc, seq, W);
            }
        }

        if (!sess) {
            free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "ane_prefill_session_create failed", @{
                @"mode": [NSString stringWithUTF8String:mode],
                @"ic": @(ic), @"oc": @(oc), @"seq": @(seq),
            });
            return 1;
        }

        uint32_t sid = ane_prefill_session_input_surface_id(sess);
        size_t inBytes = ane_prefill_session_input_bytes(sess);
        size_t outBytes = ane_prefill_session_output_bytes(sess);
        if (sid == 0 || inBytes == 0) {
            ane_prefill_session_destroy(sess);
            free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "input surface unavailable", @{@"mode": [NSString stringWithUTF8String:mode]});
            return 1;
        }

        void *Xin = malloc(inBytes);
        if (!Xin) {
            ane_prefill_session_destroy(sess);
            free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "act input alloc failed", @{@"mode": [NSString stringWithUTF8String:mode]});
            return 1;
        }
        BOOL wrote = NO;
        if (ane_prefill_session_is_int8_input(sess)) {
            int8_t *q = (int8_t *)Xin;
            float xs = xScaleKept > 0 ? xScaleKept : 1.0f;
            for (int i = 0; i < ic * seq; i++) {
                float v = X[i] / xs;
                if (v > 127.0f) v = 127.0f;
                if (v < -128.0f) v = -128.0f;
                q[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
            }
            wrote = ane_prefill_session_write_acts_int8(sess, Xin, inBytes);
        } else {
            _Float16 *X16 = (_Float16 *)Xin;
            for (int i = 0; i < ic * seq; i++) {
                X16[i] = (_Float16)X[i];
            }
            wrote = ane_prefill_session_write_acts_fp16(sess, Xin, inBytes);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        double mapMs = 0;
        if (!wrote) {
            free(Xin);
            ane_prefill_session_destroy(sess);
            free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "write acts failed", @{@"mode": [NSString stringWithUTF8String:mode]});
            return 1;
        }
        if (device) {
            IOSurfaceRef surf = IOSurfaceLookup(sid);
            if (surf) {
                IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
                void *base = (void *)IOSurfaceGetBaseAddress(surf);
                void *mappedBase = NULL;
                size_t mappedSize = 0;
                uint64_t tm0 = mach_absolute_time();
                id<MTLBuffer> mapped = ggml_map_iosurface_base(device, base, inBytes, &mappedBase, &mappedSize);
                mapMs = ticksToMs(mach_absolute_time() - tm0);
                (void)mapped;
                IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
                CFRelease(surf);
            }
        }
        free(Xin);

        for (int i = 0; i < warmup; i++) {
            if (!ane_prefill_session_eval(sess)) {
                ane_prefill_session_destroy(sess);
                free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
                emitJSON(NO, "warmup eval failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                return 1;
            }
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            if (!ane_prefill_session_eval(sess)) {
                ane_prefill_session_destroy(sess);
                free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
                emitJSON(NO, "eval failed", @{@"mode": [NSString stringWithUTF8String:mode]});
                return 1;
            }
        }
        double evalMs = ticksToMs(mach_absolute_time() - t0) / (double)iters;

        _Float16 *Y16 = (_Float16 *)malloc(outBytes);
        float *Yane = (float *)malloc((size_t)outCh * (size_t)seq * sizeof(float));
        if (!Y16 || !Yane || !ane_prefill_session_read_out_fp16(sess, Y16, outBytes)) {
            free(Y16); free(Yane);
            ane_prefill_session_destroy(sess);
            free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu);
            emitJSON(NO, "read output failed", @{@"mode": [NSString stringWithUTF8String:mode]});
            return 1;
        }
        BOOL finite = YES;
        float aneMax = 0;
        for (int i = 0; i < outCh * seq; i++) {
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
            ane_prefill_session_destroy(sess);
            free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu); free(Yane);
            emitJSON(NO, "ANE output all zeros", @{
                @"mode": [NSString stringWithUTF8String:mode],
                @"ic": @(ic), @"oc": @(oc), @"seq": @(seq),
            });
            return 1;
        }

        double goldenCos = 0;
        float maxErr = 0;
        if (parity && finite) {
            goldenCos = cosineSim(Yane, Ycpu, (size_t)outCh * (size_t)seq);
            maxErr = maxAbsErr(Yane, Ycpu, (size_t)outCh * (size_t)seq);
        }

        double compileMs = ane_prefill_session_compile_ms(sess);
        int evalCount = ane_prefill_session_eval_count(sess);
        ane_prefill_session_destroy(sess);
        free(W); free(Wg); free(Wu); free(Wd); free(X); free(Ycpu); free(Yane);

        if (!finite) {
            emitJSON(NO, "non-finite output", @{@"mode": [NSString stringWithUTF8String:mode]});
            return 1;
        }
        if (parity && goldenCos < 0.999) {
            emitJSON(NO, "golden cosine below 0.999", @{
                @"mode": [NSString stringWithUTF8String:mode],
                @"ic": @(ic), @"oc": @(oc), @"seq": @(seq),
                @"golden_cosine": @(goldenCos),
                @"max_abs_err": @(maxErr),
            });
            return 1;
        }

        // FLOPs: matmul 2*ic*oc*seq; swiglu 2*(2*ic*hidden + hidden*ic)*seq
        double gflop = swiglu
            ? 2.0 * (2.0 * (double)ic * (double)hidden + (double)hidden * (double)ic) * (double)seq / 1e9
            : 2.0 * (double)ic * (double)oc * (double)seq / 1e9;
        double tflops = evalMs > 0 ? gflop / (evalMs / 1000.0) : 0;
        NSString *variant = @"fp16-blob";
        NSString *note = @"in-process session; fp16-blob matmul; CPU golden parity";
        if (swiglu) {
            if (int8 && fuseGU && w8a8) {
                variant = @"int8-swiglu-fused-w8a8";
                note = @"int8 fused gate||up + W8A8 hid (calibrated act scale)";
            } else if (int8 && int8In) {
                variant = @"int8-swiglu-w8a8-int8in";
                note = @"int8 SwiGLU; host int8 acts + hid W8A8; half input surface";
            } else if (int8 && w8a8x) {
                variant = @"int8-swiglu-w8a8-x";
                note = @"int8 SwiGLU + W8A8 on x and hid; optional spatial tile";
            } else if (int8 && w8a8) {
                variant = @"int8-swiglu-w8a8";
                note = @"int8 SwiGLU + W8A8 hid (calibrated act scale)";
            } else if (int8 && fuseGU) {
                variant = @"int8-swiglu-fused-gu";
                note = @"int8 fused gate||up + down; golden vs dequant W";
            } else if (int8) {
                variant = @"int8-swiglu";
                note = @"int8 SwiGLU 3x BLOBFILE; golden vs dequant W";
            } else if (fuseGU) {
                variant = @"fp16-swiglu-fused-gu";
                note = @"fp16 gate||up fused 1x1 + slice; CPU golden parity";
            } else {
                variant = @"fp16-swiglu-conv";
                note = @"fused SwiGLU 1x1-conv; CPU golden parity";
            }
        } else if (int8) {
            variant = @"int8-conv";
            note = @"int8-conv BLOBFILE + dequant; golden vs dequantized W";
        }
        NSMutableDictionary *fields = [@{
            @"mode": [NSString stringWithUTF8String:mode],
            @"variant": variant,
            @"ic": @(ic),
            @"oc": @(oc),
            @"seq": @(seq),
            @"surface_id": @(sid),
            @"surface_bytes": @(inBytes),
            @"compile_ms": @(compileMs),
            @"map_ms": @(mapMs),
            @"eval_ms": @(evalMs),
            @"total_ms": @(mapMs + evalMs),
            @"gflop": @(gflop),
            @"tflops": @(tflops),
            @"eval_count": @(evalCount),
            @"kernel_reused": @YES,
            @"ane_max_abs": @(aneMax),
            @"source": @"zerollama/ane_prefill_session",
            @"note": note,
        } mutableCopy];
        if (int8) {
            fields[@"int8_scale"] = @(int8Scale);
        }
        if (w8a8) {
            fields[@"w8a8_hid"] = @YES;
        }
        if (w8a8x) {
            fields[@"w8a8_x"] = @YES;
        }
        if (int8In) {
            fields[@"int8_input"] = @YES;
        }
        if (swiglu && int8 && w8a8) {
            fields[@"sp0"] = @(sp0);
            fields[@"sp1"] = @(sp1);
        }
        if (fuseGU) {
            fields[@"fuse_gu"] = @YES;
        }
        if (swiglu) {
            fields[@"hidden"] = @(hidden);
            fields[@"out_ch"] = @(outCh);
        }
        if (parity) {
            fields[@"golden_cosine"] = @(goldenCos);
            fields[@"max_abs_err"] = @(maxErr);
            fields[@"parity"] = @YES;
        }
        // Report whether ZEROLLAMA_ANE_FFN_* policy would match this geometry (lab probe).
        {
            ane_ffn_policy_t pol;
            ane_ffn_policy_load(&pol);
            int servePort = ane_ffn_policy_parse_host_port(getenv("OLLAMA_HOST"));
            if (servePort == 0) {
                servePort = pol.lab_port;
            }
            ane_ffn_verdict_t verd = ane_ffn_policy_decide(
                &pol, ANE_FFN_OP_MUL_MAT, ic, swiglu ? hidden : oc, seq, servePort,
                "blk.0.ffn_up_shexp.weight");
            fields[@"ffn_policy_enabled"] = @(pol.enabled);
            fields[@"ffn_policy_allow"] = @(verd.allow);
            fields[@"ffn_policy_reason"] = [NSString stringWithUTF8String:verd.reason ? verd.reason : ""];
        }
        emitJSON(YES, NULL, fields);
        return 0;
    }
}
