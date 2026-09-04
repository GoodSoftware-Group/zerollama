// ggml ne0-first ↔ ANE channel-major pack (lab force path).
#include "ane_ffn_force_pack.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

// Soft IEEE754 binary16 ↔ float (enough for lab parity; Metal uses ggml when available).
static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // denormal
            float m = (float)mant / 1024.0f;
            float v = ldexpf(m, -14);
            union { float f; uint32_t u; } u = { .f = v };
            out = (u.u & 0x7fffffffu) | sign;
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u = { .u = out };
    return u.f;
}

static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } in = { .f = f };
    const uint32_t sign = (in.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((in.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = in.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x800000u;
        uint32_t t = mant >> (1 - exp + 13);
        if ((mant >> (1 - exp + 12)) & 1u) {
            t += 1u;
        }
        return (uint16_t)(sign | t);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u); // inf
    }
    uint16_t out = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if (mant & 0x1000u) {
        out = (uint16_t)(out + 1u);
    }
    return out;
}

bool ane_ffn_pack_weight_to_f32(
    const void *src, int is_f16, int ic, int oc, float *dst_oc_ic) {
    if (!src || !dst_oc_ic || ic <= 0 || oc <= 0) {
        return false;
    }
    const size_t n = (size_t)ic * (size_t)oc;
    if (is_f16) {
        const uint16_t *s = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            dst_oc_ic[i] = f16_to_f32(s[i]);
        }
    } else {
        memcpy(dst_oc_ic, src, n * sizeof(float));
    }
    return true;
}

bool ane_ffn_pack_acts_ggml_to_channel(
    const void *src, int is_f16, int ic, int seq, float *dst_ic_seq) {
    if (!src || !dst_ic_seq || ic <= 0 || seq <= 0) {
        return false;
    }
    if (is_f16) {
        const uint16_t *s = (const uint16_t *)src;
        for (int i = 0; i < ic; i++) {
            for (int t = 0; t < seq; t++) {
                dst_ic_seq[i * seq + t] = f16_to_f32(s[t * ic + i]);
            }
        }
    } else {
        const float *s = (const float *)src;
        for (int i = 0; i < ic; i++) {
            for (int t = 0; t < seq; t++) {
                dst_ic_seq[i * seq + t] = s[t * ic + i];
            }
        }
    }
    return true;
}

bool ane_ffn_pack_acts_ggml_to_channel_f16(
    const void *src, int is_f16, int ic, int seq, void *dst_ic_seq_f16) {
    if (!src || !dst_ic_seq_f16 || ic <= 0 || seq <= 0) {
        return false;
    }
    uint16_t *d = (uint16_t *)dst_ic_seq_f16;
    const int BS = 64;
    if (is_f16) {
        const uint16_t *s = (const uint16_t *)src;
        for (int i0 = 0; i0 < ic; i0 += BS) {
            int i1 = i0 + BS < ic ? i0 + BS : ic;
            for (int t0 = 0; t0 < seq; t0 += BS) {
                int t1 = t0 + BS < seq ? t0 + BS : seq;
                for (int i = i0; i < i1; i++) {
                    uint16_t *row = d + (size_t)i * (size_t)seq;
                    for (int t = t0; t < t1; t++) {
                        row[t] = s[(size_t)t * (size_t)ic + (size_t)i];
                    }
                }
            }
        }
    } else {
        const float *s = (const float *)src;
        for (int i0 = 0; i0 < ic; i0 += BS) {
            int i1 = i0 + BS < ic ? i0 + BS : ic;
            for (int t0 = 0; t0 < seq; t0 += BS) {
                int t1 = t0 + BS < seq ? t0 + BS : seq;
                for (int i = i0; i < i1; i++) {
                    uint16_t *row = d + (size_t)i * (size_t)seq;
                    for (int t = t0; t < t1; t++) {
                        row[t] = f32_to_f16(s[(size_t)t * (size_t)ic + (size_t)i]);
                    }
                }
            }
        }
    }
    return true;
}

static int8_t quant_f32_to_i8(float x, float scale) {
    float v = x / scale;
    if (v > 127.0f) v = 127.0f;
    if (v < -128.0f) v = -128.0f;
    return (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
}

bool ane_ffn_pack_acts_ggml_to_channel_i8(
    const void *src, int is_f16, int ic, int seq, float scale, int8_t *dst_ic_seq) {
    if (!src || !dst_ic_seq || ic <= 0 || seq <= 0 || !(scale > 0)) {
        return false;
    }
    const int BS = 64;
    if (is_f16) {
        const uint16_t *s = (const uint16_t *)src;
        for (int i0 = 0; i0 < ic; i0 += BS) {
            int i1 = i0 + BS < ic ? i0 + BS : ic;
            for (int t0 = 0; t0 < seq; t0 += BS) {
                int t1 = t0 + BS < seq ? t0 + BS : seq;
                for (int i = i0; i < i1; i++) {
                    int8_t *row = dst_ic_seq + (size_t)i * (size_t)seq;
                    for (int t = t0; t < t1; t++) {
                        row[t] = quant_f32_to_i8(f16_to_f32(s[(size_t)t * (size_t)ic + (size_t)i]), scale);
                    }
                }
            }
        }
    } else {
        const float *s = (const float *)src;
        for (int i0 = 0; i0 < ic; i0 += BS) {
            int i1 = i0 + BS < ic ? i0 + BS : ic;
            for (int t0 = 0; t0 < seq; t0 += BS) {
                int t1 = t0 + BS < seq ? t0 + BS : seq;
                for (int i = i0; i < i1; i++) {
                    int8_t *row = dst_ic_seq + (size_t)i * (size_t)seq;
                    for (int t = t0; t < t1; t++) {
                        row[t] = quant_f32_to_i8(s[(size_t)t * (size_t)ic + (size_t)i], scale);
                    }
                }
            }
        }
    }
    return true;
}

bool ane_ffn_unpack_dst_channel_to_ggml(
    const float *src_oc_seq, int is_f16, int oc, int seq, void *dst) {
    if (!src_oc_seq || !dst || oc <= 0 || seq <= 0) {
        return false;
    }
    if (is_f16) {
        uint16_t *d = (uint16_t *)dst;
        for (int o = 0; o < oc; o++) {
            for (int t = 0; t < seq; t++) {
                d[t * oc + o] = f32_to_f16(src_oc_seq[o * seq + t]);
            }
        }
    } else {
        float *d = (float *)dst;
        for (int o = 0; o < oc; o++) {
            for (int t = 0; t < seq; t++) {
                d[t * oc + o] = src_oc_seq[o * seq + t];
            }
        }
    }
    return true;
}

bool ane_ffn_unpack_dst_channel_f16_to_ggml(
    const void *src_oc_seq_f16, int dst_is_f16, int oc, int seq, void *dst) {
    if (!src_oc_seq_f16 || !dst || oc <= 0 || seq <= 0) {
        return false;
    }
    const uint16_t *s = (const uint16_t *)src_oc_seq_f16;
    // Blocked transpose for cache (host ggml↔channel is the hot tax — F0742).
    const int BS = 64;
    if (dst_is_f16) {
        uint16_t *d = (uint16_t *)dst;
        for (int o0 = 0; o0 < oc; o0 += BS) {
            int o1 = o0 + BS < oc ? o0 + BS : oc;
            for (int t0 = 0; t0 < seq; t0 += BS) {
                int t1 = t0 + BS < seq ? t0 + BS : seq;
                for (int o = o0; o < o1; o++) {
                    const uint16_t *row = s + (size_t)o * (size_t)seq;
                    for (int t = t0; t < t1; t++) {
                        d[(size_t)t * (size_t)oc + (size_t)o] = row[t];
                    }
                }
            }
        }
    } else {
        float *d = (float *)dst;
        for (int o0 = 0; o0 < oc; o0 += BS) {
            int o1 = o0 + BS < oc ? o0 + BS : oc;
            for (int t0 = 0; t0 < seq; t0 += BS) {
                int t1 = t0 + BS < seq ? t0 + BS : seq;
                for (int o = o0; o < o1; o++) {
                    const uint16_t *row = s + (size_t)o * (size_t)seq;
                    for (int t = t0; t < t1; t++) {
                        d[(size_t)t * (size_t)oc + (size_t)o] = f16_to_f32(row[t]);
                    }
                }
            }
        }
    }
    return true;
}

bool ane_ffn_fold_out_scale_f32(
    float *W_oc_ic, int ic, int oc,
    const void *scale, int scale_is_f16, int nscale) {
    if (!W_oc_ic || !scale || ic <= 0 || oc <= 0) {
        return false;
    }
    if (nscale != 1 && nscale != oc) {
        return false;
    }
    for (int o = 0; o < oc; o++) {
        float s;
        if (scale_is_f16) {
            const uint16_t *sp = (const uint16_t *)scale;
            s = f16_to_f32(sp[nscale == 1 ? 0 : o]);
        } else {
            const float *sp = (const float *)scale;
            s = sp[nscale == 1 ? 0 : o];
        }
        float *row = W_oc_ic + (size_t)o * (size_t)ic;
        for (int i = 0; i < ic; i++) {
            row[i] *= s;
        }
    }
    return true;
}
