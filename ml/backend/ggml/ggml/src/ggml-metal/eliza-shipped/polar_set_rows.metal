// ELIZA-POLAR-SET-ROWS-V1 — Metal SET_ROWS for Q4_POLAR KV cache writes.
// Bit-faithful to ggml-quants.c quantize_row_q4_polar_ref / polar_centroids.h.
#include <metal_stdlib>
using namespace metal;

#define QK_POLAR 128
#define QJL_RESIDUAL_BYTES (QK_POLAR / 8)
#define POLAR_Q4_N_LEVELS 16
#define POLAR_QJL_SEED 42u

struct block_q4_polar {
    half    d;
    uint8_t qs[QK_POLAR / 2];
    uint8_t qjl[QJL_RESIDUAL_BYTES];
};

struct ggml_metal_kargs_set_rows {
    int   nk0;
    int   ne01;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int   ne11;
    int   ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb1;
    ulong nb2;
    ulong nb3;
};

constant float POLAR_Q4_CENTROIDS[POLAR_Q4_N_LEVELS] = {
    -2.754354807e+00f, -2.093562707e+00f, -1.643041510e+00f, -1.279739752e+00f,
    -9.626409783e-01f, -6.723921169e-01f, -3.978971029e-01f, -1.317577823e-01f,
     1.317577823e-01f,  3.978971029e-01f,  6.723921169e-01f,  9.626409783e-01f,
     1.279739752e+00f,  1.643041510e+00f,  2.093562707e+00f,  2.754354807e+00f,
};

constant float POLAR_Q4_BOUNDARIES[POLAR_Q4_N_LEVELS - 1] = {
    -2.423958757e+00f, -1.868302108e+00f, -1.461390631e+00f, -1.121190365e+00f,
    -8.175165476e-01f, -5.351446099e-01f, -2.648274426e-01f,  4.996003611e-16f,
     2.648274426e-01f,  5.351446099e-01f,  8.175165476e-01f,  1.121190365e+00f,
     1.461390631e+00f,  1.868302108e+00f,  2.423958757e+00f,
};

static inline void polar_hadamard_inplace(thread float * x) {
    for (int h = 1; h < QK_POLAR; h <<= 1) {
        for (int i = 0; i < QK_POLAR; i += (h << 1)) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

static inline void polar_qjl_signs(thread float * out) {
    uint state = POLAR_QJL_SEED;
    if (state == 0u) state = 1u;
    for (int i = 0; i < QK_POLAR; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out[i] = (state & 1u) ? 1.0f : -1.0f;
    }
}

static inline uint8_t polar_q4_bucketize(float v) {
    uint8_t code = 0;
    for (int i = 0; i < POLAR_Q4_N_LEVELS - 1; i++) {
        if (v > POLAR_Q4_BOUNDARIES[i]) {
            code = (uint8_t)(i + 1);
        }
    }
    return code;
}

static inline void quantize_q4_polar(device const float * src, device block_q4_polar & dst) {
    float qjl_signs[QK_POLAR];
    polar_qjl_signs(qjl_signs);

    float sumsq = 0.0f;
    for (int i = 0; i < QK_POLAR; i++) {
        sumsq += src[i] * src[i];
    }
    const float l2     = sqrt(sumsq);
    const float inv_l2 = (l2 > 1e-10f) ? (1.0f / l2) : 0.0f;
    dst.d = (half) l2;

    float buf[QK_POLAR];
    for (int i = 0; i < QK_POLAR; i++) {
        buf[i] = src[i] * inv_l2;
    }
    polar_hadamard_inplace(buf);

    uint8_t codes[QK_POLAR];
    for (int i = 0; i < QK_POLAR; i++) {
        codes[i] = polar_q4_bucketize(buf[i]);
    }
    for (int i = 0; i < QK_POLAR / 2; i++) {
        const uint8_t lo = codes[2 * i];
        const uint8_t hi = codes[2 * i + 1];
        dst.qs[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
    }

    for (int i = 0; i < QJL_RESIDUAL_BYTES; i++) {
        dst.qjl[i] = 0;
    }
    float proj = 0.0f;
    for (int i = 0; i < QK_POLAR; i++) {
        const float c = POLAR_Q4_CENTROIDS[codes[i]];
        proj += (buf[i] - c) * qjl_signs[i];
    }
    dst.qjl[0] = (proj >= 0.0f) ? 1u : 0u;
}

template<typename TI>
kernel void kernel_set_rows_polar(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03 % args.ne12;
    const int32_t i11 = i02 % args.ne11;

    const int32_t i01 = tgpig.x * tptg.y + tiitg / tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *)((const device char *)src1 + i10 * args.nb10 + i11 * args.nb11 + i12 * args.nb12))[0];

    device block_q4_polar * dst_row = (device block_q4_polar *)((device char *)dst + i1 * args.nb1 + i02 * args.nb2 + i03 * args.nb3);
    const device float * src_row = (const device float *)((const device char *)src0 + i01 * args.nb01 + i02 * args.nb02 + i03 * args.nb03);

    for (int ind = tiitg % tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_q4_polar(src_row + QK_POLAR * ind, dst_row[ind]);
    }
}

typedef decltype(kernel_set_rows_polar<long>) set_rows_q4_polar_t;
template [[host_name("kernel_set_rows_f32_i64_q4_polar")]] kernel set_rows_q4_polar_t kernel_set_rows_polar<long>;
template [[host_name("kernel_set_rows_f32_i32_q4_polar")]] kernel set_rows_q4_polar_t kernel_set_rows_polar<int>;
