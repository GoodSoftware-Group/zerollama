// ELIZA-TBQ-SET-ROWS-V1 — Metal SET_ROWS for TBQ3_0 / TBQ4_0 KV cache writes.
// Bit-faithful to ggml-quants.c quantize_row_tbq{3,4}_0_ref / CUDA turboquant.cuh.
#include <metal_stdlib>
using namespace metal;

#define QK_TBQ 32

struct block_tbq3_0 {
    half    d;
    uint8_t qs[QK_TBQ * 3 / 8];
};

struct block_tbq4_0 {
    half    d;
    uint8_t qs[QK_TBQ / 2];
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

constant float k_tbq3_codebook[8] = {
    -2.1519457f, -1.3439093f, -0.7560053f, -0.2450942f,
     0.2450942f,  0.7560053f,  1.3439093f,  2.1519457f,
};

constant float k_tbq4_codebook[16] = {
    -2.7321365f, -2.0685055f, -1.6175243f, -1.2557391f,
    -0.9419147f, -0.6564307f, -0.3878412f, -0.1283243f,
     0.1283243f,  0.3878412f,  0.6564307f,  0.9419147f,
     1.2557391f,  1.6175243f,  2.0685055f,  2.7321365f,
};

constant int8_t k_tbq_signs[QK_TBQ] = {
     1, -1,  1,  1, -1,  1, -1, -1,
     1,  1, -1,  1, -1, -1,  1, -1,
    -1,  1,  1, -1,  1, -1, -1,  1,
     1, -1,  1, -1, -1,  1, -1,  1,
};

static inline void tbq3_set_code(thread uint8_t * qs, int idx, uint8_t code) {
    const int bit = idx * 3;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    qs[byte] = uint8_t(qs[byte] | ((code & 0x7u) << shift));
    if (shift > 5 && byte + 1 < (QK_TBQ * 3 / 8)) {
        qs[byte + 1] = uint8_t(qs[byte + 1] | ((code & 0x7u) >> (8 - shift)));
    }
}

static inline void tbq4_set_code(thread uint8_t * qs, int idx, uint8_t code) {
    const int j = idx % (QK_TBQ / 2);
    if (idx < QK_TBQ / 2) {
        qs[j] = uint8_t((qs[j] & 0xF0) | (code & 0x0F));
    } else {
        qs[j] = uint8_t((qs[j] & 0x0F) | ((code & 0x0F) << 4));
    }
}

static inline void tbq_hadamard32(thread float * x) {
    for (int len = 1; len < QK_TBQ; len <<= 1) {
        for (int i = 0; i < QK_TBQ; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float norm = 0.1767766952966369f;
    for (int i = 0; i < QK_TBQ; ++i) {
        x[i] *= norm;
    }
}

static inline void tbq_precondition_block(device const float * x, thread float * y) {
    for (int i = 0; i < QK_TBQ; ++i) {
        y[i] = x[i] * float(k_tbq_signs[i]);
    }
    tbq_hadamard32(y);
}

static inline uint8_t tbq_best_index(int n, constant float * codebook, float x) {
    if (x <= codebook[0]) {
        return 0;
    }
    if (x >= codebook[n - 1]) {
        return uint8_t(n - 1);
    }
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (x < codebook[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return uint8_t((x - codebook[lo] <= codebook[hi] - x) ? lo : hi);
}

static inline void quantize_tbq3_0(device const float * src, device block_tbq3_0 & dst) {
    float rotated[QK_TBQ];
    tbq_precondition_block(src, rotated);

    float sumsq = 0.0f;
    for (int i = 0; i < QK_TBQ; ++i) {
        sumsq += rotated[i] * rotated[i];
    }
    const float d = sqrt(sumsq / float(QK_TBQ));
    dst.d = half(d);

    uint8_t qs[QK_TBQ * 3 / 8];
    for (int i = 0; i < QK_TBQ * 3 / 8; ++i) {
        qs[i] = 0;
    }
    if (d != 0.0f) {
        const float id = 1.0f / d;
        for (int i = 0; i < QK_TBQ; ++i) {
            tbq3_set_code(qs, i, tbq_best_index(8, k_tbq3_codebook, rotated[i] * id));
        }
    }
    for (int i = 0; i < QK_TBQ * 3 / 8; ++i) {
        dst.qs[i] = qs[i];
    }
}

static inline void quantize_tbq4_0(device const float * src, device block_tbq4_0 & dst) {
    float rotated[QK_TBQ];
    tbq_precondition_block(src, rotated);

    float sumsq = 0.0f;
    for (int i = 0; i < QK_TBQ; ++i) {
        sumsq += rotated[i] * rotated[i];
    }
    const float d = sqrt(sumsq / float(QK_TBQ));
    dst.d = half(d);

    uint8_t qs[QK_TBQ / 2];
    for (int i = 0; i < QK_TBQ / 2; ++i) {
        qs[i] = 0;
    }
    if (d != 0.0f) {
        const float id = 1.0f / d;
        for (int i = 0; i < QK_TBQ; ++i) {
            tbq4_set_code(qs, i, tbq_best_index(16, k_tbq4_codebook, rotated[i] * id));
        }
    }
    for (int i = 0; i < QK_TBQ / 2; ++i) {
        dst.qs[i] = qs[i];
    }
}

template<typename TI, typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_tbq(
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

    const TI i1 = ((const device TI *)((const device char *)src1 + i01 * args.nb10 + i11 * args.nb11 + i12 * args.nb12))[0];

    device block_q * dst_row = (device block_q *)((device char *)dst + i1 * args.nb1 + i02 * args.nb2 + i03 * args.nb3);
    const device float * src_row = (const device float *)((const device char *)src0 + i01 * args.nb01 + i02 * args.nb02 + i03 * args.nb03);

    for (int ind = tiitg % tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + QK_TBQ * ind, dst_row[ind]);
    }
}

typedef decltype(kernel_set_rows_tbq<long, block_tbq3_0, quantize_tbq3_0>) set_rows_tbq3_t;
typedef decltype(kernel_set_rows_tbq<long, block_tbq4_0, quantize_tbq4_0>) set_rows_tbq4_t;

template [[host_name("kernel_set_rows_f32_i64_tbq3_0")]] kernel set_rows_tbq3_t kernel_set_rows_tbq<long, block_tbq3_0, quantize_tbq3_0>;
template [[host_name("kernel_set_rows_f32_i32_tbq3_0")]] kernel set_rows_tbq3_t kernel_set_rows_tbq<int,  block_tbq3_0, quantize_tbq3_0>;
template [[host_name("kernel_set_rows_f32_i64_tbq4_0")]] kernel set_rows_tbq4_t kernel_set_rows_tbq<long, block_tbq4_0, quantize_tbq4_0>;
template [[host_name("kernel_set_rows_f32_i32_tbq4_0")]] kernel set_rows_tbq4_t kernel_set_rows_tbq<int,  block_tbq4_0, quantize_tbq4_0>;
