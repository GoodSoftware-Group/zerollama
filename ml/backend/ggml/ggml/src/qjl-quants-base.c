/*
 * ggml-base shim for the QJL1_256 type-traits entries.
 *
 * ggml.c's type_traits[] table (compiled into ggml-base) references
 * quantize_row_qjl1_256_ref / dequantize_row_qjl1_256 / quantize_qjl1_256
 * directly and unconditionally, but ggml-base does not (and must not)
 * link against ggml-cpu. The full-featured/SIMD-dispatched entry points
 * (quantize_row_qjl1_256, ggml_compute_forward_attn_score_qjl) still live
 * in ggml-cpu/qjl/quants-qjl.c alongside the GGML_OP_ATTN_SCORE_QJL op;
 * this shim only needs the scalar reference path (qjl_quantize_row_ref /
 * qjl_dequantize_row_ref / qjl_make_projection_mt, built alongside it from
 * ggml-cpu/qjl/qjl_quantize_ref.c + qjl_projection.c — see CMakeLists.txt)
 * to satisfy the type-traits table on its own.
 */

#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#include "qjl/qjl.h"

#include <stdint.h>
#include <stdlib.h>

/* Portable atomic primitives for the lazy-init CAS below (mirrors
 * ggml-cpu/qjl/quants-qjl.c). */
#if defined(_MSC_VER)
#include <windows.h>
typedef volatile long qjl_base_atomic_int;
static inline int qjl_base_atomic_load_acquire(qjl_base_atomic_int * p) {
    long v = *p;
    MemoryBarrier();
    return (int) v;
}
static inline void qjl_base_atomic_store_release(qjl_base_atomic_int * p, int v) {
    MemoryBarrier();
    _InterlockedExchange(p, (long) v);
}
static inline int qjl_base_atomic_cas_acq_rel(qjl_base_atomic_int * p, int expected, int desired) {
    long prev = _InterlockedCompareExchange(p, (long) desired, (long) expected);
    return prev == (long) expected;
}
#else
typedef volatile int qjl_base_atomic_int;
static inline int qjl_base_atomic_load_acquire(qjl_base_atomic_int * p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static inline void qjl_base_atomic_store_release(qjl_base_atomic_int * p, int v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static inline int qjl_base_atomic_cas_acq_rel(qjl_base_atomic_int * p, int expected, int desired) {
    return __atomic_compare_exchange_n(p, &expected, desired, 0,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
#endif

#define QJL_BASE_INIT_UNINIT  0
#define QJL_BASE_INIT_RUNNING 1
#define QJL_BASE_INIT_READY   2

static qjl_base_atomic_int g_qjl_base_prj_state = QJL_BASE_INIT_UNINIT;
static float * g_qjl_base_prj = NULL;

static const float * qjl_base_default_projection(void) {
    int state = qjl_base_atomic_load_acquire(&g_qjl_base_prj_state);
    if (state == QJL_BASE_INIT_READY) {
        return g_qjl_base_prj;
    }

    if (qjl_base_atomic_cas_acq_rel(&g_qjl_base_prj_state, QJL_BASE_INIT_UNINIT, QJL_BASE_INIT_RUNNING)) {
        g_qjl_base_prj = (float *) malloc(sizeof(float) * QJL_HEAD_DIM * QJL_PROJECTION_DIM);
        if (g_qjl_base_prj != NULL) {
            qjl_make_projection_mt(g_qjl_base_prj, QJL_HEAD_DIM, QJL_PROJECTION_DIM, 42ULL);
        }
        qjl_base_atomic_store_release(&g_qjl_base_prj_state, QJL_BASE_INIT_READY);
        return g_qjl_base_prj;
    }

    while (qjl_base_atomic_load_acquire(&g_qjl_base_prj_state) != QJL_BASE_INIT_READY) {
        /* spin briefly; runs at most once per process lifetime */
    }
    return g_qjl_base_prj;
}

void quantize_row_qjl1_256_ref(const float * GGML_RESTRICT x, block_qjl1_256 * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k > 0);
    GGML_ASSERT((k % QJL_HEAD_DIM) == 0);
    const int64_t n_blocks = k / QJL_HEAD_DIM;
    const float * prj = qjl_base_default_projection();
    GGML_ASSERT(prj != NULL);

    for (int64_t r = 0; r < n_blocks; r++) {
        qjl_quantize_row_ref(x + r * QJL_HEAD_DIM, prj, (qjl_block_qjl1_256 *) (y + r));
    }
}

void dequantize_row_qjl1_256(const block_qjl1_256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k > 0);
    GGML_ASSERT((k % QJL_HEAD_DIM) == 0);
    const int64_t n_blocks = k / QJL_HEAD_DIM;
    const float * prj = qjl_base_default_projection();
    GGML_ASSERT(prj != NULL);

    for (int64_t r = 0; r < n_blocks; r++) {
        qjl_dequantize_row_ref((const qjl_block_qjl1_256 *) (x + r), prj, y + r * QJL_HEAD_DIM);
    }
}

size_t quantize_qjl1_256(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_QJL1_256, n_per_row);
    quantize_row_qjl1_256_ref(src, (block_qjl1_256 *) dst, (int64_t) nrow * n_per_row);
    return (size_t) nrow * row_size;
}
