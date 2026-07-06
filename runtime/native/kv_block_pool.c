/*
 * Phase 15: native KV block allocator + v8 page bind / decode batch layout.
 *
 * Why C (not Python): admission ticks and decode batches run on every request;
 * keeping block-pool bookkeeping, page-table resolve, and batch field layout here
 * reduces GIL contention when inference shares embedded CPython with training.
 *
 * v8 partial: page_bind_* registers PA block_ids per kv_slot (seq-position bind).
 * Tensor KV pages remain inside llama until upstream exposes stable handles.
 * v47: page_bind_external_alias_probe + page_bind_alias_validate (patch 0019) —
 * classify external-buffer alias feasibility without mutating ggml tensors.
 * decode_batch_layout / decode_prefill_chunks prepare llama_batch metadata; llama_decode
 * still runs from Python ctypes until a libllama-linked extension exists.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kv_decode_loop.h"
#include "kv_page_bind_internal.h"
#include "kv_tensor_probe.h"

#ifdef ZEROLLAMA_KV_DECODE_LOOP
#include "llama.h"
#endif
#if defined(ZEROLLAMA_KV_DECODE_LOOP) || defined(LLAMA_KV_EXT_EXTERNAL_ALIAS)
#include "llama-kv-ext.h"
#endif

static PyObject *BlockPoolError = NULL;

typedef struct {
    PyObject_HEAD
    int num_blocks;
    int block_size;
    int device_id;
    int *free_ids;
    int free_len;
} KvBlockPool;

static int
blocks_for_tokens(int block_size, int num_tokens)
{
    if (num_tokens <= 0) {
        return 0;
    }
    return (num_tokens + block_size - 1) / block_size;
}

static int
kv_pool_init_free_stack(KvBlockPool *self)
{
    self->free_ids = (int *)PyMem_Malloc((size_t)self->num_blocks * sizeof(int));
    if (self->free_ids == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->free_len = self->num_blocks;
    for (int i = 0; i < self->num_blocks; i++) {
        self->free_ids[i] = self->num_blocks - 1 - i;
    }
    return 0;
}

static void
kv_pool_clear_free_stack(KvBlockPool *self)
{
    if (self->free_ids != NULL) {
        PyMem_Free(self->free_ids);
        self->free_ids = NULL;
    }
    self->free_len = 0;
}

static int
kv_pool_init(KvBlockPool *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"num_blocks", "block_size", "device_id", NULL};
    int num_blocks = 0;
    int block_size = 0;
    int device_id = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "ii|i", kwlist, &num_blocks, &block_size, &device_id)) {
        return -1;
    }
    if (num_blocks <= 0) {
        PyErr_SetString(PyExc_ValueError, "num_blocks must be positive");
        return -1;
    }
    if (block_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "block_size must be positive");
        return -1;
    }
    self->num_blocks = num_blocks;
    self->block_size = block_size;
    self->device_id = device_id;
    return kv_pool_init_free_stack(self);
}

static void
kv_pool_dealloc(KvBlockPool *self)
{
    kv_pool_clear_free_stack(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
kv_pool_get_num_blocks(KvBlockPool *self, void *closure)
{
    (void)closure;
    return PyLong_FromLong(self->num_blocks);
}

static PyObject *
kv_pool_get_block_size(KvBlockPool *self, void *closure)
{
    (void)closure;
    return PyLong_FromLong(self->block_size);
}

static PyObject *
kv_pool_get_device_id(KvBlockPool *self, void *closure)
{
    (void)closure;
    return PyLong_FromLong(self->device_id);
}

static PyObject *
kv_pool_get_num_free(KvBlockPool *self, void *closure)
{
    (void)closure;
    return PyLong_FromLong(self->free_len);
}

static PyObject *
kv_pool_get_utilization(KvBlockPool *self, void *closure)
{
    (void)closure;
    if (self->num_blocks == 0) {
        return PyFloat_FromDouble(0.0);
    }
    double u = 1.0 - ((double)self->free_len / (double)self->num_blocks);
    return PyFloat_FromDouble(u);
}

static PyGetSetDef kv_pool_getset[] = {
    {"num_blocks", (getter)kv_pool_get_num_blocks, NULL, NULL, 0},
    {"block_size", (getter)kv_pool_get_block_size, NULL, NULL, 0},
    {"device_id", (getter)kv_pool_get_device_id, NULL, NULL, 0},
    {"num_free", (getter)kv_pool_get_num_free, NULL, NULL, 0},
    {"utilization", (getter)kv_pool_get_utilization, NULL, NULL, 0},
    {NULL},
};

static PyObject *
kv_pool_blocks_for_tokens(KvBlockPool *self, PyObject *args)
{
    int num_tokens = 0;
    if (!PyArg_ParseTuple(args, "i", &num_tokens)) {
        return NULL;
    }
    return PyLong_FromLong(blocks_for_tokens(self->block_size, num_tokens));
}

static PyObject *
kv_pool_can_allocate(KvBlockPool *self, PyObject *args)
{
    int n_blocks = 0;
    if (!PyArg_ParseTuple(args, "i", &n_blocks)) {
        return NULL;
    }
    return PyBool_FromLong(n_blocks <= self->free_len);
}

static PyObject *
kv_pool_allocate(KvBlockPool *self, PyObject *args)
{
    int n_blocks = 0;
    if (!PyArg_ParseTuple(args, "i", &n_blocks)) {
        return NULL;
    }
    if (n_blocks < 0) {
        PyErr_SetString(PyExc_ValueError, "n_blocks must be non-negative");
        return NULL;
    }
    if (n_blocks > self->free_len) {
        PyErr_Format(
            BlockPoolError,
            "device %d: need %d blocks, %d free",
            self->device_id,
            n_blocks,
            self->free_len);
        return NULL;
    }
    PyObject *out = PyList_New((Py_ssize_t)n_blocks);
    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < n_blocks; i++) {
        int bid = self->free_ids[--self->free_len];
        PyObject *item = PyLong_FromLong(bid);
        if (item == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(out, (Py_ssize_t)i, item);
    }
    return out;
}

static int
kv_pool_contains_free(KvBlockPool *self, int bid)
{
    for (int i = 0; i < self->free_len; i++) {
        if (self->free_ids[i] == bid) {
            return 1;
        }
    }
    return 0;
}

static PyObject *
kv_pool_free(KvBlockPool *self, PyObject *args)
{
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "O", &seq)) {
        return NULL;
    }
    PyObject *iter = PyObject_GetIter(seq);
    if (iter == NULL) {
        return NULL;
    }
    PyObject *item = NULL;
    while ((item = PyIter_Next(iter)) != NULL) {
        long bid = PyLong_AsLong(item);
        Py_DECREF(item);
        if (bid == -1 && PyErr_Occurred()) {
            Py_DECREF(iter);
            return NULL;
        }
        if (bid < 0 || bid >= self->num_blocks) {
            Py_DECREF(iter);
            PyErr_Format(PyExc_ValueError, "invalid block id %ld", bid);
            return NULL;
        }
        if (kv_pool_contains_free(self, (int)bid)) {
            Py_DECREF(iter);
            PyErr_Format(BlockPoolError, "block %ld already free", bid);
            return NULL;
        }
        if (self->free_len >= self->num_blocks) {
            Py_DECREF(iter);
            PyErr_SetString(PyExc_RuntimeError, "free stack overflow");
            return NULL;
        }
        self->free_ids[self->free_len++] = (int)bid;
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
kv_pool_reset(KvBlockPool *self, PyObject *Py_UNUSED(ignored))
{
    kv_pool_clear_free_stack(self);
    if (kv_pool_init_free_stack(self) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static unsigned long long g_scheduler_tick = 0;
static unsigned long long g_decode_steps = 0;

static KvPageBind g_page_binds[KV_MAX_PAGE_BINDS];
static unsigned long long g_page_bind_registers = 0;
/* Table updates assume CPython GIL (single interpreter thread mutating binds). */

KvPageBind *
kv_find_page_bind(int kv_slot)
{
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (g_page_binds[i].active && g_page_binds[i].kv_slot == kv_slot) {
            return &g_page_binds[i];
        }
    }
    return NULL;
}

int
kv_page_bind_validate_range(int kv_slot, int token_start, int n_tokens)
{
    if (n_tokens <= 0 || token_start < 0) {
        return 0;
    }
    KvPageBind *bind = kv_find_page_bind(kv_slot);
    if (bind == NULL || !bind->active || bind->num_pages <= 0 || bind->block_size <= 0) {
        return 0;
    }
    const int last_pos = token_start + n_tokens - 1;
    const int page_first = token_start / bind->block_size;
    const int page_last = last_pos / bind->block_size;
    if (page_first >= bind->num_pages || page_last >= bind->num_pages) {
        return -2;
    }
    return 0;
}

static KvPageBind *
kv_alloc_page_bind_slot(int kv_slot)
{
    KvPageBind *existing = kv_find_page_bind(kv_slot);
    if (existing != NULL) {
        existing->active = 0;
        existing->num_pages = 0;
        return existing;
    }
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (!g_page_binds[i].active) {
            memset(&g_page_binds[i], 0, sizeof(g_page_binds[i]));
            g_page_binds[i].kv_slot = kv_slot;
            return &g_page_binds[i];
        }
    }
    PyErr_SetString(PyExc_RuntimeError, "page bind table full");
    return NULL;
}

static int
kv_count_active_page_binds(void)
{
    int n = 0;
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (g_page_binds[i].active) {
            n++;
        }
    }
    return n;
}

static int
kv_any_tensor_pages_bound(void)
{
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (g_page_binds[i].active && g_page_binds[i].tensor_pages_bound_slot) {
            return 1;
        }
    }
    return 0;
}

static int
kv_any_physical_pages_bound(void)
{
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (g_page_binds[i].active && g_page_binds[i].physical_pages_bound) {
            return 1;
        }
    }
    return 0;
}

static const char *
kv_tensor_blocker_str(int code)
{
    switch (code) {
    case KV_TENSOR_BLOCKER_NONE:
        return "";
    case KV_TENSOR_BLOCKER_UNSUPPORTED_MEM:
        return "unsupported_memory_type";
    case KV_TENSOR_BLOCKER_CELL_GAP:
        return "cell_map_gap";
    case KV_TENSOR_BLOCKER_NO_TENSOR:
        return "kv_tensor_not_materialized";
    case KV_TENSOR_BLOCKER_MISALIGNED:
        return "pa_cap_exceeded";
    default:
        return "no_public_kv_page_handle_api";
    }
}

static const char *
kv_tensor_memory_kind_str(int kind)
{
    switch (kind) {
    case 1:
        return "kv_cache";
    case 2:
        return "iswa_base";
    case 3:
        return "hybrid_attn";
    case 4:
        return "hybrid_iswa_base";
    case 5:
        return "unsupported";
    default:
        return "none";
    }
}

static PyObject *
kv_native_page_bind_set(PyObject *Py_UNUSED(self), PyObject *args)
{
    int kv_slot = 0;
    int block_size = 0;
    PyObject *block_ids = NULL;
    if (!PyArg_ParseTuple(args, "iiO", &kv_slot, &block_size, &block_ids)) {
        return NULL;
    }
    if (kv_slot < 0) {
        PyErr_SetString(PyExc_ValueError, "kv_slot must be non-negative");
        return NULL;
    }
    if (block_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "block_size must be positive");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(block_ids, "block_ids must be a sequence");
    if (seq == NULL) {
        return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "block_ids must be non-empty");
        return NULL;
    }
    if (n > KV_MAX_PAGES_PER_BIND) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "block_ids too long (%zd pages)", n);
        return NULL;
    }
    KvPageBind *bind = kv_alloc_page_bind_slot(kv_slot);
    if (bind == NULL) {
        Py_DECREF(seq);
        return NULL;
    }
    bind->block_size = block_size;
    bind->num_pages = (int)n;
    bind->cell_pages_bound = 0;
    bind->tensor_pages_bound_slot = 0;
    bind->physical_pages_bound = 0;
    bind->physical_pages_mapped = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long bid = PyLong_AsLong(item);
        if (bid == -1 && PyErr_Occurred()) {
            bind->active = 0;
            bind->num_pages = 0;
            Py_DECREF(seq);
            return NULL;
        }
        if (bid < 0) {
            bind->active = 0;
            bind->num_pages = 0;
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError, "invalid block id %ld", bid);
            return NULL;
        }
        bind->block_ids[i] = (int)bid;
    }
    bind->active = 1;
    g_page_bind_registers++;
    Py_DECREF(seq);
    Py_RETURN_NONE;
}

static PyObject *
kv_native_page_bind_clear(PyObject *Py_UNUSED(self), PyObject *args)
{
    int kv_slot = 0;
    if (!PyArg_ParseTuple(args, "i", &kv_slot)) {
        return NULL;
    }
    KvPageBind *bind = kv_find_page_bind(kv_slot);
    if (bind != NULL) {
        bind->active = 0;
        bind->num_pages = 0;
        bind->cell_pages_bound = 0;
        bind->tensor_pages_bound_slot = 0;
        bind->physical_pages_bound = 0;
        bind->physical_pages_mapped = 0;
    }
    Py_RETURN_NONE;
}

static PyObject *
kv_native_page_bind_resolve(PyObject *Py_UNUSED(self), PyObject *args)
{
    int kv_slot = 0;
    int token_pos = 0;
    if (!PyArg_ParseTuple(args, "ii", &kv_slot, &token_pos)) {
        return NULL;
    }
    if (token_pos < 0) {
        PyErr_SetString(PyExc_ValueError, "token_pos must be non-negative");
        return NULL;
    }
    KvPageBind *bind = kv_find_page_bind(kv_slot);
    if (bind == NULL) {
        PyErr_Format(PyExc_KeyError, "no page bind for kv_slot %d", kv_slot);
        return NULL;
    }
    int page = token_pos / bind->block_size;
    if (page >= bind->num_pages) {
        PyErr_Format(
            PyExc_ValueError,
            "token_pos %d exceeds bound pages (%d pages, block_size %d)",
            token_pos,
            bind->num_pages,
            bind->block_size);
        return NULL;
    }
    int offset = token_pos % bind->block_size;
    return Py_BuildValue("(iii)", page, bind->block_ids[page], offset);
}

static PyObject *
kv_native_page_bind_stats(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    return Py_BuildValue(
        "{s:i,s:K,s:O,s:O}",
        "active_binds",
        kv_count_active_page_binds(),
        "total_registers",
        g_page_bind_registers,
        "tensor_pages_bound",
        kv_any_tensor_pages_bound() ? Py_True : Py_False,
        "physical_pages_bound",
        kv_any_physical_pages_bound() ? Py_True : Py_False);
}

static PyObject *
kv_native_page_bind_slots(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    PyObject *out = PyList_New(0);
    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (!g_page_binds[i].active) {
            continue;
        }
        PyObject *row = Py_BuildValue(
            "{s:i,s:i,s:i,s:i,s:i,s:i,s:i}",
            "kv_slot",
            g_page_binds[i].kv_slot,
            "num_pages",
            g_page_binds[i].num_pages,
            "block_size",
            g_page_binds[i].block_size,
            "cell_pages_bound",
            g_page_binds[i].cell_pages_bound,
            "tensor_pages_bound",
            g_page_binds[i].tensor_pages_bound_slot,
            "physical_pages_bound",
            g_page_binds[i].physical_pages_bound,
            "physical_pages_mapped",
            g_page_binds[i].physical_pages_mapped);
        if (row == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        if (PyList_Append(out, row) != 0) {
            Py_DECREF(row);
            Py_DECREF(out);
            return NULL;
        }
        Py_DECREF(row);
    }
    return out;
}

static PyObject *
kv_native_page_bind_table(PyObject *Py_UNUSED(self), PyObject *args)
{
    int kv_slot = 0;
    if (!PyArg_ParseTuple(args, "i", &kv_slot)) {
        return NULL;
    }
    KvPageBind *bind = kv_find_page_bind(kv_slot);
    if (bind == NULL || !bind->active || bind->num_pages <= 0) {
        return PyList_New(0);
    }
    PyObject *out = PyList_New(bind->num_pages);
    if (out == NULL) {
        return NULL;
    }
    for (int p = 0; p < bind->num_pages; p++) {
        /*
         * WHY initialise slot to NULL before Py_BuildValue: if Py_BuildValue
         * fails on OOM, PyList_SET_ITEM would have never stolen a reference for
         * indices 0..p-1 (those were set successfully).  Py_DECREF(out) releases
         * those already-set items via the list's tp_dealloc.  Slots p..n-1 must
         * be NULL so tp_dealloc does not try to decrement uninitialised pointers.
         * PyList_New zeroes the ob_item array, so only the error slot is at risk.
         */
        int token_start = p * bind->block_size;
        int token_end = token_start + bind->block_size - 1;
        PyObject *row = Py_BuildValue(
            "{s:i,s:i,s:i,s:i}",
            "page",
            p,
            "block_id",
            bind->block_ids[p],
            "token_start",
            token_start,
            "token_end",
            token_end);
        if (row == NULL) {
            /* list's tp_dealloc will DECREF items 0..p-1 that were SET_ITEM'd */
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(out, p, row);
    }
    return out;
}

#ifdef ZEROLLAMA_KV_DECODE_LOOP
static PyObject *
kv_native_probe_result_dict(const KvTensorProbeResult *probe)
{
    const char *blocker = kv_tensor_blocker_str(probe->blocker_code);
    const char *memory_kind = kv_tensor_memory_kind_str(probe->memory_kind);
    /* Format must have exactly 20 s:i before s:K,s:K — v35 added kv_cache_kv_size and
     * kv_cache_n_stream. A short format treats GPU pointers as C strings → SIGBUS. */
    return Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:K,s:K,s:s,s:s,s:O,s:O}",
        "memory_non_null",
        probe->memory_non_null,
        "can_shift",
        probe->can_shift,
        "seq_pos_min",
        (int)probe->seq_pos_min,
        "seq_pos_max",
        (int)probe->seq_pos_max,
        "llama_token_cells",
        (int)probe->llama_token_cells,
        "pa_pages_registered",
        (int)probe->pa_pages_registered,
        "pa_block_size",
        (int)probe->pa_block_size,
        "pages_fit",
        probe->pages_fit,
        "aligned",
        probe->aligned,
        "cell_pages_bound",
        probe->cell_pages_bound,
        "tensor_pages_bound",
        probe->tensor_pages_bound ? 1 : 0,
        "physical_pages_bound",
        probe->physical_pages_bound ? 1 : 0,
        "physical_pages_mapped",
        (int)probe->physical_pages_mapped,
        "kv_stream",
        (int)probe->kv_stream,
        "memory_kind",
        (int)probe->memory_kind,
        "kv_n_layers",
        (int)probe->kv_n_layers,
        "tensor_layers_verified",
        (int)probe->tensor_layers_verified,
        "kv_v_transposed",
        (int)probe->kv_v_transposed,
        "kv_cache_kv_size",
        (int)probe->kv_cache_kv_size,
        "kv_cache_n_stream",
        (int)probe->kv_cache_n_stream,
        "kv_k_data_layer0",
        probe->kv_k_data_layer0,
        "kv_v_data_layer0",
        probe->kv_v_data_layer0,
        "memory_kind_name",
        memory_kind,
        "blocker",
        blocker,
        "tensor_bind_ready",
        probe->tensor_pages_bound ? Py_True : Py_False,
        "writable_bind_ready",
        probe->physical_pages_bound ? Py_True : Py_False);
}

static PyObject *
kv_native_page_bind_tensor_probe(PyObject *Py_UNUSED(self), PyObject *args)
{
    unsigned PY_LONG_LONG ctx_ptr = 0;
    int seq_id = 0;
    int kv_slot = 0;
    if (!PyArg_ParseTuple(args, "Kii", &ctx_ptr, &seq_id, &kv_slot)) {
        return NULL;
    }
    if (ctx_ptr == 0) {
        PyErr_SetString(PyExc_ValueError, "ctx_ptr must be non-zero");
        return NULL;
    }
    KvTensorProbeResult probe;
    if (kv_tensor_probe_run((void *)(uintptr_t)ctx_ptr, seq_id, kv_slot, &probe) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "tensor probe failed");
        return NULL;
    }
    return kv_native_probe_result_dict(&probe);
}

static PyObject *
kv_native_page_bind_last_tensor_probe(PyObject *Py_UNUSED(self), PyObject *args)
{
    int kv_slot = -1;
    if (!PyArg_ParseTuple(args, "|i", &kv_slot)) {
        return NULL;
    }
    if (kv_slot >= 0) {
        KvTensorProbeResult probe;
        if (kv_tensor_probe_last_get(kv_slot, &probe) != 0) {
            Py_RETURN_NONE;
        }
        return kv_native_probe_result_dict(&probe);
    }
    PyObject *out = PyList_New(0);
    if (out == NULL) {
        return NULL;
    }
    for (int idx = 0; idx < KV_MAX_PAGE_BINDS; idx++) {
        int stored_slot = -1;
        KvTensorProbeResult probe;
        if (kv_tensor_probe_last_get_by_index(idx, &stored_slot, &probe) != 0) {
            continue;
        }
        PyObject *probe_dict = kv_native_probe_result_dict(&probe);
        if (probe_dict == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        PyObject *row = Py_BuildValue("{s:i,s:O}", "kv_slot", stored_slot, "probe", probe_dict);
        Py_DECREF(probe_dict);
        if (row == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        if (PyList_Append(out, row) != 0) {
            Py_DECREF(row);
            Py_DECREF(out);
            return NULL;
        }
        Py_DECREF(row);
    }
    return out;
}

#ifdef LLAMA_KV_EXT_WRITABLE_PAGE_MAP
static PyObject *
kv_native_page_bind_map_page(PyObject *Py_UNUSED(self), PyObject *args)
{
    unsigned PY_LONG_LONG ctx_ptr = 0;
    int seq_id = 0;
    int kv_slot = 0;
    int page_index = 0;
    int kv_layer = 0;
    if (!PyArg_ParseTuple(args, "Kiii|i", &ctx_ptr, &seq_id, &kv_slot, &page_index, &kv_layer)) {
        return NULL;
    }
    if (ctx_ptr == 0) {
        PyErr_SetString(PyExc_ValueError, "ctx_ptr must be non-zero");
        return NULL;
    }
    if (page_index < 0 || kv_layer < 0) {
        PyErr_SetString(PyExc_ValueError, "page_index and kv_layer must be non-negative");
        return NULL;
    }
    KvPageBind *bind = kv_find_page_bind(kv_slot);
    if (bind == NULL || !bind->active) {
        PyErr_Format(PyExc_KeyError, "no page bind for kv_slot %d", kv_slot);
        return NULL;
    }
    if (page_index >= bind->num_pages) {
        PyErr_Format(PyExc_ValueError, "page_index %d out of range", page_index);
        return NULL;
    }

    struct llama_context *lctx = (struct llama_context *)(uintptr_t)ctx_ptr;
    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        PyErr_SetString(PyExc_RuntimeError, "llama_get_memory returned NULL");
        return NULL;
    }

    const llama_pos base = llama_memory_seq_pos_min(mem, (llama_seq_id)seq_id);
    llama_kv_page_map page_map;
    memset(&page_map, 0, sizeof(page_map));
    const int32_t rc = llama_memory_kv_page_map(
        mem,
        (llama_seq_id)seq_id,
        base,
        (uint32_t)page_index,
        (uint32_t)bind->block_size,
        kv_layer,
        &page_map);
    if (rc != LLAMA_KV_EXT_OK || !page_map.ok) {
        PyErr_Format(
            PyExc_RuntimeError,
            "llama_memory_kv_page_map failed rc=%d page=%d",
            (int)rc,
            page_index);
        return NULL;
    }

    return Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:K,s:K,s:K,s:K,s:K,s:K,s:i,s:i}",
        "page",
        page_index,
        "block_id",
        bind->block_ids[page_index],
        "kv_layer",
        kv_layer,
        "pos_start",
        (int)page_map.pos_start,
        "pos_end",
        (int)page_map.pos_end,
        "n_cells",
        (int)page_map.n_cells,
        "cell_idx_first",
        (int)page_map.cell_idx_first,
        "cell_idx_last",
        (int)page_map.cell_idx_last,
        "stream",
        (int)page_map.stream,
        "k_data",
        page_map.k_data,
        "v_data",
        page_map.v_data,
        "k_span_bytes",
        page_map.k_span_bytes,
        "v_span_bytes",
        page_map.v_span_bytes,
        "v_transposed",
        (int)        page_map.v_transposed);
}
#endif /* LLAMA_KV_EXT_WRITABLE_PAGE_MAP */

#ifdef LLAMA_KV_EXT_EXTERNAL_ALIAS
static PyObject *
kv_native_page_bind_alias_validate(PyObject *Py_UNUSED(self), PyObject *args)
{
    unsigned PY_LONG_LONG ctx_ptr = 0;
    int seq_id = 0;
    int kv_slot = 0;
    int page_index = 0;
    unsigned PY_LONG_LONG kv_layer = 0;
    unsigned PY_LONG_LONG ext_k_data = 0;
    unsigned PY_LONG_LONG ext_k_span = 0;
    unsigned PY_LONG_LONG ext_v_data = 0;
    unsigned PY_LONG_LONG ext_v_span = 0;
    if (!PyArg_ParseTuple(
            args,
            "Kiii|KKKKK",
            &ctx_ptr,
            &seq_id,
            &kv_slot,
            &page_index,
            &kv_layer,
            &ext_k_data,
            &ext_k_span,
            &ext_v_data,
            &ext_v_span)) {
        return NULL;
    }
    if (ctx_ptr == 0) {
        PyErr_SetString(PyExc_ValueError, "ctx_ptr must be non-zero");
        return NULL;
    }
    if (page_index < 0 || (long long)kv_layer < 0) {
        PyErr_SetString(PyExc_ValueError, "page_index and kv_layer must be non-negative");
        return NULL;
    }
    KvPageBind *bind = kv_find_page_bind(kv_slot);
    if (bind == NULL || !bind->active) {
        PyErr_Format(PyExc_KeyError, "no page bind for kv_slot %d", kv_slot);
        return NULL;
    }
    if (page_index >= bind->num_pages) {
        PyErr_Format(PyExc_ValueError, "page_index %d out of range", page_index);
        return NULL;
    }

    struct llama_context *lctx = (struct llama_context *)(uintptr_t)ctx_ptr;
    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        PyErr_SetString(PyExc_RuntimeError, "llama_get_memory returned NULL");
        return NULL;
    }

    const llama_pos base = llama_memory_seq_pos_min(mem, (llama_seq_id)seq_id);
    llama_kv_page_alias_plan plan;
    memset(&plan, 0, sizeof(plan));
    const int32_t rc = llama_memory_kv_page_alias_validate(
        mem,
        (llama_seq_id)seq_id,
        base,
        (uint32_t)page_index,
        (uint32_t)bind->block_size,
        (int32_t)kv_layer,
        ext_k_data,
        ext_k_span,
        ext_v_data,
        ext_v_span,
        &plan);
    if (rc != LLAMA_KV_EXT_OK && rc != LLAMA_KV_EXT_NOT_FOUND && rc != LLAMA_KV_EXT_UNSUPPORTED) {
        PyErr_Format(PyExc_RuntimeError, "llama_memory_kv_page_alias_validate failed rc=%d", (int)rc);
        return NULL;
    }

    return Py_BuildValue(
        "{s:O,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:K,s:K,s:K,s:K,s:s}",
        "ok",
        plan.ok ? Py_True : Py_False,
        "alias_ready",
        (int)plan.alias_ready,
        "alias_mode",
        (int)plan.alias_mode,
        "buffer_host",
        (int)plan.buffer_host,
        "k_spans_match",
        (int)plan.k_spans_match,
        "v_spans_match",
        (int)plan.v_spans_match,
        "k_same_pointer",
        (int)plan.k_same_pointer,
        "v_same_pointer",
        (int)plan.v_same_pointer,
        "v_transposed",
        (int)plan.v_transposed,
        "llama_k_data",
        plan.llama_k_data,
        "llama_v_data",
        plan.llama_v_data,
        "k_span_bytes",
        plan.k_span_bytes,
        "v_span_bytes",
        plan.v_span_bytes,
        "blocker",
        plan.blocker);
}
#endif /* LLAMA_KV_EXT_EXTERNAL_ALIAS */
#endif /* ZEROLLAMA_KV_DECODE_LOOP */

#ifdef LLAMA_KV_EXT_EXTERNAL_ALIAS
static PyObject *
kv_native_page_bind_external_alias_probe(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    llama_kv_ext_external_alias_probe probe;
    memset(&probe, 0, sizeof(probe));
    llama_memory_kv_ext_external_alias_probe(&probe);
    return Py_BuildValue(
        "{s:O,s:O,s:s}",
        "external_alias_available",
        probe.available ? Py_True : Py_False,
        "external_alias_validate_api",
        probe.validate_api ? Py_True : Py_False,
        "external_alias_api",
        probe.api_name);
}
#endif /* LLAMA_KV_EXT_EXTERNAL_ALIAS */

static PyObject *
kv_native_page_bind_writable_probe(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
#if defined(LLAMA_KV_EXT_WRITABLE_PAGE_MAP) && defined(ZEROLLAMA_KV_DECODE_LOOP)
    int32_t avail = 0;
    char api_name[64] = "none";
    llama_memory_kv_ext_writable_bind_probe(&avail, api_name, sizeof(api_name));
    return Py_BuildValue(
        "{s:O,s:s,s:s}",
        "writable_bind_available",
        avail ? Py_True : Py_False,
        "writable_bind_api",
        api_name,
        "writable_bind_blocker",
        avail ? "" : "libllama_writable_page_map_not_linked");
#elif defined(LLAMA_KV_EXT_WRITABLE_PAGE_MAP)
    return Py_BuildValue(
        "{s:O,s:s,s:s}",
        "writable_bind_available",
        Py_True,
        "writable_bind_api",
        "llama_memory_kv_page_map",
        "writable_bind_blocker",
        "");
#else
    const char *blocker =
#ifdef ZEROLLAMA_KV_DECODE_LOOP
        "staging_writable_page_map_not_implemented";
#else
        "llama_kv_ext_not_linked";
#endif
    return Py_BuildValue(
        "{s:O,s:s,s:s}",
        "writable_bind_available",
        Py_False,
        "writable_bind_api",
        "none",
        "writable_bind_blocker",
        blocker);
#endif /* LLAMA_KV_EXT_WRITABLE_PAGE_MAP */
}

static PyObject *
kv_native_decode_batch_layout(PyObject *Py_UNUSED(self), PyObject *args)
{
    int seq_id = 0;
    int pos_start = 0;
    int logits_last = 1;
    PyObject *tokens = NULL;
    if (!PyArg_ParseTuple(args, "Oii|i", &tokens, &seq_id, &pos_start, &logits_last)) {
        return NULL;
    }
    PyObject *seq = PySequence_Fast(tokens, "tokens must be a sequence");
    if (seq == NULL) {
        return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "tokens must be non-empty");
        return NULL;
    }
    PyObject *pos_list = PyList_New(n);
    PyObject *seq_list = PyList_New(n);
    PyObject *logits_list = PyList_New(n);
    PyObject *token_list = PyList_New(n);
    if (pos_list == NULL || seq_list == NULL || logits_list == NULL || token_list == NULL) {
        Py_XDECREF(pos_list);
        Py_XDECREF(seq_list);
        Py_XDECREF(logits_list);
        Py_XDECREF(token_list);
        Py_DECREF(seq);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long tok = PyLong_AsLong(item);
        if (tok == -1 && PyErr_Occurred()) {
            Py_DECREF(pos_list);
            Py_DECREF(seq_list);
            Py_DECREF(logits_list);
            Py_DECREF(token_list);
            Py_DECREF(seq);
            return NULL;
        }
        PyObject *tok_obj = PyLong_FromLong(tok);
        PyObject *pos_obj = PyLong_FromLong(pos_start + (int)i);
        PyObject *seq_obj = PyLong_FromLong(seq_id);
        int logit = (logits_last && i == n - 1) ? 1 : (logits_last ? 0 : 1);
        PyObject *log_obj = PyLong_FromLong(logit);
        if (tok_obj == NULL || pos_obj == NULL || seq_obj == NULL || log_obj == NULL) {
            Py_XDECREF(tok_obj);
            Py_XDECREF(pos_obj);
            Py_XDECREF(seq_obj);
            Py_XDECREF(log_obj);
            Py_DECREF(pos_list);
            Py_DECREF(seq_list);
            Py_DECREF(logits_list);
            Py_DECREF(token_list);
            Py_DECREF(seq);
            return NULL;
        }
        PyList_SET_ITEM(token_list, i, tok_obj);
        PyList_SET_ITEM(pos_list, i, pos_obj);
        PyList_SET_ITEM(seq_list, i, seq_obj);
        PyList_SET_ITEM(logits_list, i, log_obj);
    }
    Py_DECREF(seq);
    PyObject *out = Py_BuildValue(
        "{s:O,s:O,s:O,s:O}",
        "token",
        token_list,
        "pos",
        pos_list,
        "seq_id",
        seq_list,
        "logits",
        logits_list);
    Py_DECREF(token_list);
    Py_DECREF(pos_list);
    Py_DECREF(seq_list);
    Py_DECREF(logits_list);
    return out;
}

static PyObject *
kv_native_decode_batch_layout_multi(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *tokens = NULL;
    PyObject *seq_ids = NULL;
    PyObject *positions = NULL;
    if (!PyArg_ParseTuple(args, "OOO", &tokens, &seq_ids, &positions)) {
        return NULL;
    }
    PyObject *tok_seq = PySequence_Fast(tokens, "tokens must be a sequence");
    PyObject *sid_seq = PySequence_Fast(seq_ids, "seq_ids must be a sequence");
    PyObject *pos_seq = PySequence_Fast(positions, "positions must be a sequence");
    if (tok_seq == NULL || sid_seq == NULL || pos_seq == NULL) {
        Py_XDECREF(tok_seq);
        Py_XDECREF(sid_seq);
        Py_XDECREF(pos_seq);
        return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(tok_seq);
    if (n != PySequence_Fast_GET_SIZE(sid_seq) ||
        n != PySequence_Fast_GET_SIZE(pos_seq)) {
        Py_DECREF(tok_seq);
        Py_DECREF(sid_seq);
        Py_DECREF(pos_seq);
        PyErr_SetString(PyExc_ValueError, "tokens, seq_ids, positions length mismatch");
        return NULL;
    }
    if (n <= 0) {
        Py_DECREF(tok_seq);
        Py_DECREF(sid_seq);
        Py_DECREF(pos_seq);
        PyErr_SetString(PyExc_ValueError, "batch must be non-empty");
        return NULL;
    }
    PyObject *token_list = PyList_New(n);
    PyObject *pos_list = PyList_New(n);
    PyObject *seq_list = PyList_New(n);
    PyObject *logits_list = PyList_New(n);
    if (!token_list || !pos_list || !seq_list || !logits_list) {
        Py_XDECREF(token_list);
        Py_XDECREF(pos_list);
        Py_XDECREF(seq_list);
        Py_XDECREF(logits_list);
        Py_DECREF(tok_seq);
        Py_DECREF(sid_seq);
        Py_DECREF(pos_seq);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        long tok = PyLong_AsLong(PySequence_Fast_GET_ITEM(tok_seq, i));
        long sid = PyLong_AsLong(PySequence_Fast_GET_ITEM(sid_seq, i));
        long pos = PyLong_AsLong(PySequence_Fast_GET_ITEM(pos_seq, i));
        if ((tok == -1 || sid == -1 || pos == -1) && PyErr_Occurred()) {
            Py_DECREF(token_list);
            Py_DECREF(pos_list);
            Py_DECREF(seq_list);
            Py_DECREF(logits_list);
            Py_DECREF(tok_seq);
            Py_DECREF(sid_seq);
            Py_DECREF(pos_seq);
            return NULL;
        }
        PyObject *tok_obj = PyLong_FromLong(tok);
        PyObject *pos_obj = PyLong_FromLong(pos);
        PyObject *seq_obj = PyLong_FromLong(sid);
        PyObject *log_obj = PyLong_FromLong(1L);
        if (!tok_obj || !pos_obj || !seq_obj || !log_obj) {
            Py_XDECREF(tok_obj);
            Py_XDECREF(pos_obj);
            Py_XDECREF(seq_obj);
            Py_XDECREF(log_obj);
            Py_DECREF(token_list);
            Py_DECREF(pos_list);
            Py_DECREF(seq_list);
            Py_DECREF(logits_list);
            Py_DECREF(tok_seq);
            Py_DECREF(sid_seq);
            Py_DECREF(pos_seq);
            return NULL;
        }
        PyList_SET_ITEM(token_list, i, tok_obj);
        PyList_SET_ITEM(pos_list, i, pos_obj);
        PyList_SET_ITEM(seq_list, i, seq_obj);
        PyList_SET_ITEM(logits_list, i, log_obj);
    }
    Py_DECREF(tok_seq);
    Py_DECREF(sid_seq);
    Py_DECREF(pos_seq);
    PyObject *out = Py_BuildValue(
        "{s:O,s:O,s:O,s:O}",
        "token",
        token_list,
        "pos",
        pos_list,
        "seq_id",
        seq_list,
        "logits",
        logits_list);
    Py_DECREF(token_list);
    Py_DECREF(pos_list);
    Py_DECREF(seq_list);
    Py_DECREF(logits_list);
    return out;
}

static PyObject *
kv_native_decode_prefill_chunks(PyObject *Py_UNUSED(self), PyObject *args)
{
    int block_size = 0;
    int pos_start = 0;
    PyObject *tokens = NULL;
    if (!PyArg_ParseTuple(args, "Oii", &tokens, &block_size, &pos_start)) {
        return NULL;
    }
    if (block_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "block_size must be positive");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(tokens, "tokens must be a sequence");
    if (seq == NULL) {
        return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "tokens must be non-empty");
        return NULL;
    }
    PyObject *out = PyList_New(0);
    if (out == NULL) {
        Py_DECREF(seq);
        return NULL;
    }
    Py_ssize_t i = 0;
    while (i < n) {
        Py_ssize_t chunk_end = i + block_size;
        if (chunk_end > n) {
            chunk_end = n;
        }
        Py_ssize_t chunk_len = chunk_end - i;
        PyObject *chunk = PyList_New(chunk_len);
        if (chunk == NULL) {
            Py_DECREF(out);
            Py_DECREF(seq);
            return NULL;
        }
        for (Py_ssize_t j = 0; j < chunk_len; j++) {
            PyObject *item = PySequence_Fast_GET_ITEM(seq, i + j);
            long tok = PyLong_AsLong(item);
            if (tok == -1 && PyErr_Occurred()) {
                Py_DECREF(chunk);
                Py_DECREF(out);
                Py_DECREF(seq);
                return NULL;
            }
            PyObject *tok_obj = PyLong_FromLong(tok);
            if (tok_obj == NULL) {
                Py_DECREF(chunk);
                Py_DECREF(out);
                Py_DECREF(seq);
                return NULL;
            }
            PyList_SET_ITEM(chunk, j, tok_obj);
        }
        PyObject *row = Py_BuildValue("(Oi)", chunk, pos_start + (int)i);
        Py_DECREF(chunk);
        if (row == NULL) {
            Py_DECREF(out);
            Py_DECREF(seq);
            return NULL;
        }
        if (PyList_Append(out, row) < 0) {
            Py_DECREF(row);
            Py_DECREF(out);
            Py_DECREF(seq);
            return NULL;
        }
        Py_DECREF(row);
        i = chunk_end;
    }
    Py_DECREF(seq);
    return out;
}

static unsigned long long
kv_load_u64(unsigned long long *p)
{
    return __atomic_load_n(p, __ATOMIC_SEQ_CST);
}

static unsigned long long
kv_add_u64(unsigned long long *p, unsigned long long n)
{
    return __atomic_add_fetch(p, n, __ATOMIC_SEQ_CST);
}

static PyObject *
kv_native_scheduler_tick(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    unsigned long long v = kv_add_u64(&g_scheduler_tick, 1);
    return PyLong_FromUnsignedLongLong(v);
}

static PyObject *
kv_native_decode_step(PyObject *Py_UNUSED(self), PyObject *args)
{
    int n = 1;
    if (!PyArg_ParseTuple(args, "|i", &n)) {
        return NULL;
    }
    if (n > 0) {
        unsigned long long v = kv_add_u64(&g_decode_steps, (unsigned long long)n);
        return PyLong_FromUnsignedLongLong(v);
    }
    return PyLong_FromUnsignedLongLong(kv_load_u64(&g_decode_steps));
}

static PyObject *
kv_native_stats(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    return Py_BuildValue(
        "{s:K,s:K}",
        "scheduler_tick",
        kv_load_u64(&g_scheduler_tick),
        "decode_steps",
        kv_load_u64(&g_decode_steps));
}

#ifdef ZEROLLAMA_KV_DECODE_LOOP
static PyObject *
kv_native_decode_loop_prefill(PyObject *Py_UNUSED(self), PyObject *args)
{
    /*
     * decode_loop_prefill(ctx_ptr, tokens, seq_id, block_size, pos_start) -> int steps
     *
     * WHY separate from ctypes llama_decode: v13 moves the hot forward pass into
     * this extension when ZEROLLAMA_KV_DECODE_LOOP=1 at build time — fewer Python
     * roundtrips per prefill chunk.  v14 releases the GIL during llama_decode.
     * Sampling stays in Python (see run_step).
     *
     * ctx_ptr: Python int holding the llama_context* address (ctypes c_void_p).
     * tokens:  sequence of ints (prompt token ids).
     * seq_id:  llama sequence / slot id (same as kv_slot in in-process mode).
     * block_size: PA page size for chunking (0 = single batch, no page split).
     * pos_start: first llama write position (default 0).
     */
    unsigned PY_LONG_LONG ctx_ptr = 0;
    PyObject *tokens_obj = NULL;
    int seq_id = 0;
    int block_size = 0;
    int pos_start = 0;
    if (!PyArg_ParseTuple(args, "KOii|i", &ctx_ptr, &tokens_obj, &seq_id, &block_size,
                          &pos_start)) {
        return NULL;
    }
    PyObject *seq = PySequence_Fast(tokens_obj, "tokens must be a sequence");
    if (!seq) return NULL;

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "tokens must be non-empty");
        return NULL;
    }
    int32_t *toks = (int32_t *)PyMem_Malloc((size_t)n * sizeof(int32_t));
    if (!toks) { Py_DECREF(seq); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long v = PyLong_AsLong(item);
        if (v == -1 && PyErr_Occurred()) {
            PyMem_Free(toks); Py_DECREF(seq);
            return NULL;
        }
        toks[i] = (int32_t)v;
    }
    Py_DECREF(seq);

    int32_t steps = 0;
    void *ctx = (void *)(uintptr_t)ctx_ptr;
    int rc;
    /* WHY release GIL: llama_decode is the hot path; training/inference share one
     * embedded interpreter under ZEROLLAMA_RUNTIME_SHARED_PYTHON=1. */
    Py_BEGIN_ALLOW_THREADS
    rc = kv_decode_loop_run_prefill(ctx, toks, (int32_t)n, (int32_t)seq_id,
                                    (int32_t)block_size, (int32_t)pos_start, &steps);
    Py_END_ALLOW_THREADS
    PyMem_Free(toks);
    if (rc == -3) {
        /* v31: abort flag was set between chunks (PrefillAbortedError in Python). */
        PyErr_SetString(
            PyExc_ValueError,
            "KV prefill aborted: cancel flag set between chunks");
        return NULL;
    }
    if (rc == -2) {
        PyErr_SetString(
            PyExc_ValueError,
            "KV page bind: token position out of range for kv_slot");
        return NULL;
    }
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "llama_decode failed during prefill");
        return NULL;
    }
    /* WHY after Py_END_ALLOW_THREADS: kv_decode_loop_post_prefill_probe writes
     * cell_pages_bound / tensor_pages_bound_slot into the shared bind registry.
     * Those fields are also read by page_bind_slots() from Python threads holding
     * the GIL.  Running the probe here (GIL held) prevents that data race. */
    kv_decode_loop_post_prefill_probe(ctx, (int32_t)seq_id);
    return PyLong_FromLong(steps);
}

static PyObject *
kv_native_decode_loop_step(PyObject *Py_UNUSED(self), PyObject *args)
{
    /*
     * decode_loop_step(ctx_ptr, token, seq_id, current_pos[, smpl_ptr]) -> steps
     *   or (steps, sampled_token) when smpl_ptr is given (v15).
     *
     * WHY single-token batch with logits_last=1: matches ctypes decode loop — one
     * token fed back at current_pos; logits for sampling come from this step.
     * v14: GIL released during llama_decode.  v15: optional llama_sampler_sample
     * in the same GIL-released block when smpl_ptr is non-zero.
     */
    unsigned PY_LONG_LONG ctx_ptr = 0;
    int token = 0;
    int seq_id = 0;
    int current_pos = 0;
    unsigned PY_LONG_LONG smpl_ptr = 0;
    if (!PyArg_ParseTuple(args, "Kiii|K", &ctx_ptr, &token, &seq_id, &current_pos,
                          &smpl_ptr)) {
        return NULL;
    }
    int32_t steps = 0;
    int32_t sampled = 0;
    void *ctx = (void *)(uintptr_t)ctx_ptr;
    void *smpl = smpl_ptr ? (void *)(uintptr_t)smpl_ptr : NULL;
    int32_t *sampled_out = smpl ? &sampled : NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = kv_decode_loop_run_step(ctx, (int32_t)token, (int32_t)seq_id,
                                 (int32_t)current_pos, smpl, sampled_out, &steps);
    Py_END_ALLOW_THREADS
    if (rc == -2) {
        PyErr_SetString(
            PyExc_ValueError,
            "KV page bind: token position out of range for kv_slot");
        return NULL;
    }
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "llama_decode failed during decode step");
        return NULL;
    }
    if (smpl) {
        return Py_BuildValue("(ii)", steps, sampled);
    }
    return PyLong_FromLong(steps);
}

static PyObject *
kv_native_decode_loop_batch_step(PyObject *Py_UNUSED(self), PyObject *args)
{
    /*
     * decode_loop_batch_step(ctx_ptr, tokens, seq_ids, positions[, smpl_ptrs])
     *   -> steps  or  (steps, [sampled...]) when smpl_ptrs given (v26/v30).
     *
     * smpl_ptrs: omitted | int (legacy shared smpl) | list[int] per-row (v30).
     *
     * WHY one llama_decode: continuous batch — N single-token rows in one batch.
     */
    unsigned PY_LONG_LONG ctx_ptr = 0;
    PyObject *tokens = NULL;
    PyObject *seq_ids = NULL;
    PyObject *positions = NULL;
    PyObject *smpl_arg = NULL;
    if (!PyArg_ParseTuple(args, "KOOO|O", &ctx_ptr, &tokens, &seq_ids, &positions,
                          &smpl_arg)) {
        return NULL;
    }
    PyObject *tok_seq = PySequence_Fast(tokens, "tokens must be a sequence");
    PyObject *sid_seq = PySequence_Fast(seq_ids, "seq_ids must be a sequence");
    PyObject *pos_seq = PySequence_Fast(positions, "positions must be a sequence");
    if (tok_seq == NULL || sid_seq == NULL || pos_seq == NULL) {
        Py_XDECREF(tok_seq);
        Py_XDECREF(sid_seq);
        Py_XDECREF(pos_seq);
        return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(tok_seq);
    if (n != PySequence_Fast_GET_SIZE(sid_seq) ||
        n != PySequence_Fast_GET_SIZE(pos_seq)) {
        Py_DECREF(tok_seq);
        Py_DECREF(sid_seq);
        Py_DECREF(pos_seq);
        PyErr_SetString(PyExc_ValueError, "tokens, seq_ids, positions length mismatch");
        return NULL;
    }
    if (n <= 0) {
        Py_DECREF(tok_seq);
        Py_DECREF(sid_seq);
        Py_DECREF(pos_seq);
        return PyLong_FromLong(0);
    }
    if (n > KV_DECODE_LOOP_BATCH_MAX) {
        Py_DECREF(tok_seq);
        Py_DECREF(sid_seq);
        Py_DECREF(pos_seq);
        PyErr_Format(PyExc_ValueError, "batch too large (%zd > %d)",
                     n, KV_DECODE_LOOP_BATCH_MAX);
        return NULL;
    }

    int32_t *toks = (int32_t *)PyMem_Malloc((size_t)n * sizeof(int32_t));
    int32_t *sids = (int32_t *)PyMem_Malloc((size_t)n * sizeof(int32_t));
    int32_t *posv = (int32_t *)PyMem_Malloc((size_t)n * sizeof(int32_t));
    if (!toks || !sids || !posv) {
        PyMem_Free(toks);
        PyMem_Free(sids);
        PyMem_Free(posv);
        Py_DECREF(tok_seq);
        Py_DECREF(sid_seq);
        Py_DECREF(pos_seq);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        long t = PyLong_AsLong(PySequence_Fast_GET_ITEM(tok_seq, i));
        long s = PyLong_AsLong(PySequence_Fast_GET_ITEM(sid_seq, i));
        long p = PyLong_AsLong(PySequence_Fast_GET_ITEM(pos_seq, i));
        if ((t == -1 || s == -1 || p == -1) && PyErr_Occurred()) {
            PyMem_Free(toks);
            PyMem_Free(sids);
            PyMem_Free(posv);
            Py_DECREF(tok_seq);
            Py_DECREF(sid_seq);
            Py_DECREF(pos_seq);
            return NULL;
        }
        toks[i] = (int32_t)t;
        sids[i] = (int32_t)s;
        posv[i] = (int32_t)p;
    }
    Py_DECREF(tok_seq);
    Py_DECREF(sid_seq);
    Py_DECREF(pos_seq);

    void **smpl_row = NULL;
    int want_sample = 0;
    if (smpl_arg != NULL && smpl_arg != Py_None) {
        smpl_row = (void **)PyMem_Malloc((size_t)n * sizeof(void *));
        if (!smpl_row) {
            PyMem_Free(toks);
            PyMem_Free(sids);
            PyMem_Free(posv);
            return PyErr_NoMemory();
        }
        if (PyLong_Check(smpl_arg)) {
            /* Legacy: one shared sampler for all rows (accept-state bleed if n>1). */
            void *shared = (void *)(uintptr_t)PyLong_AsUnsignedLongLong(smpl_arg);
            if (PyErr_Occurred()) {
                PyMem_Free(smpl_row);
                PyMem_Free(toks);
                PyMem_Free(sids);
                PyMem_Free(posv);
                return NULL;
            }
            for (Py_ssize_t i = 0; i < n; i++) {
                smpl_row[i] = shared;
            }
            want_sample = shared != NULL;
        } else {
            PyObject *smpl_seq = PySequence_Fast(
                smpl_arg, "smpl_ptrs must be an int or sequence of ints");
            if (smpl_seq == NULL) {
                PyMem_Free(smpl_row);
                PyMem_Free(toks);
                PyMem_Free(sids);
                PyMem_Free(posv);
                return NULL;
            }
            if (PySequence_Fast_GET_SIZE(smpl_seq) != n) {
                Py_DECREF(smpl_seq);
                PyMem_Free(smpl_row);
                PyMem_Free(toks);
                PyMem_Free(sids);
                PyMem_Free(posv);
                PyErr_SetString(PyExc_ValueError,
                                "smpl_ptrs length must match tokens");
                return NULL;
            }
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject *item = PySequence_Fast_GET_ITEM(smpl_seq, i);
                if (!PyLong_Check(item)) {
                    Py_DECREF(smpl_seq);
                    PyMem_Free(smpl_row);
                    PyMem_Free(toks);
                    PyMem_Free(sids);
                    PyMem_Free(posv);
                    PyErr_SetString(PyExc_TypeError,
                                    "smpl_ptrs entries must be integers");
                    return NULL;
                }
                smpl_row[i] = (void *)(uintptr_t)PyLong_AsUnsignedLongLong(item);
                if (PyErr_Occurred()) {
                    Py_DECREF(smpl_seq);
                    PyMem_Free(smpl_row);
                    PyMem_Free(toks);
                    PyMem_Free(sids);
                    PyMem_Free(posv);
                    return NULL;
                }
                if (smpl_row[i] != NULL) {
                    want_sample = 1;
                }
            }
            Py_DECREF(smpl_seq);
        }
    }

    int32_t steps = 0;
    int32_t *sampled = NULL;
    void *ctx = (void *)(uintptr_t)ctx_ptr;
    if (want_sample) {
        sampled = (int32_t *)PyMem_Malloc((size_t)n * sizeof(int32_t));
        if (!sampled) {
            PyMem_Free(smpl_row);
            PyMem_Free(toks);
            PyMem_Free(sids);
            PyMem_Free(posv);
            return PyErr_NoMemory();
        }
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = kv_decode_loop_run_batch_step(ctx, toks, sids, posv, (int32_t)n,
                                       (const void *const *)smpl_row,
                                       sampled, &steps);
    Py_END_ALLOW_THREADS
    PyMem_Free(smpl_row);
    PyMem_Free(toks);
    PyMem_Free(sids);
    PyMem_Free(posv);
    if (rc == -2) {
        PyMem_Free(sampled);
        PyErr_SetString(
            PyExc_ValueError,
            "KV page bind: token position out of range for kv_slot");
        return NULL;
    }
    if (rc != 0) {
        PyMem_Free(sampled);
        PyErr_SetString(PyExc_RuntimeError, "llama_decode failed during batch step");
        return NULL;
    }
    if (sampled) {
        PyObject *out_list = PyList_New(n);
        if (!out_list) {
            PyMem_Free(sampled);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *v = PyLong_FromLong((long)sampled[i]);
            if (!v) {
                Py_DECREF(out_list);
                PyMem_Free(sampled);
                return NULL;
            }
            PyList_SET_ITEM(out_list, i, v);
        }
        PyMem_Free(sampled);
        return Py_BuildValue("(iO)", steps, out_list);
    }
    return PyLong_FromLong(steps);
}

static PyObject *
kv_native_decode_loop_sample(PyObject *Py_UNUSED(self), PyObject *args)
{
    /*
     * decode_loop_sample(smpl_ptr, ctx_ptr) -> int token
     *
     * WHY separate from step: first token after prefill is sampled without a
     * preceding single-token decode batch (prefill uses logits_last=0).
     */
    unsigned PY_LONG_LONG smpl_ptr = 0;
    unsigned PY_LONG_LONG ctx_ptr = 0;
    if (!PyArg_ParseTuple(args, "KK", &smpl_ptr, &ctx_ptr)) {
        return NULL;
    }
    int32_t tok;
    void *smpl = (void *)(uintptr_t)smpl_ptr;
    void *ctx = (void *)(uintptr_t)ctx_ptr;
    Py_BEGIN_ALLOW_THREADS
    tok = kv_decode_loop_sample(smpl, ctx);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong((long)tok);
}

static PyObject *
kv_native_invalidate_cuda_graphs(PyObject *Py_UNUSED(self), PyObject *args)
{
    unsigned PY_LONG_LONG ctx_ptr;
    if (!PyArg_ParseTuple(args, "K", &ctx_ptr)) {
        return NULL;
    }
    void *ctx = (void *)(uintptr_t)ctx_ptr;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = kv_decode_loop_invalidate_cuda_graphs(ctx);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("{s:i,s:i}", "backends_cleared", rc, "ok", rc >= 0 ? 1 : 0);
}

/*
 * v31: abort flag bindings — called without GIL released because the flag
 * write is atomic and the Python caller is signalling from a different thread.
 */
static PyObject *
kv_native_decode_loop_abort_set(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    kv_decode_loop_abort_set();
    Py_RETURN_NONE;
}

static PyObject *
kv_native_decode_loop_abort_clear(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    kv_decode_loop_abort_clear();
    Py_RETURN_NONE;
}
#endif /* ZEROLLAMA_KV_DECODE_LOOP */

static PyObject *
kv_native_decode_loop_status(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
#ifdef ZEROLLAMA_KV_DECODE_LOOP
    return Py_BuildValue(
        "{s:i,s:s,s:s,s:K,s:i,s:i,s:i}",
        "available",
        1,
        "reason",
        "",
        "link",
        "native",
        "llama_max_devices",
        (unsigned PY_LONG_LONG)kv_decode_loop_llama_max_devices(),
        "gil_released",
        1,
        "sampling_in_c",
        1,
        "batch_decode_in_c",
        1);
#else
    return Py_BuildValue(
        "{s:i,s:s,s:s}",
        "available",
        0,
        "reason",
        "libllama not linked; build with ZEROLLAMA_KV_DECODE_LOOP=1 and LLAMA_CPP_LIB",
        "link",
        "ctypes");
#endif
}

static PyMethodDef kv_module_methods[] = {
    {"scheduler_tick", kv_native_scheduler_tick, METH_NOARGS,
     "Increment native scheduler admission tick (Phase 15)"},
    {"decode_step", kv_native_decode_step, METH_VARARGS,
     "Increment native decode step counter; optional arg steps (default 1)"},
    {"kv_stats", kv_native_stats, METH_NOARGS,
     "Read scheduler_tick and decode_steps counters (no increment)"},
    {"page_bind_set", kv_native_page_bind_set, METH_VARARGS,
     "Register PA block_ids page table for kv_slot (seq-position bind)"},
    {"page_bind_clear", kv_native_page_bind_clear, METH_VARARGS,
     "Clear page bind for kv_slot"},
    {"page_bind_resolve", kv_native_page_bind_resolve, METH_VARARGS,
     "Map token_pos to (page, block_id, offset) for kv_slot"},
    {"page_bind_stats", kv_native_page_bind_stats, METH_NOARGS,
     "Page bind registry counters"},
    {"page_bind_slots", kv_native_page_bind_slots, METH_NOARGS,
     "Active page bind rows (kv_slot, cell_pages_bound, tensor_pages_bound)"},
    {"page_bind_table", kv_native_page_bind_table, METH_VARARGS,
     "Export PA page table rows for kv_slot (page, block_id, token range)"},
    {"page_bind_writable_probe", kv_native_page_bind_writable_probe, METH_NOARGS,
     "Probe whether writable PA→tensor page bind API is linked (Phase 15 v32b)"},
#ifdef LLAMA_KV_EXT_EXTERNAL_ALIAS
    {"page_bind_external_alias_probe", kv_native_page_bind_external_alias_probe, METH_NOARGS,
     "Static probe: external buffer alias validate API linked (Phase 15 v47)"},
#endif
#ifdef ZEROLLAMA_KV_DECODE_LOOP
    {"page_bind_tensor_probe", kv_native_page_bind_tensor_probe, METH_VARARGS,
     "Probe llama memory vs PA page table (ctx_ptr, seq_id, kv_slot)"},
    {"page_bind_last_tensor_probe", kv_native_page_bind_last_tensor_probe, METH_VARARGS,
     "Last successful tensor probe for kv_slot (optional); all slots when omitted"},
#ifdef LLAMA_KV_EXT_WRITABLE_PAGE_MAP
    {"page_bind_map_page", kv_native_page_bind_map_page, METH_VARARGS,
     "Resolve writable K/V spans for one PA page (ctx_ptr, seq_id, kv_slot, page_index[, kv_layer=0])"},
#endif
#ifdef LLAMA_KV_EXT_EXTERNAL_ALIAS
    {"page_bind_alias_validate", kv_native_page_bind_alias_validate, METH_VARARGS,
     "Validate external K/V ptrs vs page_map (ctx_ptr, seq_id, kv_slot, page_index[, kv_layer, ext_k, ext_k_span, ext_v, ext_v_span])"},
#endif
#endif
    {"decode_batch_layout", kv_native_decode_batch_layout, METH_VARARGS,
     "Build llama_batch field lists in C (token, pos, seq_id, logits)"},
    {"decode_batch_layout_multi", kv_native_decode_batch_layout_multi, METH_VARARGS,
     "Build multi-seq continuous batch layout (tokens, seq_ids, positions)"},
    {"decode_prefill_chunks", kv_native_decode_prefill_chunks, METH_VARARGS,
     "Split prompt tokens into page-aligned decode chunks"},
    {"decode_loop_status", kv_native_decode_loop_status, METH_NOARGS,
     "Whether libllama-linked decode loop is available in this build"},
#ifdef ZEROLLAMA_KV_DECODE_LOOP
    /* WHY #ifdef: default CI builds compile kv_decode_loop.c as no-op — no llama.h */
    {"decode_loop_prefill", kv_native_decode_loop_prefill, METH_VARARGS,
     "Run prefill batches via libllama (ctx_ptr, tokens, seq_id, block_size) -> steps"},
    {"decode_loop_step", kv_native_decode_loop_step, METH_VARARGS,
     "Run one decode step via libllama (ctx_ptr, token, seq_id, current_pos) -> steps"},
    {"decode_loop_batch_step", kv_native_decode_loop_batch_step, METH_VARARGS,
     "Run continuous batch decode (ctx_ptr, tokens, seq_ids, positions) -> steps"},
    {"decode_loop_sample", kv_native_decode_loop_sample, METH_VARARGS,
     "Sample token via libllama (smpl_ptr, ctx_ptr) -> token"},
    {"invalidate_cuda_graphs", kv_native_invalidate_cuda_graphs, METH_VARARGS,
     "Clear ggml CUDA graph cache for ctx_ptr (WHY: L3 slot clear + stale graph safety)"},
    {"decode_loop_abort_set", kv_native_decode_loop_abort_set, METH_NOARGS,
     "Signal prefill abort (v31): set process-global atomic flag; checked between chunks"},
    {"decode_loop_abort_clear", kv_native_decode_loop_abort_clear, METH_NOARGS,
     "Clear prefill abort flag before next run_prefill call"},
#endif
    {NULL},
};

static PyMethodDef kv_pool_methods[] = {
    {"blocks_for_tokens", (PyCFunction)kv_pool_blocks_for_tokens, METH_VARARGS, NULL},
    {"can_allocate", (PyCFunction)kv_pool_can_allocate, METH_VARARGS, NULL},
    {"allocate", (PyCFunction)kv_pool_allocate, METH_VARARGS, NULL},
    {"free", (PyCFunction)kv_pool_free, METH_VARARGS, NULL},
    {"reset", (PyCFunction)kv_pool_reset, METH_NOARGS, NULL},
    {NULL},
};

static PyTypeObject KvBlockPoolType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "runtime.kv._kv_native.BlockPool",
    .tp_basicsize = sizeof(KvBlockPool),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)kv_pool_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Native PagedAttention block pool (Phase 15)",
    .tp_methods = kv_pool_methods,
    .tp_getset = kv_pool_getset,
    .tp_init = (initproc)kv_pool_init,
    .tp_new = PyType_GenericNew,
};

static struct PyModuleDef kv_native_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "runtime.kv._kv_native",
    .m_doc = "Phase 15 native KV allocator, page bind, decode batch layout",
    .m_size = -1,
    .m_methods = kv_module_methods,
};

PyMODINIT_FUNC
PyInit__kv_native(void)
{
    PyObject *m = PyModule_Create(&kv_native_module);
    if (m == NULL) {
        return NULL;
    }
    if (PyType_Ready(&KvBlockPoolType) < 0) {
        return NULL;
    }
    BlockPoolError = PyErr_NewException(
        "runtime.kv._kv_native.BlockPoolError", PyExc_Exception, NULL);
    if (BlockPoolError == NULL) {
        return NULL;
    }
    Py_INCREF(BlockPoolError);
    PyModule_AddObject(m, "BlockPoolError", BlockPoolError);
    Py_INCREF(&KvBlockPoolType);
    PyModule_AddObject(m, "BlockPool", (PyObject *)&KvBlockPoolType);
    return m;
}
