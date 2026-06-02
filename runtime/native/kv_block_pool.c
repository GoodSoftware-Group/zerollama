/*
 * Phase 15: native KV block allocator (CPU bookkeeping only).
 * Mirrors runtime.kv.block_pool.BlockPool; GPU backing comes later.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static PyMethodDef kv_module_methods[] = {
    {"scheduler_tick", kv_native_scheduler_tick, METH_NOARGS,
     "Increment native scheduler admission tick (Phase 15)"},
    {"decode_step", kv_native_decode_step, METH_VARARGS,
     "Increment native decode step counter; optional arg steps (default 1)"},
    {"kv_stats", kv_native_stats, METH_NOARGS,
     "Read scheduler_tick and decode_steps counters (no increment)"},
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
    .m_doc = "Phase 15 native KV allocator, scheduler tick, decode step hook",
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
