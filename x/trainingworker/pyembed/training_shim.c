/*
 * Embedded CPython: load training.py in-process for GPU training (no subprocess / gRPC).
 *
 * Why embed: PyTorch + Transformers + PEFT are Python-native; a second process would need
 * duplicate IPC, grpcio, and socket choreography. Go owns public ports; this file owns
 * Py_Initialize, GIL transitions, and a tiny native module (fire_oom) so CUDA OOM can
 * synchronously ask Go to evict inference VRAM before Python retries load_model.
 *
 * Why no Py_Finalize on success or most failure paths: torch and the Go runtime do not support
 * tearing down a half-used interpreter safely; fail_early sets g_init_aborted instead.
 *
 * Why PyGILState_* for call-ins from Go: after PyEval_SaveThread at init-end, any entry from
 * C/Go must acquire the GIL explicitly; the discarded PyThreadState* is intentional (see
 * comment at SaveThread call site).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "training_shim.h"

#define embedded_prepare_pytorch_ld_path_ex ollama_training_prepare_pytorch_ld_path_ex
#include "../../pyembed_common/embedded_pytorch_env.c"
#undef embedded_prepare_pytorch_ld_path_ex

/* Implemented in Go (cgo export). */
extern void go_training_oom_hook(const char *job_id, const char *message);

static void oom_trampoline(const char *job_id, const char *message) {
	go_training_oom_hook(job_id, message);
}

static PyObject *py_fire_oom(PyObject *self, PyObject *args) {
	(void)self;
	const char *job_id = "";
	const char *msg = "";
	if (!PyArg_ParseTuple(args, "zz", &job_id, &msg))
		return NULL;
	Py_BEGIN_ALLOW_THREADS
	oom_trampoline(job_id ? job_id : "", msg ? msg : "");
	Py_END_ALLOW_THREADS
	Py_RETURN_NONE;
}

static PyMethodDef ollama_native_methods[] = {
	{"fire_oom", py_fire_oom, METH_VARARGS, "Notify Go of CUDA OOM (job_id, message)"},
	{NULL, NULL, 0, NULL},
};

static struct PyModuleDef ollama_native_module = {
	PyModuleDef_HEAD_INIT,
	"ollama_training_native",
	NULL,
	-1,
	ollama_native_methods,
	NULL,
	NULL,
	NULL,
	NULL,
};

PyMODINIT_FUNC PyInit_ollama_training_native(void) {
	return PyModule_Create(&ollama_native_module);
}

void training_preinit_native_module(void) {
	if (PyImport_AppendInittab("ollama_training_native", PyInit_ollama_training_native) != 0)
		fprintf(stderr, "ollama: PyImport_AppendInittab(ollama_training_native) failed\n");
}

static void set_err(char **err_out, const char *msg) {
	if (!err_out)
		return;
	if (*err_out) {
		free(*err_out);
		*err_out = NULL;
	}
	if (!msg)
		return;
	*err_out = strdup(msg);
}

static char *py_unicode_to_utf8(PyObject *u) {
	if (!u)
		return NULL;
	PyObject *b = PyUnicode_AsUTF8String(u);
	if (!b)
		return NULL;
	char *out = strdup(PyBytes_AsString(b));
	Py_DECREF(b);
	return out;
}

void training_free(char *s) {
	if (s)
		free(s);
}

static int g_initialized;
static int g_init_aborted;

int training_is_initialized(void) {
	return g_initialized;
}

int training_init_aborted(void) {
	return g_init_aborted;
}

static int append_sys_path_head(const char *dir, char **err_out) {
	PyObject *sys_path = PySys_GetObject("path");
	if (!sys_path || !PyList_Check(sys_path)) {
		set_err(err_out, "sys.path not a list");
		return -1;
	}
	PyObject *p = PyUnicode_FromString(dir);
	if (!p)
		return -1;
	if (PyList_Insert(sys_path, 0, p) != 0) {
		Py_DECREF(p);
		set_err(err_out, "PyList_Insert(sys.path) failed");
		return -1;
	}
	Py_DECREF(p);
	return 0;
}

static int append_sys_path_tail(const char *dir, char **err_out) {
	PyObject *sys_path = PySys_GetObject("path");
	if (!sys_path || !PyList_Check(sys_path)) {
		set_err(err_out, "sys.path not a list");
		return -1;
	}
	PyObject *p = PyUnicode_FromString(dir);
	if (!p)
		return -1;
	if (PyList_Append(sys_path, p) != 0) {
		Py_DECREF(p);
		set_err(err_out, "PyList_Append(sys.path) failed");
		return -1;
	}
	Py_DECREF(p);
	return 0;
}

static int embedded_python_version(int *major, int *minor) {
	const char *ver = Py_GetVersion();
	if (!ver || sscanf(ver, "%d.%d", major, minor) != 2)
		return -1;
	return 0;
}

static int append_pythonpath_from_env(int py_major, int py_minor, char **err_out) {
	const char *pp = getenv("PYTHONPATH");
	if (!pp || !*pp)
		return 0;

	char *copy = strdup(pp);
	if (!copy) {
		set_err(err_out, "OOM copying PYTHONPATH");
		return -1;
	}

#ifdef _WIN32
	const char *delim = ";";
#else
	const char *delim = ":";
#endif

	for (char *entry = strtok(copy, delim); entry; entry = strtok(NULL, delim)) {
		if (!*entry)
			continue;
		if (!path_matches_python_version(entry, py_major, py_minor))
			continue;
		if (append_sys_path_head(entry, err_out) != 0) {
			free(copy);
			return -1;
		}
	}
	free(copy);
	return 0;
}

static int check_embedded_python_version(char **err_out) {
	const char *ver = Py_GetVersion();
	int major = 0, minor = 0;
	if (sscanf(ver, "%d.%d", &major, &minor) != 2)
		return 0;
	if (major < 3 || (major == 3 && minor < 10)) {
		set_err(err_out,
			"embedded Python 3.10+ required for training (got 3.x from libpython); "
			"rebuild with ./scripts/build_zerollama_mac.sh after .venv-training exists "
			"(see MAC_SETUP_TRAINING=1 ./scripts/mac_setup.sh)");
		return -1;
	}
	return 0;
}

int training_init(const char *repo_root, const char *bootstrap_py, char **err_out) {
	if (g_initialized) {
		set_err(err_out, "training_init already called");
		return -1;
	}
	if (g_init_aborted) {
		set_err(err_out, "training_init previously failed after interpreter start; restart the process");
		return -1;
	}
	if (Py_IsInitialized()) {
		set_err(err_out, "embedded Python cannot be re-initialized (e.g. after shutdown); restart the process");
		return -1;
	}
	if (!repo_root || !bootstrap_py) {
		set_err(err_out, "training_init: null repo_root or bootstrap_py");
		return -1;
	}

	{
		int py_major = 3, py_minor = 10;
		if (embedded_python_version(&py_major, &py_minor) != 0) {
			set_err(err_out, "embedded Python version unavailable");
			return -1;
		}
		ollama_training_prepare_pytorch_ld_path_ex(repo_root, py_major, py_minor);
	}

	Py_Initialize();
	if (!Py_IsInitialized()) {
		set_err(err_out, "Py_Initialize failed");
		return -1;
	}

	if (check_embedded_python_version(err_out) != 0)
		goto fail_early;

	{
		int py_major = 3, py_minor = 10;
		char site[4096];

		if (embedded_python_version(&py_major, &py_minor) != 0) {
			set_err(err_out, "embedded Python version unavailable");
			goto fail_early;
		}
		if (!resolve_training_site_packages(repo_root, py_major, py_minor, site, sizeof(site))) {
			char msg[640];
			/* WHY .venv-training in the message: canonical venv path; ABI must match embedded libpython. */
			snprintf(msg, sizeof(msg),
				"embedded Python %d.%d requires .venv-training/lib/python%d.%d/site-packages; "
				"recreate with TRAINING_UV_PYTHON_VER=%d.%d ./scripts/training_uv_venv.sh --verify "
				"(see docs/gpu-training.md)",
				py_major, py_minor, py_major, py_minor, py_major, py_minor);
			set_err(err_out, msg);
			goto fail_early;
		}
		if (append_sys_path_head(site, err_out) != 0)
			goto fail_early;
		if (append_pythonpath_from_env(py_major, py_minor, err_out) != 0)
			goto fail_early;
	}

	if (append_sys_path_tail(repo_root, err_out) != 0)
		goto fail_early;

	PyObject *bs = Py_CompileString(bootstrap_py, "<ollama_training_bootstrap>", Py_file_input);
	if (!bs) {
		PyErr_Print();
		set_err(err_out, "compile bootstrap failed");
		goto fail_early;
	}
	PyObject *main_mod = PyImport_AddModule("__main__");
	if (!main_mod) {
		Py_DECREF(bs);
		set_err(err_out, "PyImport_AddModule __main__ failed");
		goto fail_early;
	}
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *r = PyEval_EvalCode(bs, main_dict, main_dict);
	Py_DECREF(bs);
	if (!r) {
		PyErr_Print();
		set_err(err_out, "eval bootstrap failed");
		goto fail_early;
	}
	Py_DECREF(r);

	PyObject *init_fn = PyDict_GetItemString(main_dict, "init_ollama_training");
	if (!init_fn || !PyCallable_Check(init_fn)) {
		set_err(err_out, "bootstrap missing init_ollama_training()");
		goto fail_early;
	}
	PyObject *args = PyTuple_New(0);
	if (!args) {
		set_err(err_out, "OOM tuple");
		goto fail_early;
	}
	PyObject *res = PyObject_CallObject(init_fn, args);
	Py_DECREF(args);
	if (!res) {
		PyErr_Print();
		set_err(err_out, "init_ollama_training() failed");
		goto fail_early;
	}
	Py_DECREF(res);

	/* Intentionally discard PyThreadState*: do not pair with PyEval_RestoreThread at shutdown
	 * (unsafe with torch); PyGILState_* is used for all later call-ins from Go/C. */
	(void)PyEval_SaveThread();
	g_initialized = 1;
	return 0;

fail_early:
	/* Never Py_Finalize here: torch may be imported; also unsafe with Go runtime. */
	if (Py_IsInitialized())
		g_init_aborted = 1;
	return -1;
}

void training_ack_vram_headroom(const char *job_id) {
	if (!g_initialized || !job_id)
		return;
	PyGILState_STATE g = PyGILState_Ensure();
	PyObject *main_mod = PyImport_AddModule("__main__");
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *fn = PyDict_GetItemString(main_dict, "ack_vram_headroom");
	if (fn && PyCallable_Check(fn)) {
		PyObject *arg = PyUnicode_FromString(job_id);
		if (arg) {
			PyObject *t = PyTuple_Pack(1, arg);
			if (t) {
				PyObject_CallObject(fn, t);
				Py_DECREF(t);
			}
			Py_DECREF(arg);
		}
		if (PyErr_Occurred())
			PyErr_Clear();
	} else {
		fprintf(stderr,
			"ollama: training_ack_vram_headroom: __main__.ack_vram_headroom missing or not callable\n");
	}
	PyGILState_Release(g);
}

static char *call_eval_json(const char *expr, const char *err_label, char **err_out) {
	if (!g_initialized) {
		set_err(err_out, "training not initialized");
		return NULL;
	}
	PyGILState_STATE gil = PyGILState_Ensure();
	PyObject *main_mod = PyImport_AddModule("__main__");
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *r = PyRun_String(expr, Py_eval_input, main_dict, main_dict);
	char *out = NULL;
	if (!r) {
		PyErr_Print();
		set_err(err_out, err_label ? err_label : "eval failed");
	} else {
		out = py_unicode_to_utf8(r);
		Py_DECREF(r);
	}
	PyGILState_Release(gil);
	return out;
}

char *training_health(char **err_out) {
	return call_eval_json("__import__('json').dumps(_training_shim_api.health())", "health failed", err_out);
}

char *training_submit_job(const char *kind, const char *payload_json, char **err_out) {
	if (!g_initialized) {
		set_err(err_out, "training not initialized");
		return NULL;
	}
	if (!kind)
		kind = "train";
	if (!payload_json)
		payload_json = "{}";
	PyGILState_STATE gil = PyGILState_Ensure();
	PyObject *main_mod = PyImport_AddModule("__main__");
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *api = PyDict_GetItemString(main_dict, "_training_shim_api");
	char *out = NULL;
	if (!api) {
		set_err(err_out, "_training_shim_api missing");
		goto done;
	}
	PyObject *meth = PyObject_GetAttrString(api, "submit_job");
	if (!meth || !PyCallable_Check(meth)) {
		set_err(err_out, "submit_job not callable");
		Py_XDECREF(meth);
		goto done;
	}
	PyObject *k = PyUnicode_FromString(kind);
	PyObject *p = PyUnicode_FromString(payload_json);
	if (!k || !p) {
		set_err(err_out, "OOM unicode");
		Py_XDECREF(k);
		Py_XDECREF(p);
		Py_DECREF(meth);
		goto done;
	}
	PyObject *args = PyTuple_Pack(2, k, p);
	Py_DECREF(k);
	Py_DECREF(p);
	if (!args) {
		set_err(err_out, "OOM tuple");
		Py_DECREF(meth);
		goto done;
	}
	PyObject *res = PyObject_CallObject(meth, args);
	Py_DECREF(args);
	Py_DECREF(meth);
	if (!res) {
		PyErr_Print();
		set_err(err_out, "submit_job failed");
		goto done;
	}
	PyObject *json_mod = PyImport_ImportModule("json");
	PyObject *dumps = PyObject_GetAttrString(json_mod, "dumps");
	PyObject *j = PyObject_CallFunctionObjArgs(dumps, res, NULL);
	Py_DECREF(res);
	Py_DECREF(dumps);
	Py_DECREF(json_mod);
	if (!j) {
		PyErr_Print();
		set_err(err_out, "json.dumps failed");
		goto done;
	}
	out = py_unicode_to_utf8(j);
	Py_DECREF(j);
done:
	PyGILState_Release(gil);
	return out;
}

char *training_job_status(const char *job_id, char **err_out) {
	if (!job_id) {
		set_err(err_out, "job_id required");
		return NULL;
	}
	if (!g_initialized) {
		set_err(err_out, "training not initialized");
		return NULL;
	}
	PyGILState_STATE gil = PyGILState_Ensure();
	PyObject *main_mod = PyImport_AddModule("__main__");
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *api = PyDict_GetItemString(main_dict, "_training_shim_api");
	char *out = NULL;
	if (!api) {
		set_err(err_out, "_training_shim_api missing");
		goto done;
	}
	PyObject *meth = PyObject_GetAttrString(api, "job_status");
	if (!meth) {
		set_err(err_out, "job_status missing");
		goto done;
	}
	PyObject *arg = PyUnicode_FromString(job_id);
	if (!arg) {
		set_err(err_out, "OOM");
		Py_DECREF(meth);
		goto done;
	}
	PyObject *t = PyTuple_Pack(1, arg);
	Py_DECREF(arg);
	if (!t) {
		Py_DECREF(meth);
		goto done;
	}
	PyObject *res = PyObject_CallObject(meth, t);
	Py_DECREF(t);
	Py_DECREF(meth);
	if (!res) {
		PyErr_Print();
		set_err(err_out, "job_status failed");
		goto done;
	}
	PyObject *json_mod = PyImport_ImportModule("json");
	PyObject *dumps = PyObject_GetAttrString(json_mod, "dumps");
	PyObject *j = PyObject_CallFunctionObjArgs(dumps, res, NULL);
	Py_DECREF(res);
	Py_DECREF(dumps);
	Py_DECREF(json_mod);
	if (!j) {
		PyErr_Print();
		set_err(err_out, "json.dumps failed");
		goto done;
	}
	out = py_unicode_to_utf8(j);
	Py_DECREF(j);
done:
	PyGILState_Release(gil);
	return out;
}

char *training_list_jobs(char **err_out) {
	return call_eval_json("__import__('json').dumps(_training_shim_api.list_jobs())", "list_jobs failed", err_out);
}

/* Returns 1 cancelled, 0 not cancelled, -1 error (see err_out). */
int training_cancel_job(const char *job_id, char **err_out) {
	if (!job_id) {
		set_err(err_out, "job_id required");
		return -1;
	}
	if (!g_initialized) {
		set_err(err_out, "training not initialized");
		return -1;
	}
	PyGILState_STATE gil = PyGILState_Ensure();
	PyObject *main_mod = PyImport_AddModule("__main__");
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *fn = PyDict_GetItemString(main_dict, "_training_cancel_job");
	int ok = 0;
	if (!fn || !PyCallable_Check(fn)) {
		set_err(err_out, "_training_cancel_job missing or not callable");
		ok = -1;
		goto done;
	}
	PyObject *arg = PyUnicode_FromString(job_id);
	if (!arg) {
		set_err(err_out, "cancel_job: OOM");
		ok = -1;
		goto done;
	}
	PyObject *t = PyTuple_Pack(1, arg);
	Py_DECREF(arg);
	if (!t) {
		set_err(err_out, "cancel_job: OOM tuple");
		ok = -1;
		goto done;
	}
	PyObject *r = PyObject_CallObject(fn, t);
	Py_DECREF(t);
	if (!r) {
		PyErr_Print();
		set_err(err_out, "cancel_job failed");
		ok = -1;
		goto done;
	}
	{
		int truth = PyObject_IsTrue(r);
		Py_DECREF(r);
		if (truth < 0) {
			PyErr_Print();
			set_err(err_out, "cancel_job: boolean conversion failed");
			ok = -1;
		} else {
			ok = truth ? 1 : 0;
		}
	}
done:
	if (ok >= 0 && PyErr_Occurred())
		PyErr_Clear();
	PyGILState_Release(gil);
	return ok;
}

int training_unload(char **err_out) {
	char *s = call_eval_json("__import__('json').dumps(_training_shim_api.unload())", "unload failed", err_out);
	if (!s)
		return -1;
	training_free(s);
	return 0;
}

void training_shutdown(void) {
	if (!g_initialized)
		return;
	PyGILState_STATE gil = PyGILState_Ensure();
	PyObject *main_mod = PyImport_AddModule("__main__");
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *fn = PyDict_GetItemString(main_dict, "shutdown_ollama_training");
	if (fn && PyCallable_Check(fn)) {
		PyObject *r = PyObject_CallObject(fn, NULL);
		Py_XDECREF(r);
	}
	if (PyErr_Occurred())
		PyErr_Clear();
	PyGILState_Release(gil);

	/* Do not PyEval_RestoreThread / Py_Finalize — unsafe with torch and Go thread model. */
	g_initialized = 0;
}
