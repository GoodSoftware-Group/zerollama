/*
 * Embedded CPython: run zerollama-runtime in-process (uvicorn daemon thread).
 * Shares the interpreter with training when both are enabled.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime_shim.h"

#define embedded_prepare_pytorch_ld_path_ex ollama_runtime_prepare_pytorch_ld_path_ex
#include "../../pyembed_common/embedded_pytorch_env.c"
#undef embedded_prepare_pytorch_ld_path_ex

static void set_err(char **err_out, const char *msg) {
	if (!err_out)
		return;
	if (*err_out) {
		free(*err_out);
		*err_out = NULL;
	}
	if (msg)
		*err_out = strdup(msg);
}

void runtime_embed_free(char *s) {
	if (s)
		free(s);
}

static int g_runtime_started;

int runtime_embed_is_started(void) {
	return g_runtime_started;
}

static int append_sys_path(const char *dir, char **err_out) {
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

int runtime_embed_start(
	const char *repo_root,
	const char *runtime_parent,
	int port,
	const char *bootstrap_py,
	char **err_out) {
	if (g_runtime_started)
		return 0;
	if (!repo_root || !runtime_parent || !bootstrap_py || port <= 0) {
		set_err(err_out, "runtime_embed_start: invalid arguments");
		return -1;
	}

	int self_init = 0;
	if (!Py_IsInitialized()) {
		int py_major = 3, py_minor = 10;
		const char *ver = Py_GetVersion();
		if (ver)
			sscanf(ver, "%d.%d", &py_major, &py_minor);
		ollama_runtime_prepare_pytorch_ld_path_ex(repo_root, py_major, py_minor);
		Py_Initialize();
		if (!Py_IsInitialized()) {
			set_err(err_out, "Py_Initialize failed");
			return -1;
		}
		self_init = 1;
	}

	PyGILState_STATE gil = PyGILState_Ensure();

	if (append_sys_path(repo_root, err_out) != 0)
		goto out;
	if (append_sys_path(runtime_parent, err_out) != 0)
		goto out;

	PyObject *bs = Py_CompileString(bootstrap_py, "<ollama_runtime_bootstrap>", Py_file_input);
	if (!bs) {
		PyErr_Print();
		set_err(err_out, "compile runtime bootstrap failed");
		goto out;
	}
	PyObject *main_mod = PyImport_AddModule("__main__");
	if (!main_mod) {
		Py_DECREF(bs);
		set_err(err_out, "PyImport_AddModule __main__ failed");
		goto out;
	}
	PyObject *main_dict = PyModule_GetDict(main_mod);
	PyObject *r = PyEval_EvalCode(bs, main_dict, main_dict);
	Py_DECREF(bs);
	if (!r) {
		PyErr_Print();
		set_err(err_out, "eval runtime bootstrap failed");
		goto out;
	}
	Py_DECREF(r);

	PyObject *init_fn = PyDict_GetItemString(main_dict, "init_ollama_runtime_embed");
	if (!init_fn || !PyCallable_Check(init_fn)) {
		set_err(err_out, "bootstrap missing init_ollama_runtime_embed()");
		goto out;
	}

	PyObject *args = Py_BuildValue("(is)", port, runtime_parent);
	if (!args)
		goto out;
	PyObject *res = PyObject_CallObject(init_fn, args);
	Py_DECREF(args);
	if (!res) {
		PyErr_Print();
		set_err(err_out, "init_ollama_runtime_embed() failed");
		goto out;
	}
	Py_DECREF(res);

	g_runtime_started = 1;
	PyGILState_Release(gil);
	if (self_init)
		(void)PyEval_SaveThread();
	return 0;

out:
	PyGILState_Release(gil);
	if (self_init && Py_IsInitialized())
		(void)PyEval_SaveThread();
	return -1;
}
