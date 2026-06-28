package pyembed

// This package is the only CGO entry for embedded training. It links libpython3-embed
// (pkg-config python3-embed) and delegates interpreter calls to training_shim.c.
//
// WHY pkg-config name is fixed in source: Go #cgo directives cannot be conditional. On Linux
// when distro python3-embed is 3.10 but operators want 3.11, source ./scripts/training_embed_build_env.sh
// before go build — it overlays PKG_CONFIG_PATH so pkg-config python3-embed resolves to 3.11.
//
// Why JSON strings across the boundary: avoids hand-written PyObject glue for every
// request type; training.py already speaks dicts; HTTP/TCP layers in Go own framing.

import "unsafe"

/*
#cgo pkg-config: python3-embed
#cgo darwin LDFLAGS: -Wl,-rpath,/Applications/Xcode.app/Contents/Developer/Library/Frameworks -Wl,-rpath,/Library/Developer/CommandLineTools/Library/Frameworks
#cgo CFLAGS: -I${SRCDIR}
#include <stdlib.h>
#include "training_shim.h"
*/
import "C"

// IsInitialized reports whether embedded training finished startup successfully.
func IsInitialized() bool {
	return C.training_is_initialized() != 0
}

// InitAborted reports whether a prior training_init failed after Py_Initialize; the process must restart to retry.
func InitAborted() bool {
	return C.training_init_aborted() != 0
}

// InitEmbeddedPython starts embedded CPython, loads training.py via bootstrap, and releases the GIL.
// Must be called once per process after training_preinit_native_module (done here).
// Repo root must contain training.py; see OLLAMA_TRAINING_PYTHONPATH in docs/gpu-training.md.
func InitEmbeddedPython(repoRoot string) error {
	C.training_preinit_native_module()
	root := C.CString(repoRoot)
	defer C.free(unsafe.Pointer(root))
	bs := C.CString(bootstrapPy)
	defer C.free(unsafe.Pointer(bs))
	var errOut *C.char
	if C.training_init(root, bs, &errOut) != 0 {
		msg := "training_init failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.free(unsafe.Pointer(errOut))
		}
		return &initError{s: msg}
	}
	return nil
}

type initError struct{ s string }

func (e *initError) Error() string { return e.s }

func HealthJSON() (string, error) {
	var errOut *C.char
	out := C.training_health(&errOut)
	if out == nil {
		msg := "health failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.free(unsafe.Pointer(errOut))
		}
		return "", &initError{s: msg}
	}
	defer C.training_free(out)
	return C.GoString(out), nil
}

func SubmitJobJSON(kind, payloadJSON string) (string, error) {
	k := C.CString(kind)
	defer C.free(unsafe.Pointer(k))
	p := C.CString(payloadJSON)
	defer C.free(unsafe.Pointer(p))
	var errOut *C.char
	out := C.training_submit_job(k, p, &errOut)
	if out == nil {
		msg := "submit_job failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.free(unsafe.Pointer(errOut))
		}
		return "", &initError{s: msg}
	}
	defer C.training_free(out)
	return C.GoString(out), nil
}

func JobStatusJSON(jobID string) (string, error) {
	j := C.CString(jobID)
	defer C.free(unsafe.Pointer(j))
	var errOut *C.char
	out := C.training_job_status(j, &errOut)
	if out == nil {
		msg := "job_status failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.free(unsafe.Pointer(errOut))
		}
		return "", &initError{s: msg}
	}
	defer C.training_free(out)
	return C.GoString(out), nil
}

func ListJobsJSON() (string, error) {
	var errOut *C.char
	out := C.training_list_jobs(&errOut)
	if out == nil {
		msg := "list_jobs failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.free(unsafe.Pointer(errOut))
		}
		return "", &initError{s: msg}
	}
	defer C.training_free(out)
	return C.GoString(out), nil
}

func CancelJob(jobID string) (bool, error) {
	j := C.CString(jobID)
	defer C.free(unsafe.Pointer(j))
	var errOut *C.char
	rc := C.training_cancel_job(j, &errOut)
	if rc < 0 {
		msg := "cancel_job failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.free(unsafe.Pointer(errOut))
		}
		return false, &initError{s: msg}
	}
	return rc != 0, nil
}

func Unload() error {
	var errOut *C.char
	if C.training_unload(&errOut) != 0 {
		msg := "unload failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.free(unsafe.Pointer(errOut))
		}
		return &initError{s: msg}
	}
	return nil
}

func Shutdown() {
	C.training_shutdown()
}

func AckVRAMHeadroom(jobID string) {
	j := C.CString(jobID)
	defer C.free(unsafe.Pointer(j))
	C.training_ack_vram_headroom(j)
}
