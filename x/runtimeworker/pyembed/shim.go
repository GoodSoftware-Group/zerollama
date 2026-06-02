//go:build cgo

package pyembed

import "unsafe"

/*
#cgo pkg-config: python3-embed
#cgo CFLAGS: -I${SRCDIR}
#include <stdlib.h>
#include "runtime_shim.h"
*/
import "C"

func EmbedStart(repoRoot, runtimeParent string, port int) error {
	root := C.CString(repoRoot)
	defer C.free(unsafe.Pointer(root))
	rt := C.CString(runtimeParent)
	defer C.free(unsafe.Pointer(rt))
	bs := C.CString(bootstrapPy)
	defer C.free(unsafe.Pointer(bs))
	var errOut *C.char
	if C.runtime_embed_start(root, rt, C.int(port), bs, &errOut) != 0 {
		msg := "runtime_embed_start failed"
		if errOut != nil {
			msg = C.GoString(errOut)
			C.runtime_embed_free(errOut)
		}
		return &embedError{s: msg}
	}
	return nil
}

func IsStarted() bool {
	return C.runtime_embed_is_started() != 0
}

type embedError struct{ s string }

func (e *embedError) Error() string { return e.s }
