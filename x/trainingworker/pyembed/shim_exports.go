package pyembed

import "sync"

/*
#include "training_shim.h"
*/
import "C"

// OOMCallback is registered by trainingworker before InitEmbeddedPython.
// It must run inference-first VRAM relief then call AckVRAMHeadroom(jobID).
// Pass nil to clear the handler (e.g. after failed init or on Close).
// The exported C hook copies the handler under a mutex so Close cannot race with fire_oom.
type OOMCallback func(jobID, message string)

var (
	oomMu      sync.Mutex
	oomHandler OOMCallback
)

// RegisterOOMHandler sets the callback invoked from Python's CUDA OOM path (via C).
func RegisterOOMHandler(f OOMCallback) {
	oomMu.Lock()
	defer oomMu.Unlock()
	oomHandler = f
}

//export go_training_oom_hook
func go_training_oom_hook(jobID *C.char, msg *C.char) {
	oomMu.Lock()
	h := oomHandler
	oomMu.Unlock()
	if h == nil {
		return
	}
	var jid, m string
	if jobID != nil {
		jid = C.GoString(jobID)
	}
	if msg != nil {
		m = C.GoString(msg)
	}
	h(jid, m)
}
