//go:build darwin && uma

// Package uma gates mlxrunner MLX Eval through the machine-wide uma_daemon.
// Build with -tags uma after: make -C x/mlxrunner/uma
package uma

/*
#cgo LDFLAGS: ${SRCDIR}/libuma_embed.a
#include "uma_glue.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"
)

var (
	jobMu  sync.Mutex
	jobFn  func()
	active bool
)

// BuildEnabled reports that this binary was compiled with -tags uma.
func BuildEnabled() bool { return true }

// RuntimeEnabled is true when ZEROLLAMA_UMA_SCHED is not off (default: auto).
func RuntimeEnabled() bool { return C.uma_mlx_runtime_enabled() != 0 }

// Active is true when connected to the broker and the gate is active.
func Active() bool { return active && C.uma_mlx_active() != 0 }

// Acquire connects to uma_daemon (modes: auto|require|degraded|off).
// Unset ZEROLLAMA_UMA_SCHED defaults to auto.
func Acquire() error {
	if !RuntimeEnabled() {
		return nil
	}
	if C.uma_mlx_acquire() != 0 {
		return fmt.Errorf("uma broker: %s", C.GoString(C.uma_mlx_last_error()))
	}
	active = C.uma_mlx_active() != 0
	if active {
		slog.Info("uma broker gate active", "sock", "/tmp/uma_daemon.sock")
	}
	return nil
}

// Release disconnects from the broker and logs cumulative lease stats.
func Release() {
	C.uma_mlx_release()
	active = false
	var leases, evals C.uint64_t
	var waitMs, holdMs C.double
	C.uma_mlx_stats(&leases, &evals, &waitMs, &holdMs)
	if leases > 0 {
		slog.Info("uma broker stats",
			"leases", uint64(leases),
			"evals", uint64(evals),
			"wait_ms_total", float64(waitMs),
			"hold_ms_total", float64(holdMs),
		)
	}
}

// LeaseBegin takes a coarse GPU lease (prefill/decode/load). Nested is refcounted.
func LeaseBegin(phase string) error {
	if !Active() {
		return nil
	}
	cs := C.CString(phase)
	defer C.free(unsafe.Pointer(cs))
	if C.uma_mlx_lease_begin(cs) != 0 {
		return fmt.Errorf("uma lease begin: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return nil
}

// LeaseEnd releases a coarse GPU lease.
func LeaseEnd() {
	if !Active() {
		return
	}
	C.uma_mlx_lease_end()
}

//export goUmaMlxJob
func goUmaMlxJob(ctx unsafe.Pointer) {
	_ = ctx
	if jobFn != nil {
		jobFn()
	}
}

// RunGPU runs fn under an active lease (or a one-shot HOLD_GPU).
// Returns an error when require-mode admission fails; does not panic.
func RunGPU(fn func()) error {
	if fn == nil {
		return nil
	}
	if !Active() {
		fn()
		return nil
	}
	jobMu.Lock()
	defer jobMu.Unlock()
	jobFn = fn
	C.uma_mlx_run_gpu()
	if C.uma_mlx_last_failed() != 0 {
		return fmt.Errorf("uma: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return nil
}
