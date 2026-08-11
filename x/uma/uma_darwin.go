//go:build darwin && uma

// Package uma gates Darwin Metal clients (mlxrunner, ggml runners, llama-server) MLX Eval through the machine-wide uma_daemon.
// Build with -tags uma after: make -C x/uma
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
	"strings"
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

// Grain reports admission grain: "phase" (default) or "op" (per-Eval HOLD).
func Grain() string {
	return C.GoString(C.uma_mlx_grain())
}

// LeaseBegin takes a coarse GPU lease (prefill/decode/load). Nested is refcounted.
// No-op when ZEROLLAMA_UMA_GRAIN=op (F0625) — each Eval uses one-shot HOLD.
func LeaseBegin(phase string) error {
	return LeaseBeginUnit("gpu", phase)
}

// LeaseEnd releases a coarse GPU lease.
func LeaseEnd() {
	LeaseEndUnit("gpu")
}

// LeaseBeginUnit takes HOLD_GPU / HOLD_ANE / HOLD_AMX (unit: gpu|ane|amx).
// Unit leases are independent so GPU ∥ ANE is allowed by the broker.
func LeaseBeginUnit(unit, phase string) error {
	if !Active() {
		return nil
	}
	cu := C.CString(unit)
	cp := C.CString(phase)
	defer C.free(unsafe.Pointer(cu))
	defer C.free(unsafe.Pointer(cp))
	if C.uma_mlx_lease_begin_unit(cu, cp) != 0 {
		return fmt.Errorf("uma lease begin (%s): %s", unit, C.GoString(C.uma_mlx_last_error()))
	}
	return nil
}

// LeaseEndUnit releases a unit lease started with LeaseBeginUnit.
func LeaseEndUnit(unit string) {
	if !Active() {
		return
	}
	cu := C.CString(unit)
	defer C.free(unsafe.Pointer(cu))
	C.uma_mlx_lease_end_unit(cu)
}

//export goUmaMlxJob
func goUmaMlxJob(ctx unsafe.Pointer) {
	_ = ctx
	if jobFn != nil {
		jobFn()
	}
}

// RunGPU runs fn under an active GPU lease (or a one-shot HOLD_GPU).
func RunGPU(fn func()) error {
	return RunUnit("gpu", fn)
}

// RunUnit runs fn under HOLD_GPU|HOLD_ANE|HOLD_AMX (unit: gpu|ane|amx).
// Returns an error when require-mode admission fails; does not panic.
func RunUnit(unit string, fn func()) error {
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
	cu := C.CString(unit)
	defer C.free(unsafe.Pointer(cu))
	C.uma_mlx_run_unit(cu)
	if C.uma_mlx_last_failed() != 0 {
		return fmt.Errorf("uma: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return nil
}

// FormatGraph builds a GRAPH job body (abs chain by default).
func FormatGraph(ntok int, form, nodes string) (string, error) {
	return FormatGraphEx("", ntok, form, nodes, 0, -1, "")
}

// FormatGraphEx mirrors uma_client_format_graph_ex (F0620/F0624).
// level "" omits level=. ngen<=0 omits ngen=. eos<0 omits eos=.
// nodes may be a wire fragment ("; NOP@CPU! ; MARK@GPU?") or ops without
// a leading semicolon — a leading "; " is inserted when missing.
func FormatGraphEx(level string, ntok int, form, nodes string, ngen, eos int, toks string) (string, error) {
	nodes = strings.TrimSpace(nodes)
	if nodes != "" && !strings.HasPrefix(nodes, ";") {
		nodes = "; " + nodes
	}
	out := make([]byte, 4096)
	var clvl, cform, cnodes, ctoks *C.char
	if level != "" {
		clvl = C.CString(level)
		defer C.free(unsafe.Pointer(clvl))
	}
	cform = C.CString(form)
	defer C.free(unsafe.Pointer(cform))
	cnodes = C.CString(nodes)
	defer C.free(unsafe.Pointer(cnodes))
	if toks != "" {
		ctoks = C.CString(toks)
		defer C.free(unsafe.Pointer(ctoks))
	}
	if C.uma_mlx_format_graph_ex((*C.char)(unsafe.Pointer(&out[0])), C.size_t(len(out)),
		clvl, C.int(ntok), cform, cnodes, C.int(ngen), C.int(eos), ctoks) != 0 {
		return "", fmt.Errorf("format_graph: bad args or buffer overflow")
	}
	// C writes a NUL-terminated string; trim at first NUL.
	n := 0
	for n < len(out) && out[n] != 0 {
		n++
	}
	return string(out[:n]), nil
}

// Submit submits a raw job body (e.g. GRAPH …). project "" → UMA_JOB_NAME.
func Submit(project, job string) (uint64, error) {
	if !Active() {
		return 0, fmt.Errorf("uma: not connected")
	}
	var cproj *C.char
	if project != "" {
		cproj = C.CString(project)
		defer C.free(unsafe.Pointer(cproj))
	}
	cjob := C.CString(job)
	defer C.free(unsafe.Pointer(cjob))
	var ticket C.uint64_t
	if C.uma_mlx_submit(cproj, cjob, &ticket) != 0 {
		return 0, fmt.Errorf("submit: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return uint64(ticket), nil
}

// Wait waits for a ticket; timeoutSec <= 0 → 60s.
func Wait(ticket uint64, timeoutSec float64) (string, error) {
	if !Active() {
		return "", fmt.Errorf("uma: not connected")
	}
	buf := make([]byte, 2048)
	if C.uma_mlx_wait(C.uint64_t(ticket), C.double(timeoutSec),
		(*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf))) != 0 {
		return "", fmt.Errorf("wait: %s", C.GoString(C.uma_mlx_last_error()))
	}
	n := 0
	for n < len(buf) && buf[n] != 0 {
		n++
	}
	return string(buf[:n]), nil
}

// Graph formats nothing — submits a prebuilt GRAPH job and waits.
func Graph(project, job string, timeoutSec float64) (string, error) {
	ticket, err := Submit(project, job)
	if err != nil {
		return "", err
	}
	return Wait(ticket, timeoutSec)
}

func cname(name string) *C.char { return C.CString(name) }

// BufAlloc creates a named broker buffer.
func BufAlloc(name string, nbytes int) error {
	if !Active() {
		return fmt.Errorf("uma: not connected")
	}
	cn := cname(name)
	defer C.free(unsafe.Pointer(cn))
	if C.uma_mlx_buf_alloc(cn, C.size_t(nbytes)) != 0 {
		return fmt.Errorf("buf_alloc: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return nil
}

// BufFree drops a named buffer (best-effort).
func BufFree(name string) {
	if !Active() {
		return
	}
	cn := cname(name)
	defer C.free(unsafe.Pointer(cn))
	_ = C.uma_mlx_buf_free(cn)
}

// BufPut writes bytes into a named buffer.
func BufPut(name string, data []byte) error {
	if !Active() {
		return fmt.Errorf("uma: not connected")
	}
	if len(data) == 0 {
		return fmt.Errorf("buf_put: empty")
	}
	cn := cname(name)
	defer C.free(unsafe.Pointer(cn))
	if C.uma_mlx_buf_put(cn, unsafe.Pointer(&data[0]), C.size_t(len(data))) != 0 {
		return fmt.Errorf("buf_put: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return nil
}

// BufGet reads bytes from a named buffer into dst (must be sized).
func BufGet(name string, dst []byte) (int, error) {
	if !Active() {
		return 0, fmt.Errorf("uma: not connected")
	}
	if len(dst) == 0 {
		return 0, fmt.Errorf("buf_get: empty dst")
	}
	cn := cname(name)
	defer C.free(unsafe.Pointer(cn))
	var got C.size_t
	if C.uma_mlx_buf_get(cn, unsafe.Pointer(&dst[0]), C.size_t(len(dst)), &got) != 0 {
		return 0, fmt.Errorf("buf_get: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return int(got), nil
}

// BufExport exports a buffer for peer IMPORT; returns iosurface id + reclaim token.
func BufExport(name string) (iosurfaceID, token uint32, err error) {
	if !Active() {
		return 0, 0, fmt.Errorf("uma: not connected")
	}
	cn := cname(name)
	defer C.free(unsafe.Pointer(cn))
	var id, tok C.uint32_t
	if C.uma_mlx_buf_export(cn, &id, &tok) != 0 {
		return 0, 0, fmt.Errorf("buf_export: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return uint32(id), uint32(tok), nil
}

// BufReclaim reclaims an exported buffer with the export token.
func BufReclaim(name string, token uint32) error {
	if !Active() {
		return fmt.Errorf("uma: not connected")
	}
	cn := cname(name)
	defer C.free(unsafe.Pointer(cn))
	if C.uma_mlx_buf_reclaim(cn, C.uint32_t(token)) != 0 {
		return fmt.Errorf("buf_reclaim: %s", C.GoString(C.uma_mlx_last_error()))
	}
	return nil
}
