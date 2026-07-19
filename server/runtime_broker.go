package server

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/internal/runtimeclient"
	"github.com/ollama/ollama/server/vram"
)

// ErrRuntimeVRAMPinnedGGML is returned when pin/fulfillment/in-use ggml runners remain
// after a pin-respecting unload — starting runtime alongside them risks OOM.
var ErrRuntimeVRAMPinnedGGML = errors.New("runtime inference blocked: ggml runners still resident (pin, fulfillment, or in-use); release pins or retry")

// ErrRuntimeVRAMPinnedGGUF is returned when an active pin holds a different runtime GGUF
// than the request — Go cannot keep two Python GGUFs warm.
var ErrRuntimeVRAMPinnedGGUF = errors.New("runtime inference blocked: active pin holds a different GGUF; release the pin or use the pinned model")

// prepareRuntimeVRAM frees ggml VRAM before runtime proxy unless the request GGUF is
// already resident and ggml is empty (Phase B0 thrash dampen). forceUnload always
// force-evicts including pins (e.g. exclusive bench) and skips conflict checks.
//
// Fail-closed: residual protected ggml or a conflicting runtime GGUF pin returns an
// error *before* ResumeInference so we do not start Python on top of pinned ggml.
func (s *Server) prepareRuntimeVRAM(ctx context.Context, reqGGUF string, forceUnload bool) error {
	if !forceUnload {
		if err := s.errIfRuntimePinConflicts(reqGGUF); err != nil {
			return err
		}
	}

	var ev vram.Evictor
	if s != nil && s.sched != nil {
		ev = s.sched
	}
	// WHY ggml-empty gate: skipping unload while ggml still holds VRAM risks OOM when
	// llama-server grows KV / loads alongside leftover runners.
	skip := !forceUnload && runtimeGGUFAlreadyResident(ctx, reqGGUF) && !s.ggmlRunnersLoaded()
	if skip {
		runtimeclient.ResumeInference(ctx)
		return nil
	}
	if forceUnload {
		vram.PrepareForRuntimeInference(ctx, ev, vram.PrepareRuntimeOpts{ForceUnload: true})
		return nil
	}
	// Unload with pin respect; only resume runtime when ggml is actually clear.
	if ev != nil {
		ev.UnloadAllRunners()
	}
	if s.ggmlRunnersLoaded() {
		return ErrRuntimeVRAMPinnedGGML
	}
	runtimeclient.ResumeInference(ctx)
	return nil
}

// errIfRuntimePinConflicts rejects a runtime request whose GGUF differs from an
// active pin's runtime GGUF (single-resident Python contract).
func (s *Server) errIfRuntimePinConflicts(reqGGUF string) error {
	reqGGUF = strings.TrimSpace(reqGGUF)
	if s == nil || s.sched == nil {
		return nil
	}
	pinned := uniquePinnedRuntimeGGUFs(s.sched.mlxGate.pinnedRuntimeGGUFs())
	if len(pinned) == 0 {
		return nil
	}
	if reqGGUF == "" {
		return ErrRuntimeVRAMPinnedGGUF
	}
	for _, p := range pinned {
		if ggufPathsEqual(reqGGUF, p) {
			return nil
		}
	}
	return ErrRuntimeVRAMPinnedGGUF
}

func uniquePinnedRuntimeGGUFs(in []string) []string {
	var out []string
	for _, p := range in {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		dup := false
		for _, existing := range out {
			if ggufPathsEqual(existing, p) || existing == p {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, p)
		}
	}
	return out
}

// ggmlRunnersLoaded reports whether the ggml scheduler still has resident (or loading) runners.
func (s *Server) ggmlRunnersLoaded() bool {
	if s == nil || s.sched == nil {
		return false
	}
	return s.sched.ggmlRunnersLoaded()
}

// runtimeGGUFAlreadyResident reports whether /health model_swap.loaded_gguf matches reqGGUF.
func runtimeGGUFAlreadyResident(ctx context.Context, reqGGUF string) bool {
	reqGGUF = strings.TrimSpace(reqGGUF)
	if reqGGUF == "" || !runtimeHealthProbeRequired() {
		return false
	}
	h := runtimeInferenceHealth(ctx)
	if !h.ok || !h.llamaLoaded || h.loadedGGUF == "" {
		return false
	}
	return ggufPathsEqual(reqGGUF, h.loadedGGUF)
}

// runtimeForceUnload is true when exclusive fulfillment/benchmark is active — always clear ggml peers.
func runtimeForceUnload(s *Server, opts map[string]any) bool {
	qos := mlxQoSFromOptions(opts)
	if qos.Fulfillment.Exclusive() {
		return true
	}
	if s == nil || s.sched == nil {
		return false
	}
	if hold, ok := s.sched.mlxGate.fulfillmentActive(time.Now()); ok && hold.mode.Exclusive() {
		return true
	}
	return false
}

// abortIfPrepareRuntimeVRAMFailed writes a busy 503 when prepareRuntimeVRAM fails.
// WHY 503 + Retry-After (not 409): same client contract as queue-full / Metal contention —
// Orient/Decide already back off on busy; pin conflicts are transient once leases expire.
func (s *Server) abortIfPrepareRuntimeVRAMFailed(c *gin.Context, err error) bool {
	if err == nil {
		return false
	}
	c.Header("Retry-After", strconv.Itoa(defaultBusyRetryAfterSec))
	body := gin.H{
		"error":       err.Error(),
		"retry_after": defaultBusyRetryAfterSec,
	}
	switch {
	case errors.Is(err, ErrRuntimeVRAMPinnedGGUF):
		body["cause"] = "runtime_pin_gguf"
	case errors.Is(err, ErrRuntimeVRAMPinnedGGML):
		body["cause"] = "runtime_pin_ggml"
	default:
		body["cause"] = "runtime_vram"
	}
	c.AbortWithStatusJSON(http.StatusServiceUnavailable, body)
	return true
}
