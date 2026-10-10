package server

// GET /slots — thin llama-server discovery shim for Odysseus / Hermes.
//
// WHY: Agents that treat the base URL as llama.cpp probe /slots for serving
// n_ctx and parallel count. Zerollama never mounted it (404), so Odysseus fell
// through and often assumed train-max context (e.g. 262144). This returns a
// llama-server-shaped array only — no slot restore / action= APIs.

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
)

// llamaCompatSlot matches the fields Odysseus reads from llama-server GET /slots.
type llamaCompatSlot struct {
	ID           int  `json:"id"`
	NCtx         int  `json:"n_ctx"`
	IsProcessing bool `json:"is_processing"`
	Speculative  bool `json:"speculative"`
}

// SlotsHandler serves GET /slots (llama.cpp compatibility).
func (s *Server) SlotsHandler(c *gin.Context) {
	c.JSON(http.StatusOK, s.llamaCompatSlots())
}

func (s *Server) llamaCompatSlots() []llamaCompatSlot {
	if s == nil || s.sched == nil {
		return []llamaCompatSlot{}
	}

	s.sched.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.sched.loaded))
	for _, r := range s.sched.loaded {
		runners = append(runners, r)
	}
	s.sched.loadedMu.Unlock()

	var primary *runnerRef
	for _, runner := range runners {
		runner.refMu.Lock()
		hasModel := runner.model != nil
		loading := runner.loading
		runner.refMu.Unlock()
		if !hasModel && !loading {
			continue
		}
		// Prefer a ready runner; otherwise keep a loading one so n_ctx is still visible.
		if primary == nil {
			primary = runner
			continue
		}
		primary.refMu.Lock()
		pLoading := primary.loading
		primary.refMu.Unlock()
		if pLoading && !loading {
			primary = runner
		}
	}
	if primary == nil {
		return []llamaCompatSlot{}
	}

	primary.refMu.Lock()
	nCtx := runnerEffectiveNumCtx(primary)
	loading := primary.loading
	busy := primary.refCount > 0
	isMLX := primary.model != nil && primary.model.IsMLX()
	primary.refMu.Unlock()

	if nCtx <= 0 {
		nCtx = 4096
	}

	nSlots := 1
	if !isMLX {
		nSlots = max(1, int(envconfig.NumParallel()))
	}

	processing := loading || busy
	out := make([]llamaCompatSlot, nSlots)
	for i := range out {
		out[i] = llamaCompatSlot{
			ID:           i,
			NCtx:         nCtx,
			IsProcessing: processing,
			Speculative:  false,
		}
	}
	return out
}
