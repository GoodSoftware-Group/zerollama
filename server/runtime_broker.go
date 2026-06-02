package server

import (
	"context"

	"github.com/ollama/ollama/server/vram"
)

func (s *Server) prepareRuntimeVRAM(ctx context.Context) {
	var ev vram.Evictor
	if s != nil && s.sched != nil {
		ev = s.sched
	}
	vram.PrepareForRuntimeInference(ctx, ev)
}
