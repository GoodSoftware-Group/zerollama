// Package vram coordinates GPU memory between the ggml scheduler and the embedded Python runtime.
//
// Phase 8 scaffolding: Go owns ordering (release runtime before legacy load; evict runners
// before runtime inference). Policy moves to Python in a later phase.
package vram

import (
	"context"

	"github.com/ollama/ollama/internal/runtimeclient"
)

// Evictor unloads ggml inference runners (implemented by server.Scheduler).
type Evictor interface {
	UnloadAllRunners()
}

// TrainingEvictor adds pause/resume around eviction (training OOM and proactive load_model).
type TrainingEvictor interface {
	Evictor
	PauseNewLoads()
	ResumeLoads()
}

// ReleaseRuntimeVRAM stops the embedded runtime's llama-server subprocess when configured.
func ReleaseRuntimeVRAM(ctx context.Context) {
	runtimeclient.TrainingHandoff(ctx)
}

// PrepareForLegacyRunner frees Python runtime GPU before a ggml runner load.
func PrepareForLegacyRunner(ctx context.Context) {
	ReleaseRuntimeVRAM(ctx)
}

// PrepareForRuntimeInference evicts ggml runners and resumes Python inference after handoff.
func PrepareForRuntimeInference(ctx context.Context, evictor Evictor) {
	if evictor != nil {
		evictor.UnloadAllRunners()
	}
	runtimeclient.ResumeInference(ctx)
}

// PrepareForTraining evicts inference VRAM before training load_model (OOM path uses the same).
// Does not resume ggml loads here — server.runTrainingGPUPolicyMonitor holds PauseNewLoads
// while training occupies the GPU when ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING is enabled.
func PrepareForTraining(ctx context.Context, evictor TrainingEvictor) {
	if evictor != nil {
		evictor.PauseNewLoads()
		evictor.UnloadAllRunners()
	}
	ReleaseRuntimeVRAM(ctx)
}
