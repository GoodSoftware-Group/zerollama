// Package vram coordinates GPU memory between the ggml scheduler and the embedded Python runtime.
//
// Phase 8 scaffolding: Go owns ordering (release runtime before legacy load; evict runners
// before runtime inference). Policy moves to Python in a later phase.
//
// Phase B (wishlist): UnloadAllRunners is soft — pin/fulfillment keys survive — so /api/pin
// is not a no-op against the runtime broker. Training and exclusive bench use
// UnloadAllRunnersForced. PrepareRuntimeOpts.SkipUnload is only safe when the caller has
// already verified ggml is empty (leftover ggml + skip = dual-stack OOM).
package vram

import (
	"context"

	"github.com/ollama/ollama/internal/runtimeclient"
)

// Evictor unloads ggml inference runners (implemented by server.Scheduler).
// Soft unload: implementations should keep pin/fulfillment-protected keys.
type Evictor interface {
	UnloadAllRunners()
}

// ForcedEvictor can clear pin/fulfillment-protected runners.
// WHY separate from Evictor: soft pin and "reclaim GPU for training/bench" are different contracts.
type ForcedEvictor interface {
	UnloadAllRunnersForced()
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

// PrepareRuntimeOpts controls PrepareForRuntimeInference thrash dampening.
// Why SkipUnload: when the requested GGUF is already the runtime resident and ggml is
// empty, unloading on every chat turn causes needless cross-stack thrash (wishlist Phase B0).
type PrepareRuntimeOpts struct {
	// SkipUnload skips UnloadAllRunners when the runtime already holds the request GGUF
	// and no ggml runners are loaded (caller must enforce the ggml-empty check).
	SkipUnload bool
	// ForceUnload ignores pin/fulfillment protection (exclusive fulfillment=benchmark).
	ForceUnload bool
}

// PrepareForRuntimeInference evicts ggml runners (unless SkipUnload) and resumes Python inference.
func PrepareForRuntimeInference(ctx context.Context, evictor Evictor, opts ...PrepareRuntimeOpts) {
	skip := false
	force := false
	if len(opts) > 0 {
		skip = opts[0].SkipUnload
		force = opts[0].ForceUnload
	}
	if evictor != nil && !skip {
		if force {
			if f, ok := evictor.(ForcedEvictor); ok {
				f.UnloadAllRunnersForced()
			} else {
				evictor.UnloadAllRunners()
			}
		} else {
			evictor.UnloadAllRunners()
		}
	}
	runtimeclient.ResumeInference(ctx)
}

// PrepareForTraining evicts inference VRAM before training load_model (OOM path uses the same).
// Always force-clears pins: training must reclaim GPU even if /api/pin leases are active.
// Does not resume ggml loads here — server.runTrainingGPUPolicyMonitor holds PauseNewLoads
// while training occupies the GPU when ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING is enabled.
func PrepareForTraining(ctx context.Context, evictor TrainingEvictor) {
	if evictor != nil {
		evictor.PauseNewLoads()
		if f, ok := evictor.(ForcedEvictor); ok {
			f.UnloadAllRunnersForced()
		} else {
			evictor.UnloadAllRunners()
		}
	}
	ReleaseRuntimeVRAM(ctx)
}

// ImageGenEvictor evicts non-image runners before MLX imagegen load.
type ImageGenEvictor interface {
	TrainingEvictor
	UnloadOtherRunners(keepModelKey string)
}

// PrepareForImageGen frees other ggml runners and the Python runtime sidecar before MLX imagegen load.
// keepModelKey is the scheduler key for the image model being loaded (may already be resident).
//
// WHY UnloadOtherRunners not UnloadAllRunners: if the same image model is already loaded
// and serving an in-flight request, we must not tear it down — routes.go skips this call
// when findLoadedRunner hits, but broker callers pass keepModelKey for partial eviction.
func PrepareForImageGen(ctx context.Context, evictor ImageGenEvictor, keepModelKey string) {
	if evictor != nil {
		evictor.UnloadOtherRunners(keepModelKey)
	}
	ReleaseRuntimeVRAM(ctx)
}
