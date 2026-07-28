package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestProbeRunnerMetadata(t *testing.T) {
	runner := &runnerRef{
		numParallel: 2,
		model: &Model{
			Config: model.ConfigV2{
				ModelFamily: "qwen35",
				Parser:      "qwen3.5",
			},
			Options: map[string]any{"num_ctx": float64(8192)},
		},
		Options: &api.Options{
			Runner: api.Runner{
				NumCtx: 4096,
				NumGPU: 99,
			},
		},
		llama: &mockLlm{},
	}
	meta := probeRunnerMetadata(runner)
	if meta.NumCtx != 4096 {
		t.Fatalf("num_ctx=%d", meta.NumCtx)
	}
	if meta.ManifestNumCtx != 8192 {
		t.Fatalf("manifest_num_ctx=%d", meta.ManifestNumCtx)
	}
	if meta.NumParallel != 2 {
		t.Fatalf("num_parallel=%d", meta.NumParallel)
	}
	if meta.NumGPU != 99 {
		t.Fatalf("num_gpu=%d", meta.NumGPU)
	}
	if meta.Backend != "ggml" {
		t.Fatalf("backend=%q", meta.Backend)
	}
	if meta.Parser != "qwen3.5" {
		t.Fatalf("parser=%q", meta.Parser)
	}
	if !meta.SupportsThinking {
		t.Fatal("expected supports_thinking")
	}
	if meta.ProbedAt.IsZero() {
		t.Fatal("expected probed_at")
	}
}

type mockLLMWithOffload struct {
	mockLlm
	offloaded, total uint64
}

func (m *mockLLMWithOffload) GPULayerOffload() (uint64, uint64) {
	return m.offloaded, m.total
}

func TestProbeRunnerMetadataGPULayers(t *testing.T) {
	runner := &runnerRef{
		model: &Model{ShortName: "partial"},
		Options: &api.Options{
			Runner: api.Runner{NumCtx: 4096, NumGPU: 12},
		},
		llama: &mockLLMWithOffload{offloaded: 12, total: 32},
	}
	meta := probeRunnerMetadata(runner)
	if meta.GPULayersOffloaded != 12 || meta.GPULayersTotal != 32 {
		t.Fatalf("gpu layers = %d/%d, want 12/32", meta.GPULayersOffloaded, meta.GPULayersTotal)
	}
}

func TestProcessModelsSnapshotSkipsLoading(t *testing.T) {
	sched := InitScheduler(t.Context())
	ready := &runnerRef{
		model: &Model{ShortName: "ready"},
		llama: &mockLlm{},
		loadedMeta: api.LoadedModelMetadata{
			NumCtx:   4096,
			ProbedAt: time.Now().UTC(),
		},
	}
	loading := &runnerRef{
		model:   &Model{ShortName: "loading"},
		loading: true,
		llama:   &mockLlm{},
	}
	sched.loadedMu.Lock()
	sched.loaded["ready"] = ready
	sched.loaded["loading"] = loading
	sched.loadedMu.Unlock()

	models := sched.ProcessModelsSnapshot()
	if len(models) != 1 || models[0].Name != "ready" {
		t.Fatalf("models=%+v", models)
	}
	if models[0].LoadedMetadata == nil || models[0].LoadedMetadata.NumCtx != 4096 {
		t.Fatalf("metadata=%+v", models[0].LoadedMetadata)
	}
}

func TestSyncRunnerLoadOptionsProbesMetadata(t *testing.T) {
	runner := &runnerRef{
		model: &Model{
			Options: map[string]any{"num_ctx": float64(8192)},
		},
		Options: &api.Options{
			Runner: api.Runner{NumCtx: 999},
		},
		llama: &mockLlm{contextLength: 4096},
	}
	syncRunnerLoadOptions(runner)
	if runner.loadedMeta.ProbedAt.IsZero() {
		t.Fatal("expected metadata probe after sync")
	}
	if runner.loadedMeta.NumCtx != 4096 {
		t.Fatalf("num_ctx=%d", runner.loadedMeta.NumCtx)
	}
}
