package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestEnsureQoSDefaultsImageGeneration(t *testing.T) {
	opts := map[string]any{}
	ensureQoSDefaults(opts, mlxScheduleHints{Modality: mlxModalityImageGeneration})
	z, ok := opts["zerollama"].(map[string]any)
	if !ok || z["qos_class"] != "background" || z["cache_scope"] != qosCacheScopeShared {
		t.Fatalf("defaults: %v", opts)
	}
}

func TestEnsureQoSDefaultsExplicitSkipped(t *testing.T) {
	opts := map[string]any{
		"zerollama": map[string]any{"qos_class": "interactive"},
	}
	ensureQoSDefaults(opts, mlxScheduleHints{Modality: mlxModalityImageGeneration})
	z := opts["zerollama"].(map[string]any)
	if z["qos_class"] != "interactive" {
		t.Fatalf("explicit qos should not be overwritten: %v", z)
	}
}

func TestMLXModalityFromChat(t *testing.T) {
	text := &api.ChatRequest{Messages: []api.Message{{Role: "user", Content: "hi"}}}
	if mlxModalityFromChat(text) != mlxModalityText {
		t.Fatal("text chat")
	}
	vision := &api.ChatRequest{Messages: []api.Message{{Role: "user", Images: []api.ImageData{{1, 2, 3}}}}}
	if mlxModalityFromChat(vision) != mlxModalityVision {
		t.Fatal("vision chat")
	}
	video := &api.ChatRequest{
		Messages: []api.Message{{
			Role:   "user",
			Videos: []api.VideoData{{9}},
		}},
	}
	if mlxModalityFromChat(video) != mlxModalityVideoUnderstanding {
		t.Fatal("video chat")
	}
}

func TestClassifyModalityImageGeneration(t *testing.T) {
	class, _ := classifyMLXSession("", mlxScheduleHints{
		Route:    "image_generation",
		Modality: mlxModalityImageGeneration,
	}, nil)
	if class != mlxClassBackground {
		t.Fatalf("image gen = %s", class)
	}
}

func TestClassifyModalityVisionStream(t *testing.T) {
	class, _ := classifyMLXSession("", mlxScheduleHints{
		Route:    "openai",
		Modality: mlxModalityVision,
		Stream:   true,
	}, nil)
	if class != mlxClassAuxiliary {
		t.Fatalf("vision stream = %s", class)
	}
}

func TestMLXAgentGateWaitBehindAnyInteractive(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:test"
	primary := "hermes:agent:discord:1"

	// Hold interactive inflight; 90s post-release cooldown would make a clean return hang.
	release := g.begin(modelKey, primary, mlxClassInteractive, mlxQoS{})
	defer release()

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := g.waitBehindAnyInteractive(ctx, mlxClassBackground, "bg:gpu:zerollama:media")
	waited := time.Since(start)
	if err == nil {
		t.Fatal("expected context cancellation while deferred behind interactive")
	}
	if waited < 150*time.Millisecond {
		t.Fatalf("waited %v, expected defer behind interactive", waited)
	}
}

func TestWaitRequestQoSNonMLX(t *testing.T) {
	s := &Server{sched: InitScheduler(context.Background())}
	modelKey := schedulerModelKey(&Model{Digest: "abc", Config: model.ConfigV2{ModelFormat: "safetensors"}})

	release := s.sched.mlxGate.begin(modelKey, "hermes:agent:1", mlxClassInteractive, mlxQoS{})
	defer release()

	opts := map[string]any{}
	hints := mlxScheduleHints{Route: "video_generation", Modality: mlxModalityVideoGeneration}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := s.waitRequestQoS(ctx, nil, opts, hints)
	if err == nil {
		t.Fatal("expected timeout waiting behind interactive")
	}
}
