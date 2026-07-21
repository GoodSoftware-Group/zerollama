package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestThinkNeedsLegacyRunner(t *testing.T) {
	if thinkNeedsLegacyRunner(nil) {
		t.Fatal("nil think should stay on runtime")
	}
	falseVal := api.ThinkValue{Value: false}
	if thinkNeedsLegacyRunner(&falseVal) {
		t.Fatal("think:false should stay on runtime")
	}
	trueVal := api.ThinkValue{Value: true}
	if !thinkNeedsLegacyRunner(&trueVal) {
		t.Fatal("think:true needs legacy")
	}
	highVal := api.ThinkValue{Value: "high"}
	if !thinkNeedsLegacyRunner(&highVal) {
		t.Fatal("think:high needs legacy")
	}
	emptyVal := api.ThinkValue{Value: "  "}
	if thinkNeedsLegacyRunner(&emptyVal) {
		t.Fatal("think empty string should stay on runtime")
	}
}

func TestModelEligibleForAgentCacheRuntime_thinkingText(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	m := &Model{
		ModelPath: "/models/qwen.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{"completion", "thinking"},
		},
	}
	if !modelEligibleForAgentCacheRuntime(m) {
		t.Fatal("thinking text GGUF should be agent-cache eligible")
	}
	if modelEligibleForLlamaCppRuntime(m) {
		t.Fatal("thinking model should remain excluded from generic runtime default")
	}
}

func TestModelEligibleForAgentCacheRuntime_visionExcluded(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	m := &Model{
		ModelPath: "/models/vl.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{"completion", "vision"},
		},
	}
	if modelEligibleForAgentCacheRuntime(m) {
		t.Fatal("vision model should not use agent-cache runtime")
	}
}

func TestWaitScheduleQoSGgufDefersBehindInteractive(t *testing.T) {
	s := &Server{sched: InitScheduler(context.Background())}
	g := &s.sched.mlxGate
	mlxKey := schedulerModelKey(&Model{Digest: "mlx1", Config: model.ConfigV2{ModelFormat: "safetensors"}})

	// Register an interactive MLX session; hold it for the duration of the test.
	release := g.begin(mlxKey, "hermes:agent:1", mlxClassInteractive, mlxQoS{})
	defer release()

	// Cancel the context after 200ms so waitScheduleQoS returns via ctx.Err()
	// rather than waiting out the 90s cooldown.
	ctx, cancel := context.WithTimeout(
		ctxWithMLXScheduleHints(context.Background(), mlxScheduleHints{Route: "generate", Stream: false}),
		300*time.Millisecond,
	)
	defer cancel()

	gguf := &Model{Digest: "gguf1", ModelPath: "/models/gguf1.gguf", Config: model.ConfigV2{ModelFormat: "gguf"}}
	opts := map[string]any{"zerollama": map[string]any{"qos_class": "background"}}

	start := time.Now()
	err := s.waitScheduleQoS(ctx, gguf, opts)
	waited := time.Since(start)
	// We expect context cancellation (deferral was active), not a clean pass.
	if err == nil {
		t.Fatal("expected context cancellation while deferred behind interactive MLX, got nil")
	}
	if waited < 150*time.Millisecond {
		t.Fatalf("expected deferral of at least 150ms, waited %v", waited)
	}
}

func TestAgentSessionBeginGgufRegistersGate(t *testing.T) {
	s := &Server{sched: InitScheduler(context.Background())}
	g := &s.sched.mlxGate
	gguf := &Model{Digest: "gguf1", ModelPath: "/models/gguf1.gguf", Config: model.ConfigV2{ModelFormat: "gguf"}}
	modelKey := schedulerModelKey(gguf)
	ctx := ctxWithMLXScheduleHints(context.Background(), mlxScheduleHints{
		Route:  "openai",
		Stream: true,
	})
	opts := map[string]any{"prompt_cache_key": "hermes:agent:test:1"}

	release, err := s.reserveScheduleQoS(ctx, gguf, opts)
	if err != nil {
		t.Fatalf("reserveScheduleQoS: %v", err)
	}
	defer release()

	bgKey, _, _ := scheduleSessionMeta(ctx, gguf, map[string]any{
		"zerollama": map[string]any{"qos_class": "background"},
	})
	now := time.Now()
	if defer_, _, _ := g.shouldDefer(modelKey, bgKey, "", mlxClassBackground, mlxQoS{}, now); !defer_ {
		t.Fatalf("background %q should defer behind gguf interactive session", bgKey)
	}
}

func TestReserveScheduleQoSClaimsBeforeRunnerWait(t *testing.T) {
	s := &Server{sched: InitScheduler(context.Background())}
	m := &Model{Digest: "mlx1", Config: model.ConfigV2{ModelFormat: "safetensors"}}
	modelKey := schedulerModelKey(m)

	releasePrimary, err := s.reserveScheduleQoS(context.Background(), m, map[string]any{
		"prompt_cache_key": "hermes:agent:discord:1",
	})
	if err != nil {
		t.Fatalf("reserve primary: %v", err)
	}
	defer releasePrimary()

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err = s.reserveScheduleQoS(ctx, m, map[string]any{
		"prompt_cache_key": "hermes:20260704_ephemeral",
	})
	waited := time.Since(start)
	if err == nil {
		t.Fatal("auxiliary should defer while primary slot is claimed")
	}
	if waited < 150*time.Millisecond {
		t.Fatalf("expected deferral, waited %v", waited)
	}

	now := time.Now()
	if defer_, _, _ := s.sched.mlxGate.shouldDefer(modelKey, "hermes:20260704_ephemeral", "", mlxClassAuxiliary, mlxQoS{}, now); !defer_ {
		t.Fatal("gate should show primary inflight during reserved slot")
	}
}
