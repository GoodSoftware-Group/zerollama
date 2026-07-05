package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
)

func TestMLXAgentGateWaitWhileInflight(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:test"
	sessionKey := "hermes:agent:discord:1"

	release := g.begin(modelKey, sessionKey, mlxClassInteractive, mlxQoS{})
	defer release()

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := g.waitForSlot(ctx, modelKey, "hermes:other:2", mlxClassAuxiliary)
	waited := time.Since(start)
	if err == nil {
		t.Fatal("expected context cancellation while primary inflight")
	}
	if waited < 150*time.Millisecond {
		t.Fatalf("waited %v, expected defer while inflight", waited)
	}
}

func TestMLXAgentGateSameSessionNoDefer(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:test"
	sessionKey := "hermes:agent:discord:1"

	release := g.begin(modelKey, sessionKey, mlxClassInteractive, mlxQoS{})
	defer release()

	start := time.Now()
	if err := g.waitForSlot(context.Background(), modelKey, sessionKey, mlxClassInteractive); err != nil {
		t.Fatalf("waitForSlot: %v", err)
	}
	if waited := time.Since(start); waited > 50*time.Millisecond {
		t.Fatalf("same session should not block, waited %v", waited)
	}
}

func TestMLXAgentGateProceedWhenIdle(t *testing.T) {
	g := newMLXAgentGate()
	start := time.Now()
	if err := g.waitForSlot(context.Background(), "digest:idle", "hermes:any", mlxClassAuxiliary); err != nil {
		t.Fatalf("waitForSlot: %v", err)
	}
	if time.Since(start) > 50*time.Millisecond {
		t.Fatal("idle gate should not block")
	}
}

func TestMLXAgentGateCooldownBlocksDifferentKey(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:test"
	sessionKey := "hermes:agent:discord:1"

	release := g.begin(modelKey, sessionKey, mlxClassInteractive, mlxQoS{})
	release()

	now := time.Now()
	if defer_, _, _ := g.shouldDefer(modelKey, "hermes:other", mlxClassAuxiliary, now); !defer_ {
		t.Fatal("different auxiliary key should defer during primary cooldown")
	}
	if defer_, _, _ := g.shouldDefer(modelKey, sessionKey, mlxClassInteractive, now); defer_ {
		t.Fatal("same primary key should not defer during cooldown")
	}
	future := now.Add(mlxSidecarAgentCooldown + time.Second)
	if defer_, _, _ := g.shouldDefer(modelKey, "hermes:other", mlxClassAuxiliary, future); defer_ {
		t.Fatal("should not defer after cooldown expires")
	}
}

func TestMLXAgentGatePrimaryPreemptsAuxiliaryCooldown(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:test"

	release := g.begin(modelKey, "aux:digest:test", mlxClassAuxiliary, mlxQoS{})
	release()

	now := time.Now()
	defer_, policy, _ := g.shouldDefer(modelKey, "hermes:agent:discord:1", mlxClassInteractive, now)
	if defer_ {
		t.Fatalf("primary should preempt auxiliary cooldown, policy=%s", policy)
	}
	if policy != "interactive_preempt_cooldown" {
		t.Fatalf("policy=%s want interactive_preempt_cooldown", policy)
	}
}

func TestMLXAgentGateAuxiliaryWaitsForPrimary(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:test"
	primary := "hermes:agent:discord:1"

	release := g.begin(modelKey, primary, mlxClassInteractive, mlxQoS{})
	defer release()

	defer_, policy, _ := g.shouldDefer(modelKey, "aux:digest:test", mlxClassAuxiliary, time.Now())
	if !defer_ {
		t.Fatal("auxiliary should defer behind active primary")
	}
	if policy != "lower_wait_interactive" {
		t.Fatalf("policy=%s want lower_wait_interactive", policy)
	}
}

func TestMLXAgentGateContextCancel(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:busy"
	sessionKey := "hermes:agent:discord:1"

	release := g.begin(modelKey, sessionKey, mlxClassInteractive, mlxQoS{})
	defer release()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := g.waitForSlot(ctx, modelKey, "hermes:other", mlxClassAuxiliary)
	if err == nil {
		t.Fatal("expected context error")
	}
}

func TestMLXNeedsDeferUnkeyedMLX(t *testing.T) {
	g := newMLXAgentGate()
	m := &Model{Digest: "abc", Config: model.ConfigV2{ModelFormat: "safetensors"}}
	modelKey := schedulerModelKey(m)
	sessionKey := "hermes:agent:discord:1"

	if defer_, _ := g.mlxNeedsDefer(m, modelKey, "bg:"+modelKey, mlxClassBackground); defer_ {
		t.Fatal("should not defer when no session is hot")
	}

	release := g.begin(modelKey, sessionKey, mlxClassInteractive, mlxQoS{})
	defer release()
	if defer_, _ := g.mlxNeedsDefer(m, modelKey, "bg:"+modelKey, mlxClassBackground); !defer_ {
		t.Fatal("background should defer when interactive session is hot")
	}
}

func TestMLXNeedsDeferDifferentKey(t *testing.T) {
	g := newMLXAgentGate()
	m := &Model{Digest: "abc", Config: model.ConfigV2{ModelFormat: "safetensors"}}
	modelKey := schedulerModelKey(m)
	hot := "hermes:agent:discord:1"
	ephemeral := "hermes:20260704_session"

	release := g.begin(modelKey, hot, mlxClassInteractive, mlxQoS{})
	defer release()

	if defer_, _ := g.mlxNeedsDefer(m, modelKey, ephemeral, mlxClassAuxiliary); !defer_ {
		t.Fatal("auxiliary should defer while primary is active")
	}
	if defer_, _ := g.mlxNeedsDefer(m, modelKey, hot, mlxClassInteractive); defer_ {
		t.Fatal("same primary session should not defer")
	}
}

func TestMLXNeedsDeferGGUF(t *testing.T) {
	g := newMLXAgentGate()
	gguf := &Model{ModelPath: "/models/llama.gguf"}
	if defer_, _ := g.mlxNeedsDefer(gguf, gguf.ModelPath, "", mlxClassBackground); defer_ {
		t.Fatal("GGUF should never defer via MLX gate")
	}
}

func TestMLXAgentGateTracksProjectMetadata(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:test"
	release := g.begin(modelKey, "hermes:agent:main:1", mlxClassInteractive, mlxQoS{
		SessionGroup: "simpleagent",
		ProjectID:    "simpleagent",
		ProjectName:  "bmtl",
	})
	defer release()

	sessions := g.activeSessionsForModel(modelKey, time.Now())
	if len(sessions) != 1 {
		t.Fatalf("sessions=%d want 1", len(sessions))
	}
	if sessions[0].ProjectID != "simpleagent" || sessions[0].ProjectName != "bmtl" {
		t.Fatalf("project metadata = %+v", sessions[0])
	}
	if sessions[0].SessionGroup != "simpleagent" {
		t.Fatalf("session_group=%q", sessions[0].SessionGroup)
	}
	if formatProcessProjectLabel(sessions[0].ProjectID, sessions[0].ProjectName) != "simpleagent/bmtl" {
		t.Fatal("project label mismatch")
	}
}

func TestMLXAgentSessionBeginNoop(t *testing.T) {
	s := &Server{sched: InitScheduler(context.Background())}
	ctx := context.Background()
	release := s.mlxAgentSessionBegin(ctx, nil, nil, true)
	release()
	m := &Model{Digest: "abc", Config: model.ConfigV2{ModelFormat: "safetensors"}}
	release = s.mlxAgentSessionBegin(ctx, m, nil, false)
	release()
	release = s.mlxAgentSessionBegin(ctx, m, nil, true)
	release()
}

func TestMLXSessionKey(t *testing.T) {
	if mlxSessionKey(nil) != "" {
		t.Fatal("nil opts should return empty")
	}
	if mlxSessionKey(map[string]any{"prompt_cache_key": "hermes:1"}) != "hermes:1" {
		t.Fatal("should extract key")
	}
	if mlxSessionKey(map[string]any{}) != "" {
		t.Fatal("missing key should return empty")
	}
}
