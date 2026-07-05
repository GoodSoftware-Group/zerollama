package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
)

func TestProcessModelsSnapshotIncludesProjectSessions(t *testing.T) {
	sched := InitScheduler(t.Context())
	m := &Model{
		ShortName: "gemma4:test",
		Digest:    "abc123",
		Config:    model.ConfigV2{ModelFormat: "safetensors"},
	}
	modelKey := schedulerModelKey(m)

	runner := &runnerRef{
		model:     m,
		modelKey:  modelKey,
		loading:   false,
		totalSize: 1,
		vramSize:  1,
	}
	sched.loadedMu.Lock()
	sched.loaded[modelKey] = runner
	sched.loadedMu.Unlock()

	release := sched.mlxGate.begin(modelKey, "hermes:agent:main:1", mlxClassInteractive, mlxQoS{
		ProjectID:   "simpleagent",
		ProjectName: "zerollama",
	})
	defer release()

	models := sched.ProcessModelsSnapshot()
	if len(models) != 1 {
		t.Fatalf("models=%d want 1", len(models))
	}
	if models[0].Zerollama == nil || len(models[0].Zerollama.Sessions) != 1 {
		t.Fatalf("zerollama sessions missing: %+v", models[0].Zerollama)
	}
	session := models[0].Zerollama.Sessions[0]
	if session.ProjectID != "simpleagent" || session.ProjectName != "zerollama" {
		t.Fatalf("session=%+v", session)
	}
}

func TestMLXQoSProjectFields(t *testing.T) {
	q := mlxQoSFromOptions(map[string]any{
		"zerollama": map[string]any{
			"project_id":   "cursor",
			"project_name": "zerollama repo",
		},
	})
	if q.ProjectID != "cursor" || q.ProjectName != "zerollama repo" {
		t.Fatalf("project fields = %+v", q)
	}
}

func TestProcessSessionInfoHotUntil(t *testing.T) {
	slot := &mlxSessionSlot{
		sessionKey:   "hermes:agent:1",
		sessionClass: mlxClassInteractive,
		projectID:    "app",
		hotUntil:     time.Now().Add(time.Minute),
	}
	info := slot.processSessionInfo(time.Now())
	if info.Inflight != 0 {
		t.Fatalf("inflight=%d", info.Inflight)
	}
	if info.HotUntil.IsZero() {
		t.Fatal("expected hot_until on cooldown slot")
	}
}

func TestProcessProjectLabel(t *testing.T) {
	if got := formatProcessProjectLabel("a", "b"); got != "a/b" {
		t.Fatalf("got %q", got)
	}
	if got := formatProcessProjectLabel("", "b"); got != "b" {
		t.Fatalf("got %q", got)
	}
}
