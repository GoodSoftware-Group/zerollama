package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestConcurrencyGroupsOverlap(t *testing.T) {
	if !concurrencyGroupsOverlap([]string{"vram-heavy"}, []string{"VRAM-heavy"}) {
		t.Fatal("expected case-insensitive overlap")
	}
	if concurrencyGroupsOverlap([]string{"a"}, []string{"b"}) {
		t.Fatal("expected no overlap")
	}
}

func TestNormalizeConcurrencyGroups(t *testing.T) {
	got := normalizeConcurrencyGroups([]any{" heavy ", "vision", "heavy"})
	if len(got) != 2 || got[0] != "heavy" || got[1] != "vision" {
		t.Fatalf("got=%v", got)
	}
	got = normalizeConcurrencyGroups("heavy, vision")
	if len(got) != 2 {
		t.Fatalf("string form got=%v", got)
	}
}

func TestFindConcurrencyGroupConflict(t *testing.T) {
	ctx := t.Context()
	sched := InitScheduler(ctx)

	conflict := &runnerRef{
		modelKey: "a",
		model: &Model{
			Config: model.ConfigV2{ConcurrencyGroups: []string{"vram-heavy"}},
		},
	}
	other := &runnerRef{
		modelKey: "b",
		model: &Model{
			Config: model.ConfigV2{ConcurrencyGroups: []string{"vision"}},
		},
	}
	sched.loadedMu.Lock()
	sched.loaded["a"] = conflict
	sched.loaded["b"] = other
	sched.loadedMu.Unlock()

	pending := &Model{
		Config: model.ConfigV2{ConcurrencyGroups: []string{"vram-heavy"}},
	}
	victim := sched.findConcurrencyGroupConflict(pending)
	if victim != conflict {
		t.Fatalf("victim=%p want conflict=%p", victim, conflict)
	}
}
