package server

import (
	"errors"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestFindOtherMLXRunnerExclusive(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_EXCLUSIVE", "1")
	s := InitScheduler(t.Context())
	peer := &runnerRef{
		model:    &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}, ShortName: "qwen-mlx"},
		modelKey: "peer-key",
	}
	pendingModel := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}, ShortName: "gemma-mlx"}
	s.loadedMu.Lock()
	s.loaded["peer-key"] = peer
	s.loadedMu.Unlock()

	pending := &LlmRequest{model: pendingModel}
	got, err := s.findOtherMLXRunner(pending)
	if err != nil {
		t.Fatal(err)
	}
	if got != peer {
		t.Fatalf("want peer evicted, got %#v", got)
	}

	t.Setenv("ZEROLLAMA_MLX_EXCLUSIVE", "0")
	got, err = s.findOtherMLXRunner(pending)
	if err != nil || got != nil {
		t.Fatalf("exclusive off → no victim, got=%v err=%v", got, err)
	}
}

func TestFindOtherMLXRunnerProtectedBusy(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_EXCLUSIVE", "1")
	s := InitScheduler(t.Context())
	peer := &runnerRef{
		model:    &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}, ShortName: "held-mlx"},
		modelKey: "digest:held",
	}
	s.loadedMu.Lock()
	s.loaded["digest:held"] = peer
	s.loadedMu.Unlock()
	release := s.mlxGate.beginFulfillment("digest:held", "fulfill:complete:test", fulfillmentComplete)
	defer release()

	pending := &LlmRequest{
		model: &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}, ShortName: "gemma-mlx"},
	}
	got, err := s.findOtherMLXRunner(pending)
	if got != nil {
		t.Fatalf("protected peer must not be eviction victim: %#v", got)
	}
	if !errors.Is(err, ErrMLXExclusiveBusy) {
		t.Fatalf("want ErrMLXExclusiveBusy, got %v", err)
	}
}
