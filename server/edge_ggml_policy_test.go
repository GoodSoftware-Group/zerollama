package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestSchedSkipGgmlRunnerLoadEdgeMode(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	m := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	skip, err := schedSkipGgmlRunnerLoad(m)
	if !skip || err == nil {
		t.Fatalf("skip=%v err=%v", skip, err)
	}
}

func TestSchedSkipGgmlRunnerLoadEdgeWithLlamaServer(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	m := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	skip, err := schedSkipGgmlRunnerLoad(m)
	if skip || err != nil {
		t.Fatalf("skip=%v err=%v", skip, err)
	}
}

func TestSchedSkipGgmlRunnerLoadMLX(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	m := &Model{
		ModelPath: "/tmp/mlx",
		Config: model.ConfigV2{
			ModelFormat:  "safetensors",
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	skip, err := schedSkipGgmlRunnerLoad(m)
	if skip || err != nil {
		t.Fatalf("MLX should not be blocked: skip=%v err=%v", skip, err)
	}
}
