//go:build edge

package server

import (
	"errors"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func TestSchedSkipGgmlRunnerLoadEdgeBuildUnlinked(t *testing.T) {
	if envconfig.GgmlRunnerLinked() {
		t.Fatal("edge build should not link ggml runner")
	}
	t.Setenv("ZEROLLAMA_EDGE", "0")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	m := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	skip, err := schedSkipGgmlRunnerLoad(m)
	if !skip || !errors.Is(err, llm.ErrGgmlRunnerUnlinked) {
		t.Fatalf("skip=%v err=%v", skip, err)
	}
}

func TestSchedSkipGgmlRunnerLoadEdgeBuildExplicitLlamaServer(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "0")
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
