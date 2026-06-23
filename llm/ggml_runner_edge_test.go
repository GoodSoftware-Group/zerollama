//go:build edge

package llm

import (
	"errors"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestGgmlRunnerRequiredEdgeWithoutLlamaServer(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	t.Setenv("ZEROLLAMA_EDGE", "0")
	if envconfig.GgmlRunnerLinked() {
		t.Fatal("edge build should not link ggml runner")
	}
	if err := ggmlRunnerRequired(nil); !errors.Is(err, ErrGgmlRunnerUnlinked) {
		t.Fatalf("err=%v", err)
	}
}

func TestGgmlRunnerRequiredEdgeWithExplicitLlamaServer(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if err := ggmlRunnerRequired(nil); err != nil {
		t.Fatalf("explicit llama-server should bypass ggml requirement: %v", err)
	}
}
