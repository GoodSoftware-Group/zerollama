package server

import (
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

func TestInferenceBackendPolicyEdge(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	t.Setenv("ZEROLLAMA_RUNTIME", "0")
	p := inferenceBackendPolicy()
	if !p.Edge {
		t.Fatal("expected edge")
	}
	if p.LlamaServer != "explicit" {
		t.Fatalf("llama_server=%q", p.LlamaServer)
	}
	if p.GgufPath != "llama-server" {
		t.Fatalf("gguf_path=%q", p.GgufPath)
	}
	if p.RuntimeChat != "off" {
		t.Fatalf("runtime_chat=%q", p.RuntimeChat)
	}
}

func TestInferenceBackendPolicyLinuxAuto(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "0")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	p := inferenceBackendPolicy()
	if p.LlamaServer != "auto" {
		t.Fatalf("llama_server=%q", p.LlamaServer)
	}
	if p.GgufPath != "llama-server" {
		t.Fatalf("gguf_path=%q", p.GgufPath)
	}
}

func TestInferenceBackendPolicyEdgeBuild(t *testing.T) {
	version.EdgeBuild = "true"
	t.Cleanup(func() { version.EdgeBuild = "false" })
	t.Setenv("ZEROLLAMA_EDGE", "")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	p := inferenceBackendPolicy()
	if !p.EdgeBuild {
		t.Fatal("expected edge_build true")
	}
	if !p.GgmlLinked && envconfig.GgmlRunnerLinked() {
		t.Fatal("unexpected ggml_linked mismatch")
	}
}
