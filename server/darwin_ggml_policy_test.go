package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"runtime"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

func darwinMetalPolicyTestServer(t *testing.T, llamaLoaded bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"llama_server": llamaLoaded})
	}))
}

func TestDarwinRuntimeMetalBlocksGgml(t *testing.T) {
	if modelEligibleForRuntimeDefault(nil) {
		t.Fatal("nil model should not be eligible")
	}

	text := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}

	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "0")

	ctx := context.Background()
	if darwinRuntimeMetalBlocksGgml(ctx, text) {
		t.Fatal("expected false without health server")
	}
	if darwinGgmlContentionWithRuntime(ctx, text) {
		t.Fatal("expected false without health server")
	}

	if envconfig.LegacyRunnerForced() {
		t.Fatal("legacy runner should be off")
	}
}

func TestDarwinMetalPolicyWithRuntimeLoaded(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only Metal contention policy")
	}

	srv := darwinMetalPolicyTestServer(t, true)
	defer srv.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "0")

	text := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	vision := &Model{
		ModelPath: "/tmp/v.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{
				string(model.CapabilityCompletion),
				string(model.CapabilityVision),
			},
		},
	}

	ctx := context.Background()
	if !darwinRuntimeMetalBlocksGgml(ctx, text) {
		t.Fatal("runtime-eligible text model should defer ggml when sidecar holds Metal")
	}
	if darwinGgmlContentionWithRuntime(ctx, text) {
		t.Fatal("runtime-eligible text should use defer path, not contention error")
	}
	if !darwinGgmlContentionWithRuntime(ctx, vision) {
		t.Fatal("vision model should hit contention when runtime holds Metal")
	}
	if darwinRuntimeMetalBlocksGgml(ctx, vision) {
		t.Fatal("vision model should not defer via runtime-default path")
	}
	if !darwinGgmlContentionWithRuntime(ctx, nil) {
		t.Fatal("nil model should block ggml when runtime holds Metal")
	}
}

func TestDarwinMetalPolicyLegacyRunnerBypass(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only Metal contention policy")
	}

	srv := darwinMetalPolicyTestServer(t, true)
	defer srv.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "1")

	text := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}

	ctx := context.Background()
	if darwinRuntimeMetalBlocksGgml(ctx, text) {
		t.Fatal("legacy runner forced should bypass runtime defer")
	}
	if darwinGgmlContentionWithRuntime(ctx, text) {
		t.Fatal("legacy runner forced should bypass contention guard")
	}
}

func TestDarwinMetalPolicyRuntimeIdle(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only Metal contention policy")
	}

	srv := darwinMetalPolicyTestServer(t, false)
	defer srv.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "0")

	text := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	vision := &Model{
		ModelPath: "/tmp/v.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{
				string(model.CapabilityCompletion),
				string(model.CapabilityVision),
			},
		},
	}

	ctx := context.Background()
	if darwinRuntimeMetalBlocksGgml(ctx, text) {
		t.Fatal("expected false when runtime llama_server is idle")
	}
	if darwinGgmlContentionWithRuntime(ctx, vision) {
		t.Fatal("expected false when runtime llama_server is idle")
	}
}

func TestDeferInferenceIncludesDarwinRuntimeMetal(t *testing.T) {
	text := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
			ModalityBackends: map[string]string{
				model.ModalityInference: model.BackendZerollamaRuntime,
			},
		},
	}
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	if !deferInferenceToRuntime(text) {
		t.Fatal("expected defer for explicit runtime backend")
	}
}
