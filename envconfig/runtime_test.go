package envconfig

import (
	"runtime"
	"testing"
)

func TestRuntimeDefaultOn(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	if RuntimeDefaultOn() {
		t.Fatal("no URL")
	}

	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	if runtime.GOOS == "darwin" {
		if RuntimeDefaultOn() {
			t.Fatal("darwin implicit off (ggml default)")
		}
	} else if !RuntimeDefaultOn() {
		t.Fatal("implicit on")
	}

	t.Setenv("ZEROLLAMA_RUNTIME", "0")
	if RuntimeDefaultOn() {
		t.Fatal("explicit off")
	}

	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	if !RuntimeDefaultOn() {
		t.Fatal("explicit on")
	}
}

func TestRuntimeEmbedDisabledOnDarwinByDefault(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only default")
	}
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	if RuntimeEmbedEnabled() {
		t.Fatal("embed should default off on darwin")
	}
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "1")
	if !RuntimeEmbedEnabled() {
		t.Fatal("embed explicit on")
	}
}

func TestRuntimeDarwinSidecarLikely(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only")
	}
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	if !RuntimeDarwinSidecarLikely() {
		t.Fatal("expected default sidecar on darwin")
	}
	if !RuntimeConfigured() {
		t.Fatal("expected runtime configured via darwin sidecar default")
	}

	t.Setenv("ZEROLLAMA_RUNTIME", "0")
	if RuntimeDarwinSidecarLikely() {
		t.Fatal("runtime=0 should disable sidecar")
	}

	t.Setenv("ZEROLLAMA_RUNTIME", "")
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "0")
	if RuntimeDarwinSidecarLikely() {
		t.Fatal("explicit sidecar off")
	}
}

func TestDarwinSidecarKillOnServeExit(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "")
	if DarwinSidecarKillOnServeExit() {
		t.Fatal("default should persist sidecar across serve restarts")
	}
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "managed")
	if !DarwinSidecarKillOnServeExit() {
		t.Fatal("managed should kill sidecar with serve")
	}
}

func TestLlamaCppBackend(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_CPP_BACKEND", "")
	if LlamaCppBackend() {
		t.Fatal("unset")
	}
	t.Setenv("ZEROLLAMA_LLAMA_CPP_BACKEND", "1")
	if !LlamaCppBackend() {
		t.Fatal("explicit on")
	}
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	t.Setenv("ZEROLLAMA_AUTO_CONFIG", "")
	ApplyLlamaCppBackendDefaults()
	if Var("ZEROLLAMA_RUNTIME") != "1" {
		t.Fatalf("runtime=%q", Var("ZEROLLAMA_RUNTIME"))
	}
	if Var("ZEROLLAMA_AUTO_CONFIG") != "1" {
		t.Fatalf("autoconfig=%q", Var("ZEROLLAMA_AUTO_CONFIG"))
	}
}

func TestLlamaServerBackendDisabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	if !LlamaServerBackendDisabled() {
		t.Fatal("0 should disable")
	}
	if LlamaServerBackend() {
		t.Fatal("0 should not be enabled")
	}
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if LlamaServerBackendDisabled() {
		t.Fatal("1 should not be disabled")
	}
}

func TestLlamaServerBackend(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	if LlamaServerBackend() {
		t.Fatal("unset")
	}
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if !LlamaServerBackend() {
		t.Fatal("explicit on")
	}
	if !LlamaServerBackendExplicit() {
		t.Fatal("1 should be explicit")
	}
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if !LlamaServerBackend() {
		t.Fatal("auto should enable backend")
	}
	if LlamaServerBackendExplicit() {
		t.Fatal("auto should not be explicit")
	}
	if !LlamaServerBackendAuto() {
		t.Fatal("auto should be detected")
	}
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "")
	ApplyLlamaServerBackendDefaults()
	if Var("ZEROLLAMA_LEGACY_RUNNER") != "1" {
		t.Fatalf("legacy_runner=%q", Var("ZEROLLAMA_LEGACY_RUNNER"))
	}
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	ApplyLlamaServerBackendDefaults()
	if Var("ZEROLLAMA_LEGACY_RUNNER") != "1" {
		t.Fatal("should not clear explicit legacy runner when flag off")
	}
}

func TestEdgeModeDefaults(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "")
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "")
	ApplyEdgeModeDefaults()
	if Var("ZEROLLAMA_LLAMA_SERVER") != "1" {
		t.Fatalf("llama_server=%q", Var("ZEROLLAMA_LLAMA_SERVER"))
	}
	if Var("ZEROLLAMA_LEGACY_RUNNER") != "1" {
		t.Fatalf("legacy_runner=%q", Var("ZEROLLAMA_LEGACY_RUNNER"))
	}
	if Var("ZEROLLAMA_RUNTIME") != "0" {
		t.Fatalf("runtime=%q", Var("ZEROLLAMA_RUNTIME"))
	}
	if Var("ZEROLLAMA_RUNTIME_EMBED") != "0" {
		t.Fatalf("runtime_embed=%q", Var("ZEROLLAMA_RUNTIME_EMBED"))
	}
	if Var("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR") != "0" {
		t.Fatalf("darwin_sidecar=%q", Var("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR"))
	}
	if RuntimeEmbedEnabled() {
		t.Fatal("embedded runtime must be off in edge mode")
	}
}

func TestEdgeModeDefaultsForcesLlamaServer(t *testing.T) {
	// Operator set ZEROLLAMA_LLAMA_SERVER=0 before --edge. Edge must override it
	// because disabling llama-server while also disabling the Python runtime would
	// leave no local inference backend at all.
	t.Setenv("ZEROLLAMA_EDGE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	ApplyEdgeModeDefaults()
	if Var("ZEROLLAMA_LLAMA_SERVER") != "1" {
		t.Fatalf("edge must force llama-server=1 even when operator set 0; got %q", Var("ZEROLLAMA_LLAMA_SERVER"))
	}
}
