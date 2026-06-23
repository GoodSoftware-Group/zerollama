package envconfig

import (
	"testing"

	"github.com/ollama/ollama/version"
)

func TestLinuxLlamaServerAutoEnv(t *testing.T) {
	if got := LinuxLlamaServerAutoEnv("linux", "", false, true); got != "auto" {
		t.Fatalf("linux discoverable: got %q", got)
	}
	if got := LinuxLlamaServerAutoEnv("darwin", "", false, true); got != "" {
		t.Fatalf("darwin: got %q", got)
	}
	if got := LinuxLlamaServerAutoEnv("linux", "0", false, true); got != "" {
		t.Fatalf("explicit env: got %q", got)
	}
	if got := LinuxLlamaServerAutoEnv("linux", "", true, true); got != "" {
		t.Fatalf("edge mode: got %q", got)
	}
	if got := LinuxLlamaServerAutoEnv("linux", "", false, false); got != "" {
		t.Fatalf("not discoverable: got %q", got)
	}
}

func TestApplyServeBackendEnvLinuxAuto(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "")
	ApplyServeBackendEnv(ServeBackendOpts{
		GOOS:                    "linux",
		LlamaServerDiscoverable: true,
	})
	if Var("ZEROLLAMA_LLAMA_SERVER") != "auto" {
		t.Fatalf("llama_server=%q", Var("ZEROLLAMA_LLAMA_SERVER"))
	}
	if Var("ZEROLLAMA_LEGACY_RUNNER") != "1" {
		t.Fatalf("legacy_runner=%q", Var("ZEROLLAMA_LEGACY_RUNNER"))
	}
}

func TestApplyServeBackendEnvEdge(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "")
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "")
	ApplyServeBackendEnv(ServeBackendOpts{
		GOOS:                    "linux",
		Edge:                    true,
		LlamaServerDiscoverable: true,
	})
	if Var("ZEROLLAMA_LLAMA_SERVER") != "1" {
		t.Fatalf("llama_server=%q", Var("ZEROLLAMA_LLAMA_SERVER"))
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
	if !EdgeMode() {
		t.Fatal("edge mode not enabled")
	}
	if RuntimeEmbedEnabled() {
		t.Fatal("embedded runtime must be off in edge mode")
	}
}

func TestWarnDeprecatedNewEngineNoPanic(t *testing.T) {
	t.Setenv("OLLAMA_NEW_ENGINE", "1")
	ApplyServeBackendEnv(ServeBackendOpts{GOOS: "linux"})
}

func TestApplyServeBackendEnvEdgeBuildDefault(t *testing.T) {
	version.EdgeBuild = "true"
	t.Cleanup(func() { version.EdgeBuild = "false" })
	t.Setenv("ZEROLLAMA_EDGE", "")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	ApplyServeBackendEnv(ServeBackendOpts{
		GOOS:                    "linux",
		LlamaServerDiscoverable: true,
	})
	if !EdgeMode() {
		t.Fatal("edge-marked binary should enable edge mode by default")
	}
	if Var("ZEROLLAMA_LLAMA_SERVER") != "1" {
		t.Fatalf("llama_server=%q", Var("ZEROLLAMA_LLAMA_SERVER"))
	}
}

func TestApplyServeBackendEnvEdgeBuildOptOut(t *testing.T) {
	version.EdgeBuild = "true"
	t.Cleanup(func() { version.EdgeBuild = "false" })
	t.Setenv("ZEROLLAMA_EDGE", "0")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	ApplyServeBackendEnv(ServeBackendOpts{
		GOOS:                    "linux",
		LlamaServerDiscoverable: true,
	})
	if EdgeMode() {
		t.Fatal("ZEROLLAMA_EDGE=0 must disable edge defaults on edge-marked binary")
	}
	if Var("ZEROLLAMA_LLAMA_SERVER") != "auto" {
		t.Fatalf("linux auto should still apply, got %q", Var("ZEROLLAMA_LLAMA_SERVER"))
	}
}
