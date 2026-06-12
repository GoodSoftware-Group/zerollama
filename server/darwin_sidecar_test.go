package server

import (
	"testing"
)

func TestDarwinSidecarEnvEnabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	if !darwinSidecarEnvEnabled() {
		t.Fatal("expected enabled by default env")
	}

	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "0")
	if darwinSidecarEnvEnabled() {
		t.Fatal("explicit disable")
	}

	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	if darwinSidecarEnvEnabled() {
		t.Fatal("external URL set")
	}

	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "0")
	if darwinSidecarEnvEnabled() {
		t.Fatal("runtime off")
	}
}

func TestDarwinSidecarListenDefaults(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	host, port := darwinSidecarListen()
	if host != "127.0.0.1" || port != 8081 {
		t.Fatalf("defaults: host=%q port=%d", host, port)
	}

	t.Setenv("ZEROLLAMA_RUNTIME_EMBED_PORT", "9090")
	host, port = darwinSidecarListen()
	if port != 9090 {
		t.Fatalf("embed port: %d", port)
	}
}
