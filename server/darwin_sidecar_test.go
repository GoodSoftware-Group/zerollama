package server

import (
	"context"
	"strings"
	"testing"
	"time"
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

func TestDarwinSidecarSkipReasonURL(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	if reason := darwinSidecarSkipReason(); !strings.Contains(reason, "ZEROLLAMA_RUNTIME_URL") {
		t.Fatalf("reason=%q", reason)
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

func TestWaitRuntimeHealthRespectsContextTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := waitRuntimeHealth(ctx, "http://127.0.0.1:59999", 50*time.Millisecond)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("waitRuntimeHealth blocked too long without sidecar")
	}
}
