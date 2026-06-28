//go:build !edge

package envconfig

import "testing"

func TestGgmlPauseWhenRuntimeBusyEmbedAuto(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "")
	if !GgmlPauseWhenRuntimeBusy() {
		t.Fatal("expected auto on with default embedded runtime")
	}
	if !RuntimeConfigured() {
		t.Fatal("expected runtime configured with embed default")
	}
}

func TestGgmlPauseWhenRuntimeBusyEmbedDisabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "0")
	t.Setenv("ZEROLLAMA_RUNTIME", "0")
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "0")
	if GgmlPauseWhenRuntimeBusy() {
		t.Fatal("expected auto off with embed disabled and no URL")
	}
}

func TestGgmlPauseWhenRuntimeBusyExternalURL(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "0")
	if !GgmlPauseWhenRuntimeBusy() {
		t.Fatal("expected auto on with external runtime URL")
	}
}
