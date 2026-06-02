package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestRuntimeBacklogPausesGgml(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "1")
	t.Setenv("ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG", "3")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"waiting":2,"running":2,"inference_state":"running","llama_server":true}`))
	}))
	defer srv.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	if !runtimeBacklogPausesGgml(context.Background()) {
		t.Fatal("expected pause when runtime backlog >= 3")
	}
}

func TestRuntimeBacklogPausesGgmlOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "0")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"waiting":99,"running":99}`))
	}))
	defer srv.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	if runtimeBacklogPausesGgml(context.Background()) {
		t.Fatal("expected no pause when policy off")
	}
}

func TestGgmlPauseWhenRuntimeBusyAutoURL(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "0")
	if !envconfig.GgmlPauseWhenRuntimeBusy() {
		t.Fatal("expected auto on when runtime URL set")
	}
}

func TestGgmlPauseWhenRuntimeBusyAutoEmbed(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "")
	if !envconfig.GgmlPauseWhenRuntimeBusy() {
		t.Fatal("expected auto on with default embedded runtime")
	}
}
