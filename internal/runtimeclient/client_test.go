package runtimeclient

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestSetTrainingGPUBusyPostsJSON(t *testing.T) {
	var got struct {
		Busy bool `json:"busy"`
	}
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/training-gpu-busy" {
			t.Fatalf("path %q", r.URL.Path)
		}
		hits.Add(1)
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	SetTrainingGPUBusy(context.Background(), true)
	if hits.Load() != 1 {
		t.Fatalf("hits %d", hits.Load())
	}
	if !got.Busy {
		t.Fatal("expected busy true")
	}
}
