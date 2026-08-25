package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
)

func TestWrapRunnerErrPrefersContextCancelOverStaleStatus(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	time.Sleep(2 * time.Millisecond)

	status := llm.NewStatusWriter(nil)
	status.SetLastError("MLX: optional symbol mlx_array_detach missing from libmlxc (no-op)")

	err := wrapRunnerErr(ctx, errors.New("connection reset"), status)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestWrapRunnerErrUsesStatusWhenContextAlive(t *testing.T) {
	status := llm.NewStatusWriter(nil)
	status.SetLastError("MLX: Failed to load symbol: mlx_array_free")

	err := wrapRunnerErr(context.Background(), errors.New("connection reset"), status)
	if err == nil || !strings.Contains(err.Error(), "mlx_array_free") {
		t.Fatalf("err = %v, want status detail", err)
	}
}

func TestCompletionDecodesResponseLargerThanScannerLimit(t *testing.T) {
	const tokenCount = 40_000
	tokens := make([]int32, tokenCount)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewEncoder(w).Encode(CompletionResponse{Done: true, Tokens: tokens}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	defer server.Close()

	port := server.Listener.Addr().(*net.TCPAddr).Port
	client := &Client{
		port:   port,
		client: server.Client(),
		status: llm.NewStatusWriter(nil),
	}

	var got llm.CompletionResponse
	if err := client.Completion(context.Background(), llm.CompletionRequest{}, func(response llm.CompletionResponse) {
		got = response
	}); err != nil {
		t.Fatalf("Completion() error = %v", err)
	}
	if len(got.Tokens) != tokenCount {
		t.Fatalf("got %d tokens, want %d", len(got.Tokens), tokenCount)
	}
}
