package mlxrunner

import (
	"context"
	"errors"
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
