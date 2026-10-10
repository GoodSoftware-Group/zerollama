package llm

import (
	"strings"
	"testing"
	"time"
)

func TestStatusWriterIgnoresInformationalMLXOptionalSymbol(t *testing.T) {
	status := NewStatusWriter(nil)
	if _, err := status.Write([]byte("MLX: optional symbol mlx_array_detach missing from libmlxc (no-op)\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if got := status.LastError(); got != "" {
		t.Fatalf("LastError = %q, want empty", got)
	}
}

func TestStatusWriterCapturesRealMLXError(t *testing.T) {
	status := NewStatusWriter(nil)
	if _, err := status.Write([]byte("MLX: Failed to load symbol: mlx_array_free\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got := status.LastError()
	if got == "" || !strings.Contains(got, "mlx_array_free") {
		t.Fatalf("LastError = %q, want MLX load failure", got)
	}
}

func TestStatusWriterIgnoresUMAAutoUngated(t *testing.T) {
	status := NewStatusWriter(nil)
	line := "uma_mlx: auto — broker not running, MLX ungated (socket /tmp/uma_daemon.sock)\n"
	if _, err := status.Write([]byte(line)); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if got := status.LastError(); got != "" {
		t.Fatalf("LastError = %q, want empty (UMA optional / ungated is not a failure)", got)
	}
}

func TestStatusWriterIgnoresUMAPrefixInsideIdent(t *testing.T) {
	// Regression: strings.Index("uma_mlx: …", "mlx:") matched mid-token and
	// surfaced as timeout waiting for mlx runner: mlx: auto — broker not running…
	status := NewStatusWriter(nil)
	if _, err := status.Write([]byte("prefix uma_mlx: connected mode=1\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if got := status.LastError(); got != "" {
		t.Fatalf("LastError = %q, want empty", got)
	}
}

func TestStatusWriterLastWriteTracksProgress(t *testing.T) {
	status := NewStatusWriter(nil)
	if !status.LastWrite().IsZero() {
		t.Fatalf("expected zero LastWrite before any Write")
	}
	before := time.Now()
	if _, err := status.Write([]byte("mlx materialize eval done=32 total=100\n")); err != nil {
		t.Fatal(err)
	}
	got := status.LastWrite()
	if got.Before(before) {
		t.Fatalf("LastWrite=%v want after %v", got, before)
	}
}

func TestStatusWriterStillCapturesBareMLXPrefix(t *testing.T) {
	status := NewStatusWriter(nil)
	if _, err := status.Write([]byte("mlx: failed to map weights\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got := status.LastError()
	if got != "mlx: failed to map weights" {
		t.Fatalf("LastError = %q, want bare mlx: failure", got)
	}
}
