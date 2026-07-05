package llm

import (
	"strings"
	"testing"
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
