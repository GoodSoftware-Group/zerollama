package runtimeclient

import (
	"bytes"
	"log/slog"
	"testing"
)

func TestLogVramBudgetFromSnap_onlyHostTight(t *testing.T) {
	var buf bytes.Buffer
	h := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})
	old := slog.Default()
	slog.SetDefault(slog.New(h))
	defer slog.SetDefault(old)

	logVramBudgetFromSnap("m", "/tmp/m.gguf", map[string]any{
		"vram_budget": map[string]any{
			"host_ram": map[string]any{"fits": false},
		},
	})
	if buf.Len() == 0 {
		t.Fatal("expected log when host_ram tight")
	}
}

func TestLogVramBudgetFromSnap_missingGpuFieldNoLog(t *testing.T) {
	var buf bytes.Buffer
	h := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})
	old := slog.Default()
	slog.SetDefault(slog.New(h))
	defer slog.SetDefault(old)

	logVramBudgetFromSnap("m", "/tmp/m.gguf", map[string]any{
		"vram_budget": map[string]any{
			"host_ram": map[string]any{"fits": true},
		},
	})
	if buf.Len() != 0 {
		t.Fatalf("unexpected log: %s", buf.String())
	}
}
