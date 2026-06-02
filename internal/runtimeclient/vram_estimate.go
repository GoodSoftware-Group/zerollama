package runtimeclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// ProbeVramEstimate calls Python /internal/vram-estimate (best-effort).
// Returns the JSON body or nil when the runtime is unavailable.
func ProbeVramEstimate(
	ctx context.Context,
	gguf string,
	opts map[string]any,
) map[string]any {
	base := runtimeBaseURL()
	if base == "" || strings.TrimSpace(gguf) == "" {
		return nil
	}
	body := map[string]any{"gguf": gguf}
	if opts != nil {
		body["options"] = opts
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil
	}
	url := strings.TrimSuffix(base, "/") + "/internal/vram-estimate"
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, url, bytes.NewReader(raw),
	)
	if err != nil {
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := handoffClient.Do(req)
	if err != nil {
		slog.Debug("runtime vram-estimate: request failed", "error", err)
		return nil
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode >= 300 {
		if resp.StatusCode >= 300 {
			slog.Debug("runtime vram-estimate: unexpected status", "status", resp.StatusCode)
		}
		return nil
	}
	var out map[string]any
	if json.Unmarshal(b, &out) != nil {
		return nil
	}
	return out
}

func logVramBudgetFromSnap(model, gguf string, snap map[string]any) {
	budget, _ := snap["vram_budget"].(map[string]any)
	if budget == nil {
		return
	}
	var gpuTight, hostTight bool
	if v, ok := budget["fits_with_margin"].(bool); ok && !v {
		gpuTight = true
	}
	if host, ok := budget["host_ram"].(map[string]any); ok {
		if fits, ok := host["fits"].(bool); ok && !fits {
			hostTight = true
		}
	}
	if !gpuTight && !hostTight {
		return
	}
	slog.Info(
		"runtime proxy: load budget tight",
		"model", model,
		"gguf", gguf,
		"gpu_tight", gpuTight,
		"host_ram_tight", hostTight,
	)
}

// LogVramBudgetIfTight probes the runtime in the background and logs when
// GPU fits_with_margin or host_ram fits is false (best-effort, non-blocking).
func LogVramBudgetIfTight(ctx context.Context, model string, gguf string, opts map[string]any) {
	if strings.TrimSpace(gguf) == "" {
		return
	}
	go func() {
		snap := ProbeVramEstimate(context.Background(), gguf, opts)
		if snap == nil {
			return
		}
		logVramBudgetFromSnap(model, gguf, snap)
	}()
}
