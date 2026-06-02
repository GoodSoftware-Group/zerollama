// Package runtimeclient calls the optional zerollama Python inference sidecar.
package runtimeclient

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/runtimeworker"
)

var (
	handoffClient          = &http.Client{Timeout: 30 * time.Second}
	goCoordPushWarnOnce    sync.Once
)

func runtimeBaseURL() string {
	if u := strings.TrimSpace(runtimeworker.BaseURL()); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return envconfig.RuntimeURL()
}

// TrainingHandoff asks the Python runtime to release GPU memory (stops llama-server).
// Why called from training OOM handler: after Go evicts ggml runners, Python subprocess
// may still hold LLAMA_MODEL weights; handoff clears that VRAM before AckVRAMHeadroom.
// Operators also POST /internal/training-handoff manually; use /internal/inference/resume
// to run runtime generate again (see docs/testing-smoke.md).
func TrainingHandoff(ctx context.Context) {
	base := runtimeBaseURL()
	if base == "" {
		return
	}
	url := strings.TrimSuffix(base, "/") + "/internal/training-handoff"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		slog.Debug("runtime handoff: build request", "error", err)
		return
	}
	resp, err := handoffClient.Do(req)
	if err != nil {
		slog.Warn("runtime handoff: request failed", "error", err)
		return
	}
	resp.Body.Close()
	if resp.StatusCode >= 300 {
		slog.Warn("runtime handoff: unexpected status", "status", resp.StatusCode)
	}
}

// PushGoCoordination updates Python /health with Go-side training defer and policy flags.
func PushGoCoordination(ctx context.Context, snap map[string]any) {
	base := runtimeBaseURL()
	if base == "" || snap == nil {
		return
	}
	body, err := json.Marshal(snap)
	if err != nil {
		return
	}
	url := strings.TrimSuffix(base, "/") + "/internal/go-coordination"
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, url, strings.NewReader(string(body)),
	)
	if err != nil {
		slog.Debug("runtime go-coordination: build request", "error", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := handoffClient.Do(req)
	if err != nil {
		goCoordPushWarnOnce.Do(func() {
			slog.Warn(
				"runtime go-coordination: push failed (defer admission mirror may go stale)",
				"error", err,
			)
		})
		return
	}
	resp.Body.Close()
	if resp.StatusCode >= 300 {
		goCoordPushWarnOnce.Do(func() {
			slog.Warn(
				"runtime go-coordination: unexpected status (defer admission mirror may go stale)",
				"status", resp.StatusCode,
			)
		})
	}
}

// SetTrainingGPUBusy tells the Python runtime whether Go training currently occupies the GPU.
// Used for VRAM admission reserve while the runtime may still be RUNNING (before handoff).
func SetTrainingGPUBusy(ctx context.Context, busy bool) {
	base := runtimeBaseURL()
	if base == "" {
		return
	}
	url := strings.TrimSuffix(base, "/") + "/internal/training-gpu-busy"
	body := `{"busy":false}`
	if busy {
		body = `{"busy":true}`
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, url, strings.NewReader(body),
	)
	if err != nil {
		slog.Debug("runtime training-gpu-busy: build request", "error", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := handoffClient.Do(req)
	if err != nil {
		slog.Debug("runtime training-gpu-busy: request failed", "error", err)
		return
	}
	resp.Body.Close()
	if resp.StatusCode >= 300 {
		slog.Warn("runtime training-gpu-busy: unexpected status", "status", resp.StatusCode)
	}
}

// ResumeInference asks the Python runtime to accept inference again after training-handoff.
func ResumeInference(ctx context.Context) {
	base := runtimeBaseURL()
	if base == "" {
		return
	}
	url := strings.TrimSuffix(base, "/") + "/internal/inference/resume"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		slog.Debug("runtime resume: build request", "error", err)
		return
	}
	resp, err := handoffClient.Do(req)
	if err != nil {
		slog.Warn("runtime resume: request failed", "error", err)
		return
	}
	resp.Body.Close()
	if resp.StatusCode >= 300 {
		slog.Warn("runtime resume: unexpected status", "status", resp.StatusCode)
	}
}
