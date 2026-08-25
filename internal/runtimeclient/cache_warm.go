package runtimeclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// CacheWarmResult is the decoded response from POST /internal/cache/warm.
type CacheWarmResult struct {
	Warmed         bool           `json:"warmed"`
	PromptCacheKey string         `json:"prompt_cache_key"`
	RequestID      string         `json:"request_id"`
	KVDecodeSteps  *int           `json:"kv_decode_steps"`
	VRAMNumCtx     map[string]any `json:"vram_num_ctx"`
	Pin            map[string]any `json:"pin"`
}

// warmClient uses a longer timeout than the fire-and-forget handoff calls since
// warm runs a real prefill (llama_decode over the whole prompt) synchronously.
var warmClient = &http.Client{Timeout: 120 * time.Second}

// CacheWarm asks the Python runtime to prefill and pin an L3 slot for
// prompt_cache_key without generating a real completion. Returns an error when
// the runtime sidecar is unreachable or rejects the request.
func CacheWarm(
	ctx context.Context,
	prompt, promptCacheKey, gguf string,
	numCtx *int,
	pinID string,
	expiresAt *time.Time,
	options map[string]any,
) (*CacheWarmResult, error) {
	base := runtimeBaseURL()
	if base == "" {
		return nil, fmt.Errorf("runtime sidecar not configured")
	}
	if strings.TrimSpace(promptCacheKey) == "" {
		return nil, fmt.Errorf("prompt_cache_key is required")
	}

	payload := map[string]any{
		"prompt_cache_key": promptCacheKey,
		"prompt":           prompt,
	}
	if gguf != "" {
		payload["gguf"] = gguf
	}
	if numCtx != nil {
		payload["num_ctx"] = *numCtx
	}
	if pinID != "" {
		payload["pin_id"] = pinID
	}
	if expiresAt != nil {
		payload["expires_at"] = expiresAt.UTC().Format(time.RFC3339Nano)
	}
	if len(options) > 0 {
		payload["options"] = options
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	url := strings.TrimSuffix(base, "/") + "/internal/cache/warm"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := warmClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("runtime cache warm request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("runtime cache warm: status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}
	var out CacheWarmResult
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("runtime cache warm: decode response: %w", err)
	}
	return &out, nil
}
