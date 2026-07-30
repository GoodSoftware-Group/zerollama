package runtimeclient

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// NotifyCachePin best-effort POSTs /internal/cache/pin so Python L3 extends disk TTL.
func NotifyCachePin(ctx context.Context, pinID, promptCacheKey string, expiresAt time.Time) {
	base := runtimeBaseURL()
	if base == "" || strings.TrimSpace(pinID) == "" || strings.TrimSpace(promptCacheKey) == "" {
		return
	}
	body, err := json.Marshal(map[string]any{
		"pin_id":           pinID,
		"prompt_cache_key": promptCacheKey,
		"expires_at":       expiresAt.UTC().Format(time.RFC3339Nano),
	})
	if err != nil {
		return
	}
	url := strings.TrimSuffix(base, "/") + "/internal/cache/pin"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := handoffClient.Do(req)
	if err != nil {
		slog.Debug("runtime cache pin notify failed", "error", err)
		return
	}
	resp.Body.Close()
}

// NotifyCacheUnpin best-effort POSTs /internal/cache/unpin.
func NotifyCacheUnpin(ctx context.Context, pinID, promptCacheKey string) {
	base := runtimeBaseURL()
	if base == "" {
		return
	}
	body, err := json.Marshal(map[string]any{
		"pin_id":           pinID,
		"prompt_cache_key": promptCacheKey,
	})
	if err != nil {
		return
	}
	url := strings.TrimSuffix(base, "/") + "/internal/cache/unpin"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := handoffClient.Do(req)
	if err != nil {
		slog.Debug("runtime cache unpin notify failed", "error", err)
		return
	}
	resp.Body.Close()
}
