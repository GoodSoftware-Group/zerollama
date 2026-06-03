package runtimeclient

import (
	"context"
	"io"
	"net/http"
	"strings"
)

// FetchKVSnapshot proxies GET /internal/kv-snapshot from the Python runtime.
func FetchKVSnapshot(ctx context.Context) ([]byte, int, error) {
	base := runtimeBaseURL()
	if base == "" {
		return nil, http.StatusServiceUnavailable, nil
	}
	url := strings.TrimSuffix(base, "/") + "/internal/kv-snapshot"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, http.StatusInternalServerError, err
	}
	resp, err := handoffClient.Do(req)
	if err != nil {
		return nil, http.StatusBadGateway, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, http.StatusBadGateway, err
	}
	return body, resp.StatusCode, nil
}
