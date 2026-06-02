package runtimeclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ErrUnavailable is returned when the runtime sidecar cannot tokenize (not configured or unreachable).
var ErrUnavailable = errors.New("runtime tokenize unavailable")

// Tokenize calls Python /internal/tokenize (libllama vocab-only).
// ok is false only when runtime URL or gguf path is empty (caller should not invoke).
func Tokenize(ctx context.Context, gguf, text string, addSpecial bool) (tokens []int, ok bool, err error) {
	base := runtimeBaseURL()
	gguf = strings.TrimSpace(gguf)
	if base == "" || gguf == "" {
		return nil, false, nil
	}
	body, err := json.Marshal(map[string]any{
		"gguf":         gguf,
		"text":         text,
		"add_special":  addSpecial,
	})
	if err != nil {
		return nil, false, err
	}
	url := strings.TrimSuffix(base, "/") + "/internal/tokenize"
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, url, bytes.NewReader(body),
	)
	if err != nil {
		return nil, false, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := handoffClient.Do(req)
	if err != nil {
		return nil, false, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false, err
	}
	if resp.StatusCode >= 300 {
		return nil, false, fmt.Errorf("runtime tokenize: HTTP %d: %s", resp.StatusCode, string(b))
	}
	var out struct {
		Tokens []int `json:"tokens"`
		Count  int   `json:"count"`
	}
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, false, err
	}
	if out.Tokens != nil {
		return out.Tokens, true, nil
	}
	return nil, true, nil
}
