//go:build !edge

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
)

func (s *llmServer) Score(ctx context.Context, req ScoreRequest) (ScoreResponse, error) {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return ScoreResponse{}, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return ScoreResponse{}, err
	}
	if status != ServerStatusReady {
		return ScoreResponse{}, fmt.Errorf("unexpected server status: %s", status)
	}

	data, err := json.Marshal(req)
	if err != nil {
		return ScoreResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/score", s.port), bytes.NewReader(data))
	if err != nil {
		return ScoreResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return ScoreResponse{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ScoreResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return ScoreResponse{}, api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: string(body)}
	}

	var out ScoreResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return ScoreResponse{}, err
	}
	return out, nil
}
