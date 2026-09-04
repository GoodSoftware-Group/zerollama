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

func (s *llamaServerRunner) Rerank(ctx context.Context, req RerankRequest) (RerankResponse, error) {
	if len(req.Documents) == 0 {
		return RerankResponse{}, fmt.Errorf("documents required")
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return RerankResponse{}, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return RerankResponse{}, err
	}
	if status != ServerStatusReady {
		return RerankResponse{}, fmt.Errorf("unexpected server status: %s", status)
	}

	body := map[string]any{
		"query":     req.Query,
		"documents": req.Documents,
	}
	if req.TopN > 0 {
		body["top_n"] = req.TopN
	}
	data, err := json.Marshal(body)
	if err != nil {
		return RerankResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/rerank", s.port), bytes.NewReader(data))
	if err != nil {
		return RerankResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(httpReq)
	if err != nil {
		return RerankResponse{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return RerankResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return RerankResponse{}, api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: string(raw)}
	}

	var parsed struct {
		Model   string `json:"model"`
		Results []struct {
			Index          int     `json:"index"`
			RelevanceScore float64 `json:"relevance_score"`
			Score          float64 `json:"score"`
		} `json:"results"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return RerankResponse{}, fmt.Errorf("decode rerank: %w", err)
	}

	out := RerankResponse{
		Model:        parsed.Model,
		PromptTokens: parsed.Usage.PromptTokens,
		TotalTokens:  parsed.Usage.TotalTokens,
	}
	for _, r := range parsed.Results {
		score := r.RelevanceScore
		if score == 0 && r.Score != 0 {
			score = r.Score
		}
		out.Results = append(out.Results, RerankHit{Index: r.Index, RelevanceScore: score})
	}
	return out, nil
}
