package llm

import "context"

// RerankRequest is a Jina/llama.cpp-style query + documents score.
type RerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

// RerankHit is one document's relevance (sorted descending by llama-server).
type RerankHit struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// RerankResponse matches llama.cpp Jina `/v1/rerank` (not TEI array form).
type RerankResponse struct {
	Model        string      `json:"model,omitempty"`
	Results      []RerankHit `json:"results"`
	PromptTokens int         `json:"prompt_tokens,omitempty"`
	TotalTokens  int         `json:"total_tokens,omitempty"`
}

// Reranker scores query–document pairs (cross-encoder / RANK pooling).
type Reranker interface {
	Rerank(ctx context.Context, req RerankRequest) (RerankResponse, error)
}
