package llm

import "context"

// ScoreRequest scores fixed candidate continuations against a shared prompt.
// Why: agent routing / classifier models (LocalAI Score RPC pattern) without generation.
type ScoreRequest struct {
	Prompt               string   `json:"prompt"`
	Candidates           []string `json:"candidates"`
	LengthNormalize      bool     `json:"length_normalize,omitempty"`
	IncludeTokenLogprobs bool     `json:"include_token_logprobs,omitempty"`
}

// CandidateScore is one scored continuation.
type CandidateScore struct {
	Candidate               string           `json:"candidate"`
	LogProb                 float64          `json:"log_prob"`
	LengthNormalizedLogProb float64          `json:"length_normalized_log_prob,omitempty"`
	NumTokens               int              `json:"num_tokens"`
	Tokens                  []TokenLogprob   `json:"tokens,omitempty"`
}

// ScoreResponse is returned by runner POST /score and server POST /api/score.
type ScoreResponse struct {
	Model      string           `json:"model,omitempty"`
	Candidates []CandidateScore `json:"candidates"`
}

// Scorer evaluates candidate continuations without full generation.
type Scorer interface {
	Score(ctx context.Context, req ScoreRequest) (ScoreResponse, error)
}
