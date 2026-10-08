package llm

import (
	"context"

	"github.com/ollama/ollama/api"
)

// Scorer evaluates candidate continuations or System One score rows.
type Scorer interface {
	Score(ctx context.Context, req ScoreRequest) (ScoreResponse, error)
}

// ScoreRequest scores fixed candidate continuations (/api/score) or carries
// model-native System One inputs (tev1 / nimble / clef / strands).
type ScoreRequest struct {
	// Legacy /api/score — one shared prompt and string candidates.
	Prompt               string   `json:"prompt,omitempty"`
	Candidates           []string `json:"candidates,omitempty"`
	LengthNormalize      bool     `json:"length_normalize,omitempty"`
	IncludeTokenLogprobs bool     `json:"include_token_logprobs,omitempty"`

	// System One — see decision.Compile and upstream llm/score.go.
	PointerRows   []ScorePointerRow `json:"pointer_rows,omitempty"`
	State         string            `json:"state,omitempty"`
	Rows          []ScoreRow        `json:"rows,omitempty"`
	Segments      []string          `json:"segments,omitempty"`
	Fields        []ScoreField      `json:"fields,omitempty"`
	MaxTokens     int               `json:"max_tokens"`
	Images        []api.ImageData   `json:"images,omitempty"`
	ImagePosition int               `json:"image_position,omitempty"`
}

// ScoreRow is one tev1/nimble prompt row with single-token candidates.
type ScoreRow struct {
	Prompt     string         `json:"prompt"`
	Candidates []string       `json:"candidates"`
	Question   *ScoreQuestion `json:"question,omitempty"`
}

// ScoreQuestion preserves a field schema for model-native decision heads.
type ScoreQuestion struct {
	Type         string   `json:"type"`
	Instructions string   `json:"instructions"`
	Options      []string `json:"options"`
}

// ScoreField identifies a Clef question and option spans in Segments.
// Type is 0 noul, 1 choice, 2 score.
type ScoreField struct {
	Type     int      `json:"type"`
	Question [2]int   `json:"question"`
	Options  [][2]int `json:"options"`
}

// ScorePointerRow is a Strands pointer-head row (MLX path; GGUF may reject).
type ScorePointerRow struct {
	Prefix  string   `json:"prefix"`
	Prompt  string   `json:"prompt"`
	Type    int      `json:"type"`
	Options [][2]int `json:"options"`
}

// CandidateScore is one scored continuation (/api/score).
type CandidateScore struct {
	Candidate               string         `json:"candidate"`
	LogProb                 float64        `json:"log_prob"`
	LengthNormalizedLogProb float64        `json:"length_normalized_log_prob,omitempty"`
	NumTokens               int            `json:"num_tokens"`
	Tokens                  []TokenLogprob `json:"tokens,omitempty"`
}

// ScoreResponse is returned by POST /api/score or System One scoring.
type ScoreResponse struct {
	Model      string           `json:"model,omitempty"`
	Candidates []CandidateScore `json:"candidates,omitempty"`
	// Logits may be log-probabilities; a shared row offset does not change softmax.
	Logits       [][]float32 `json:"logits,omitempty"`
	InputTokens  int         `json:"input_tokens,omitempty"`
	OutputTokens int         `json:"output_tokens,omitempty"`
	CachedTokens *int        `json:"cached_tokens,omitempty"`
}

// ScoreUsesSystemOnePath reports whether req targets the compile→score pipeline.
func ScoreUsesSystemOnePath(req ScoreRequest) bool {
	return len(req.Rows) > 0 || len(req.Segments) > 0 || len(req.Fields) > 0 || len(req.PointerRows) > 0
}
