package llm

import (
	"context"
	"encoding/json"
)

// DecisionsRequest is a Jev-shaped typed-decision batch (Laya System-1).
type DecisionsRequest struct {
	Model     string                      `json:"model,omitempty"`
	State     any                         `json:"state"`
	Questions map[string]DecisionQuestion `json:"questions"`
}

// DecisionQuestion is one choice / score / noul question.
type DecisionQuestion struct {
	Type         string            `json:"type"` // choice|score|noul
	Instructions string            `json:"instructions"`
	// Criteria is a JSON object (choice/noul) or array (score). RawMessage preserves
	// key order for choice options (required for logit ↔ label alignment).
	Criteria json.RawMessage   `json:"criteria,omitempty"`
	Labels   map[string]string `json:"labels,omitempty"` // noul only
}

// DecisionAction is the act_head output attached to every answer.
type DecisionAction struct {
	ActProbability float64 `json:"act_probability"`
}

// ChoiceAnswer is a discrete multi-option decision.
type ChoiceAnswer struct {
	Type          string             `json:"type"` // "choice"
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
	Action        DecisionAction     `json:"action"`
}

// ScoreAnswer is an expected-value score over ordered levels.
type ScoreAnswer struct {
	Type          string             `json:"type"` // "score"
	Score         float64            `json:"score"`
	Legend        map[string]string  `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
	Action        DecisionAction     `json:"action"`
}

// NoulAnswer is P(true) for a binary true/false question (noul = "not only until later").
type NoulAnswer struct {
	Type       string         `json:"type"` // "noul"
	Noul       float64        `json:"noul"`
	Confidence float64        `json:"confidence"`
	Action     DecisionAction `json:"action"`
}

// DecisionsResponse is the calibrated typed answers for one state.
type DecisionsResponse struct {
	Model   string                     `json:"model,omitempty"`
	Answers map[string]json.RawMessage `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// Decider evaluates typed System-1 questions in one forward pass (non-autoregressive).
type Decider interface {
	Decisions(ctx context.Context, req DecisionsRequest) (DecisionsResponse, error)
}
