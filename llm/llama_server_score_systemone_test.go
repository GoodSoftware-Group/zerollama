package llm

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestScoreSystemOneRejectsBadPointerRows(t *testing.T) {
	s := &llamaServerRunner{}
	_, err := s.scoreSystemOne(context.Background(), ScoreRequest{
		PointerRows: []ScorePointerRow{{Prompt: "x", Options: [][2]int{{0, 1}}}},
		MaxTokens:   16,
	})
	if err == nil {
		t.Fatal("expected error for invalid pointer_rows")
	}
	se, ok := err.(api.StatusError)
	if !ok {
		t.Fatalf("got %T %v, want api.StatusError", err, err)
	}
	if se.StatusCode != 400 {
		t.Fatalf("status=%d want 400", se.StatusCode)
	}
	if !strings.Contains(strings.ToLower(se.ErrorMessage), "invalid") {
		t.Fatalf("error=%q want mention of invalid", se.ErrorMessage)
	}
}

func TestGgufNeedsDecisionPoolingNone(t *testing.T) {
	if !ggufNeedsDecisionPoolingNone(map[string]any{
		"general.architecture": "qwen35",
		"qwen35.decision.type": "strands",
	}) {
		t.Fatal("strands should need pooling none")
	}
	if !ggufNeedsDecisionPoolingNone(map[string]any{
		"general.architecture": "qwen35",
		"qwen35.decision.type": "clef",
	}) {
		t.Fatal("clef should need pooling none")
	}
	if ggufNeedsDecisionPoolingNone(map[string]any{"general.architecture": "qwen35"}) {
		t.Fatal("plain qwen35 should not")
	}
}
