package llm

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestScoreSystemOneRejectsPointerRows(t *testing.T) {
	s := &llamaServerRunner{}
	_, err := s.scoreSystemOne(context.Background(), ScoreRequest{
		PointerRows: []ScorePointerRow{{Prompt: "x", Options: [][2]int{{0, 1}}}},
		MaxTokens:   16,
	})
	if err == nil {
		t.Fatal("expected error for pointer_rows on llama-server")
	}
	se, ok := err.(api.StatusError)
	if !ok {
		t.Fatalf("got %T %v, want api.StatusError", err, err)
	}
	if se.StatusCode != 400 {
		t.Fatalf("status=%d want 400", se.StatusCode)
	}
	if !strings.Contains(strings.ToLower(se.ErrorMessage), "mlx") {
		t.Fatalf("error=%q want mention of MLX", se.ErrorMessage)
	}
}
