package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
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

// TestFetchScoreRowTokensParallelAndCached pins the System One tokenize fix: every
// tokenization still happens (results in row / candidate order), and candidate-only
// tokenizations are cached so a repeat request skips them.
func TestFetchScoreRowTokensParallelAndCached(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var req struct {
			Content string `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
		}
		ids := []int{}
		for _, b := range []byte(req.Content) {
			ids = append(ids, int(b))
		}
		json.NewEncoder(w).Encode(map[string][]int{"tokens": ids})
	}))
	defer srv.Close()
	port, _ := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
	s := &llamaServerRunner{port: port}

	rows := []ScoreRow{{Prompt: "Q1:", Candidates: []string{"A", "B", "C"}}, {Prompt: "Q2:", Candidates: []string{"A", "B", "C"}}}
	got, err := s.fetchScoreRowTokens(t.Context(), rows)
	if err != nil {
		t.Fatal(err)
	}
	// 2 prompts + 6 joined + 3 distinct literals (rows share labels; concurrent first
	// fetches of the same label may each miss, so allow up to 6).
	if n := calls.Load(); n < 11 || n > 14 {
		t.Fatalf("cold: %d /tokenize calls, want 11–14", n)
	}
	for i, row := range rows {
		if string(rune(got[i].prompt[len(got[i].prompt)-1])) != ":" {
			t.Fatalf("row %d prompt tokens %v", i, got[i].prompt)
		}
		for j, c := range row.Candidates {
			if got[i].literal[j][0] != int(c[0]) || got[i].joined[j][len(got[i].joined[j])-1] != int(c[0]) {
				t.Fatalf("row %d cand %q: literal %v joined %v", i, c, got[i].literal[j], got[i].joined[j])
			}
		}
	}

	calls.Store(0)
	if _, err := s.fetchScoreRowTokens(t.Context(), rows); err != nil {
		t.Fatal(err)
	}
	if n := calls.Load(); n != 8 {
		t.Fatalf("warm: %d /tokenize calls, want 8 (labels cached)", n)
	}
}
