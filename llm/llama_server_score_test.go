package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sync/semaphore"
)

func TestLlamaServerTokenLogprob(t *testing.T) {
	require.Equal(t, -1.5, llamaServerTokenLogprob(llamaServerTokenProb{Logprob: -1.5}))
	require.InDelta(t, math.Log(0.5), llamaServerTokenLogprob(llamaServerTokenProb{Prob: 0.5}), 1e-9)
	require.Equal(t, 0.0, llamaServerTokenLogprob(llamaServerTokenProb{}))
}

func TestLlamaServerLogprobForToken(t *testing.T) {
	probs := []llamaServerTokenProb{{
		ID:      42,
		Token:   "yes",
		Logprob: -0.1,
		TopLogprobs: []llamaServerTokenProb{
			{ID: 7, Token: "no", Logprob: -2.0},
		},
		TopProbs: []llamaServerTokenProb{
			{ID: 9, Token: "maybe", Prob: 0.25},
		},
	}}

	lp, tok, ok := llamaServerLogprobForToken(probs, 42)
	require.True(t, ok)
	require.Equal(t, "yes", tok)
	require.Equal(t, -0.1, lp)

	lp, tok, ok = llamaServerLogprobForToken(probs, 7)
	require.True(t, ok)
	require.Equal(t, "no", tok)
	require.Equal(t, -2.0, lp)

	lp, tok, ok = llamaServerLogprobForToken(probs, 9)
	require.True(t, ok)
	require.Equal(t, "maybe", tok)
	require.InDelta(t, math.Log(0.25), lp, 1e-9)

	_, _, ok = llamaServerLogprobForToken(probs, 999)
	require.False(t, ok)
}

// fakeScoreServer serves /health, /tokenize (one token per byte, id = byte value) and
// /completion, whose top logprobs cover only the bytes in top. It records completion
// bodies so tests can check what went over the wire.
func fakeScoreServer(t *testing.T, top string) (*llamaServerRunner, *[]map[string]any) {
	t.Helper()
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/tokenize":
			var req struct {
				Content string `json:"content"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			ids := []int{}
			for _, b := range []byte(req.Content) {
				ids = append(ids, int(b))
			}
			json.NewEncoder(w).Encode(map[string][]int{"tokens": ids})
		case "/completion":
			var body map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
			bodies = append(bodies, body)
			tops := []map[string]any{}
			for i, b := range []byte(top) {
				tops = append(tops, map[string]any{"id": int(b), "token": string(b), "logprob": -float64(i + 1)})
			}
			json.NewEncoder(w).Encode(map[string]any{"completion_probabilities": []map[string]any{{
				"id": tops[0]["id"], "token": tops[0]["token"], "logprob": tops[0]["logprob"], "top_logprobs": tops,
			}}})
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)
	port, err := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
	require.NoError(t, err)
	return &llamaServerRunner{port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1)}, &bodies
}

// TestScoreSingleTokenCandidatesShareOneCall pins two fixes: llama-server b11351 rejects
// repeat_penalty 0 (sent on every score request before), and single-token candidates
// share the prompt's next-token distribution — one /completion call, not one each.
func TestScoreSingleTokenCandidatesShareOneCall(t *testing.T) {
	r, bodies := fakeScoreServer(t, "abc")
	resp, err := r.Score(t.Context(), ScoreRequest{Prompt: "p", Candidates: []string{"a", "b", "c"}})
	require.NoError(t, err)
	require.Len(t, *bodies, 1)
	rp, ok := (*bodies)[0]["repeat_penalty"].(float64)
	require.True(t, ok && rp > 0, "repeat_penalty = %v, want > 0", (*bodies)[0]["repeat_penalty"])
	for i, c := range resp.Candidates {
		require.Equal(t, -float64(i+1), c.LogProb, c.Candidate)
	}
}

// TestScoreAllowMissing: strict scoring still fails on a candidate outside the top
// logprobs (the /api/score contract); AllowMissing (the LA11 router) scores it -Inf.
func TestScoreAllowMissing(t *testing.T) {
	r, _ := fakeScoreServer(t, "ab")
	_, err := r.Score(t.Context(), ScoreRequest{Prompt: "p", Candidates: []string{"a", "z"}})
	require.ErrorContains(t, err, "not in top")

	resp, err := r.Score(t.Context(), ScoreRequest{Prompt: "p", Candidates: []string{"a", "z"}, AllowMissing: true})
	require.NoError(t, err)
	require.Equal(t, -1.0, resp.Candidates[0].LogProb)
	require.False(t, resp.Candidates[0].Truncated)
	require.True(t, math.IsInf(resp.Candidates[1].LogProb, -1))
	require.True(t, resp.Candidates[1].Truncated)
}
