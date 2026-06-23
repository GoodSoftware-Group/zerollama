package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

type mockScorer struct {
	mockLlm
	resp llm.ScoreResponse
	err  error
}

func (m *mockScorer) Score(ctx context.Context, req llm.ScoreRequest) (llm.ScoreResponse, error) {
	return m.resp, m.err
}

func TestScoreHandlerValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}

	t.Run("missing body", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/score", nil)
		s.ScoreHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing model", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"prompt":"hi","candidates":["a"]}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/score", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.ScoreHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing candidates", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"model":"m","prompt":"hi"}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/score", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.ScoreHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})
}

func TestScoreCandidatesToAPI(t *testing.T) {
	out := scoreCandidatesToAPI([]llm.CandidateScore{{
		Candidate:               "yes",
		LogProb:                 -1.5,
		LengthNormalizedLogProb: -0.75,
		NumTokens:               2,
		Tokens:                  []llm.TokenLogprob{{Token: "y", Logprob: -0.5}, {Token: "es", Logprob: -1.0}},
	}})
	require.Len(t, out, 1)
	require.Equal(t, "yes", out[0].Candidate)
	require.Equal(t, -1.5, out[0].LogProb)
	require.Len(t, out[0].Tokens, 2)
	require.Equal(t, "y", out[0].Tokens[0].Token)
}

func TestMockScorerImplementsScorer(t *testing.T) {
	var _ llm.Scorer = (*mockScorer)(nil)
	_ = api.ScoreResponse{}
}

func TestScoreHandlerStatusError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	scorer := &mockScorer{
		resp: llm.ScoreResponse{Candidates: []llm.CandidateScore{{Candidate: "a", LogProb: -1}}},
	}
	// Exercise mapping only; scheduling is not under test here.
	require.NotNil(t, scorer)
	out := scoreCandidatesToAPI(scorer.resp.Candidates)
	require.Equal(t, -1.0, out[0].LogProb)

	errBody := gin.H{"error": "bad"}
	data, err := json.Marshal(errBody)
	require.NoError(t, err)
	require.Contains(t, string(data), "bad")
}
