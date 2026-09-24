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

type mockDecider struct {
	mockLlm
	resp llm.DecisionsResponse
	err  error
	last llm.DecisionsRequest
}

func (m *mockDecider) Decisions(ctx context.Context, req llm.DecisionsRequest) (llm.DecisionsResponse, error) {
	m.last = req
	return m.resp, m.err
}

func TestDecisionsHandlerValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}

	t.Run("missing body", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/decisions", nil)
		s.DecisionsHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing model", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/decisions", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.DecisionsHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing questions", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"model":"m","state":"s"}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/decisions", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.DecisionsHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing state", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"model":"m","questions":{"q":{"type":"noul","instructions":"?"}}}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/decisions", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.DecisionsHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})
}

func TestMockDeciderImplementsDecider(t *testing.T) {
	var _ llm.Decider = (*mockDecider)(nil)
	_ = api.DecisionsResponse{}
}

func TestAPIDecisionsToLLM(t *testing.T) {
	req := api.DecisionsRequest{
		Model: "laya",
		State: "hello",
		Questions: map[string]api.DecisionQuestion{
			"q1": {
				Type:         "Choice",
				Instructions: "pick",
				Criteria:     json.RawMessage(`{"a":"x","b":"y"}`),
			},
		},
	}
	got, err := apiDecisionsToLLM(req)
	require.NoError(t, err)
	require.Equal(t, "choice", got.Questions["q1"].Type)
	require.Equal(t, "pick", got.Questions["q1"].Instructions)

	_, err = apiDecisionsToLLM(api.DecisionsRequest{
		Questions: map[string]api.DecisionQuestion{
			"bad": {Type: "chat", Instructions: "x"},
		},
	})
	require.Error(t, err)
}

func TestMockDeciderReturnsAnswers(t *testing.T) {
	ans, _ := json.Marshal(llm.NoulAnswer{
		Type:       "noul",
		Noul:       0.8,
		Confidence: 0.8,
		Action:     llm.DecisionAction{ActProbability: 0.9},
	})
	m := &mockDecider{
		resp: llm.DecisionsResponse{
			Model:   "laya",
			Answers: map[string]json.RawMessage{"q": ans},
		},
	}
	resp, err := m.Decisions(context.Background(), llm.DecisionsRequest{
		Model: "laya",
		State: "s",
		Questions: map[string]llm.DecisionQuestion{
			"q": {Type: "noul", Instructions: "?"},
		},
	})
	require.NoError(t, err)
	require.Contains(t, resp.Answers, "q")
	require.Equal(t, "laya", m.last.Model)
}
