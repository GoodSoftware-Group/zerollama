package server

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

type mockReranker struct {
	mockLlm
	resp llm.RerankResponse
	err  error
}

func (m *mockReranker) Rerank(ctx context.Context, req llm.RerankRequest) (llm.RerankResponse, error) {
	return m.resp, m.err
}

func TestRerankHandlerValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}

	t.Run("missing body", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/rerank", nil)
		s.RerankHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing model", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"query":"q","documents":["a"]}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/rerank", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.RerankHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing documents", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"model":"m","query":"q"}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/rerank", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.RerankHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing query", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		body := bytes.NewBufferString(`{"model":"m","documents":["a"]}`)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/rerank", body)
		c.Request.Header.Set("Content-Type", "application/json")
		s.RerankHandler(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})
}

func TestMockRerankerImplementsReranker(t *testing.T) {
	var _ llm.Reranker = (*mockReranker)(nil)
	_ = api.RerankResponse{}
}
