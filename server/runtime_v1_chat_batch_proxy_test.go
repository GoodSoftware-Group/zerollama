package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRuntimeV1ChatBatchProxy_RejectsMixedModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/v1/chat/completions/batch", s.runtimeV1ChatCompletionsBatchProxy())

	body, _ := json.Marshal(map[string]any{
		"requests": []any{
			map[string]any{"model": "a", "messages": []any{map[string]any{"role": "user", "content": "1"}}},
			map[string]any{"model": "b", "messages": []any{map[string]any{"role": "user", "content": "2"}}},
		},
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	if !bytes.Contains(w.Body.Bytes(), []byte("same model")) {
		t.Fatalf("body=%s", w.Body.String())
	}
}

func TestRuntimeV1ChatBatchProxy_RejectsTools(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/v1/chat/completions/batch", s.runtimeV1ChatCompletionsBatchProxy())

	body, _ := json.Marshal(map[string]any{
		"model": "m",
		"requests": []any{
			map[string]any{
				"messages": []any{map[string]any{"role": "user", "content": "1"}},
				"tools":    []any{map[string]any{"type": "function"}},
			},
		},
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}
