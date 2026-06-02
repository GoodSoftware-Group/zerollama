package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

func TestRenderChatHandler_missingModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/internal/render-chat", s.RenderChatHandler)

	body := bytes.NewBufferString(`{"model":"no-such-model-xyz","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/internal/render-chat", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
}

func TestRenderChatHandler_requiresModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/internal/render-chat", s.RenderChatHandler)

	req := httptest.NewRequest(
		http.MethodPost,
		"/internal/render-chat",
		bytes.NewBufferString(`{"messages":[]}`),
	)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status %d", w.Code)
	}
}

// Ensure request type unmarshals Ollama messages.
func TestRenderChatRequestShape(t *testing.T) {
	raw := `{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
	}`
	var req renderChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
		t.Fatalf("messages %+v", req.Messages)
	}
	if len(req.Tools) != 1 || req.Tools[0].Function.Name != "f" {
		t.Fatalf("tools %+v", req.Tools)
	}
	_ = api.Message{}
}
