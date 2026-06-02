package server

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func TestToolParseSessionRegistry_templateParser(t *testing.T) {
	t.Parallel()
	tpl, err := template.Parse(`{{if .ToolCalls}}{{end}}`)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Template: tpl}
	tools := api.Tools{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{Type: "object"},
			},
		},
	}
	id, method, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if method != "template" {
		t.Fatalf("method %q", method)
	}
	_, _, _, methodOut, err := toolParseSessions.add(
		id,
		`{"name":"get_weather","arguments":{"city":"Paris"}}`,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if methodOut != "template" {
		t.Fatalf("chunk method %q", methodOut)
	}
	// session closed on done
	_, _, _, _, err = toolParseSessions.add(id, "x", false)
	if err != errSessionNotFound {
		t.Fatalf("expected session gone, got %v", err)
	}
}

func TestToolParseSessionChunkHandler_notFound(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/chunk", s.ToolParseSessionChunkHandler)
	body := bytes.NewBufferString(`{"session_id":"nope","content":"x"}`)
	req := httptest.NewRequest(http.MethodPost, "/chunk", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status %d", w.Code)
	}
}
