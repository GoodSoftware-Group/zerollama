package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRuntimeV1ChatCompletionsProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "chat.completion",
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/v1/chat/completions", s.runtimeV1ChatCompletionsProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "legacy"})
	})

	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body.String())
	}
}

func TestRuntimeV1ChatCompletionsProxyStream(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"))
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/v1/chat/completions", s.runtimeV1ChatCompletionsProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "legacy"})
	})

	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type %q", ct)
	}
	if !strings.Contains(w.Body.String(), "chat.completion.chunk") {
		t.Fatalf("body %q", w.Body.String())
	}
}

func TestRuntimeV1ChatCompletionsProxyForwardsTools(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var got map[string]any
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "chat.completion",
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/v1/chat/completions", s.runtimeV1ChatCompletionsProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "legacy"})
	})

	body := `{
		"model":"m",
		"messages":[{"role":"user","content":"hi"}],
		"stream":false,
		"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}],
		"options":{"num_ctx":4096}
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body.String())
	}
	tools, ok := got["tools"].([]any)
	if !ok || len(tools) == 0 {
		t.Fatalf("tools not forwarded: %v", got)
	}
	opts, ok := got["options"].(map[string]any)
	if !ok {
		t.Fatalf("options missing: %v", got)
	}
	if opts["num_ctx"] != float64(4096) && opts["num_ctx"] != 4096 {
		t.Fatalf("num_ctx=%v", opts["num_ctx"])
	}
}
