package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

func TestRuntimeChatProxyForwardsToolsAndOptions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var got map[string]any
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"role": "assistant", "content": "ok"},
			"done":    true,
		})
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/api/chat", s.runtimeChatProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "legacy"})
	})

	body := `{
		"model":"m",
		"messages":[{"role":"user","content":"hi"}],
		"stream":false,
		"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}],
		"options":{"num_ctx":4096,"gguf":"/data/tools.gguf"}
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
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
	if opts["gguf"] != "/data/tools.gguf" {
		t.Fatalf("gguf=%v", opts["gguf"])
	}
	if opts["num_ctx"] != float64(4096) && opts["num_ctx"] != 4096 {
		t.Fatalf("num_ctx=%v", opts["num_ctx"])
	}
	_ = api.Message{}
}
