package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRuntimeChatProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/chat" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"message":         map[string]any{"role": "assistant", "content": "chat via runtime"},
				"done":            true,
				"kv_decode_steps": 7,
				"vram_num_ctx": map[string]any{
					"num_ctx": 8192,
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/api/chat", s.runtimeChatProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "should not reach"})
	})

	body := `{"model":"llama3.2","messages":[{"role":"user","content":"hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	msg, ok := resp["message"].(map[string]any)
	if !ok || msg["content"] != "chat via runtime" {
		t.Fatalf("message %v", resp["message"])
	}
	if resp["kv_decode_steps"] != float64(7) {
		t.Fatalf("kv_decode_steps %v", resp["kv_decode_steps"])
	}
	vram, ok := resp["vram_num_ctx"].(map[string]any)
	if !ok || vram["num_ctx"] != float64(8192) {
		t.Fatalf("vram_num_ctx %v", resp["vram_num_ctx"])
	}
}
