package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRuntimeChatProxyStream(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"hi"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":""},"done":true}` + "\n"))
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

	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"done":true`) {
		t.Fatalf("body %q", w.Body.String())
	}
}
