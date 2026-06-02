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
	"github.com/ollama/ollama/envconfig"
)

func TestRuntimeGenerateProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model":       "test",
				"response":    "hello from runtime",
				"done":        true,
				"done_reason": "stop",
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/api/generate", s.runtimeGenerateProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "should not reach"})
	})

	body := `{"model":"llama3.2","prompt":"hi","stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	var resp api.GenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Response != "hello from runtime" {
		t.Fatalf("response %q", resp.Response)
	}
}

func TestRuntimeURLUnset(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	if envconfig.RuntimeURL() != "" {
		t.Fatal("expected empty runtime url")
	}
	_ = io.Discard
}
