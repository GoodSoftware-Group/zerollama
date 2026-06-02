package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestInternalLoopbackOnly_rejectsRemote(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/internal/x", internalLoopbackOnly(), func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/internal/x", nil)
	req.RemoteAddr = "203.0.113.1:12345"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Fatalf("status %d", w.Code)
	}
}

func TestEnsureLoopbackGoURLEnv_mapsUnspecifiedBind(t *testing.T) {
	t.Setenv("ZEROLLAMA_GO_URL", "")
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:8080")
	ensureLoopbackGoURLEnv()
	got := os.Getenv("ZEROLLAMA_GO_URL")
	if got != "http://127.0.0.1:8080" {
		t.Fatalf("ZEROLLAMA_GO_URL=%q want http://127.0.0.1:8080", got)
	}
}

func TestEnsureLoopbackGoURLEnv_respectsExplicit(t *testing.T) {
	t.Setenv("ZEROLLAMA_GO_URL", "http://10.0.0.5:9999")
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:8080")
	ensureLoopbackGoURLEnv()
	if got := os.Getenv("ZEROLLAMA_GO_URL"); got != "http://10.0.0.5:9999" {
		t.Fatalf("expected explicit URL preserved, got %q", got)
	}
}

func TestInternalLoopbackOnly_allowsLoopback(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/internal/x", internalLoopbackOnly(), func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/internal/x", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
}
