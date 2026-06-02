package server

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRuntimeGenerateStreamProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"m","response":"a","done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"m","response":"","done":true,"done_reason":"stop"}` + "\n"))
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/api/generate", s.runtimeGenerateProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "legacy"})
	})

	body := `{"model":"m","prompt":"hi","stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "ndjson") {
		t.Fatalf("content-type %q", ct)
	}
	sc := bufio.NewScanner(w.Body)
	var lines int
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatal(err)
		}
		lines++
	}
	if lines != 2 {
		t.Fatalf("expected 2 ndjson lines, got %d", lines)
	}
}

func TestRuntimeGenerateStreamDefaultNil(t *testing.T) {
	gin.SetMode(gin.TestMode)
	called := false
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/internal/inference/resume":
			w.WriteHeader(http.StatusOK)
			return
		case "/api/generate":
			called = true
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			if body["stream"] != true {
				t.Errorf("stream=%v", body["stream"])
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = io.WriteString(w, `{"done":true}`+"\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	r.POST("/api/generate", s.runtimeGenerateProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "legacy"})
	})

	body := `{"model":"m","prompt":"hi"}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if !called {
		t.Fatal("runtime not called for nil stream")
	}
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
}
