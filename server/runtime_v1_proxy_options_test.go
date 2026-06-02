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

func TestV1NumPredictFromBody(t *testing.T) {
	n, ok := v1NumPredictFromBody(map[string]any{"max_tokens": float64(64)})
	if !ok || n != 64 {
		t.Fatalf("got %d %v", n, ok)
	}
	if _, ok := v1NumPredictFromBody(map[string]any{}); ok {
		t.Fatal("expected no limit")
	}
}

func TestRuntimeV1ProxyOptionsMaxTokens(t *testing.T) {
	opts := runtimeV1ProxyOptions("m", map[string]any{"max_tokens": float64(32)})
	np, ok := opts["num_predict"].(int)
	if !ok || np != 32 {
		t.Fatalf("num_predict=%v", opts["num_predict"])
	}
}

func TestRuntimeV1ProxyOptionsPreservesClientGGUF(t *testing.T) {
	body := map[string]any{
		"options": map[string]any{
			"gguf":    "/data/custom.gguf",
			"num_ctx": float64(8192),
		},
	}
	opts := runtimeV1ProxyOptions("m", body)
	g, ok := opts["gguf"].(string)
	if !ok || g != "/data/custom.gguf" {
		t.Fatalf("gguf=%v", opts["gguf"])
	}
}

func TestRuntimeV1ChatBodyWithOptionsAddsOptions(t *testing.T) {
	raw := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"options":{"num_ctx":4096}}`)
	out, err := runtimeV1ChatBodyWithOptions("m", raw)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	opts, ok := m["options"].(map[string]any)
	if !ok {
		t.Fatalf("missing options: %s", string(out))
	}
	if _, ok := opts["num_ctx"]; !ok {
		t.Fatalf("num_ctx missing: %v", opts)
	}
}

func TestRuntimeV1ChatCompletionsProxyInjectsOptions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var gotBody string
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"object":"chat.completion","choices":[]}`))
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

	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"options":{"num_ctx":4096,"gguf":"/data/proxy-test.gguf"}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body.String())
	}
	var forwarded map[string]any
	if err := json.Unmarshal([]byte(gotBody), &forwarded); err != nil {
		t.Fatal(err)
	}
	opts, ok := forwarded["options"].(map[string]any)
	if !ok {
		t.Fatalf("proxy did not inject options: %s", gotBody)
	}
	if _, ok := opts["num_ctx"]; !ok {
		t.Fatalf("num_ctx missing in forwarded options: %v", opts)
	}
	if g, ok := opts["gguf"].(string); !ok || g != "/data/proxy-test.gguf" {
		t.Fatalf("gguf=%v", opts["gguf"])
	}
}

func TestRuntimeV1ChatCompletionsProxySkipsReasoning(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var hitRuntime bool
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitRuntime = true
		_, _ = w.Write([]byte(`{"object":"chat.completion","choices":[]}`))
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")

	s := &Server{}
	r := gin.New()
	legacy := false
	r.POST("/v1/chat/completions", s.runtimeV1ChatCompletionsProxy(), func(c *gin.Context) {
		legacy = true
		c.JSON(http.StatusTeapot, gin.H{"error": "legacy"})
	})

	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if !legacy || hitRuntime {
		t.Fatalf("legacy=%v hitRuntime=%v", legacy, hitRuntime)
	}
}
