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

func TestRuntimeV1ChatCompletionsProxyExtraBodyCompression(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "chat.completion",
			"usage":  map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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
		"options":{"num_ctx":48},
		"messages":[
			{"role":"user","content":"q"},
			{"role":"assistant","content":"call"},
			{"role":"tool","content":"` + strings.Repeat("tool-bytes ", 80) + `","tool_call_id":"c1"},
			{"role":"user","content":"latest"}
		],
		"extra_body":{"compression":{"elide_from":2}}
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body.String())
	}
	if w.Header().Get("X-Zerollama-Compressed") != "1" {
		t.Fatalf("missing compressed header")
	}
	if !strings.Contains(w.Body.String(), `"compression_meta"`) {
		t.Fatalf("missing compression_meta: %s", w.Body.String())
	}
}

func TestRuntimeV1ChatCompletionsProxyStickyPromptCacheKey(t *testing.T) {
	gin.SetMode(gin.TestMode)
	resetStickyElideForTest()
	t.Cleanup(resetStickyElideForTest)

	var forwarded []map[string]any
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		if msgs, ok := body["messages"].([]any); ok {
			forwarded = nil
			for _, m := range msgs {
				if mm, ok := m.(map[string]any); ok {
					forwarded = append(forwarded, mm)
				}
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "chat.completion",
			"usage":  map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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

	late := strings.Repeat("late-tool-bytes ", 200)
	turn1 := `{
		"model":"m",
		"options":{"num_ctx":200},
		"messages":[
			{"role":"system","content":"rules"},
			{"role":"user","content":"q"},
			{"role":"tool","content":"` + strings.Repeat("early-tool ", 20) + `"},
			{"role":"assistant","content":"next"},
			{"role":"tool","content":"` + late + `"},
			{"role":"user","content":"latest"}
		],
		"extra_body":{"prompt_cache_key":"hermes:sticky:1"}
	}`
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(turn1))
	req1.Header.Set("Content-Type", "application/json")
	w1 := httptest.NewRecorder()
	r.ServeHTTP(w1, req1)
	if w1.Code != http.StatusOK {
		t.Fatalf("turn1 status %d %s", w1.Code, w1.Body.String())
	}
	if w1.Header().Get("X-Zerollama-Compressed") != "1" {
		t.Fatalf("turn1 should compress")
	}

	turn2 := `{
		"model":"m",
		"options":{"num_ctx":5000},
		"messages":[
			{"role":"system","content":"rules"},
			{"role":"user","content":"q"},
			{"role":"tool","content":"` + strings.Repeat("early-tool ", 20) + `"},
			{"role":"assistant","content":"next"},
			{"role":"tool","content":"` + late + `"},
			{"role":"assistant","content":"more"},
			{"role":"tool","content":"` + strings.Repeat("newer-tool-bytes ", 200) + `"},
			{"role":"user","content":"follow-up"}
		],
		"extra_body":{"prompt_cache_key":"hermes:sticky:1"}
	}`
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(turn2))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("turn2 status %d %s", w2.Code, w2.Body.String())
	}
	if len(forwarded) < 5 {
		t.Fatalf("forwarded %d msgs", len(forwarded))
	}
	if content, _ := forwarded[4]["content"].(string); content != chatCompressionToolPlaceholder {
		t.Fatalf("roomy turn2 should keep sticky elide, tool[4]=%q forwarded=%v", content, forwarded)
	}
}
