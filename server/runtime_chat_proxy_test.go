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

func TestRuntimeChatProxyElidesToolOutput(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var saw []any
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		saw, _ = body["messages"].([]any)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"role": "assistant", "content": "ok"},
			"done":    true,
		})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION_MODE", "")

	s := &Server{}
	r := gin.New()
	r.POST("/api/chat", s.runtimeChatProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "should not reach"})
	})
	payload := map[string]any{
		"model":  "llama3.2",
		"stream": false,
		"options": map[string]any{
			"num_ctx": 48,
		},
		"messages": []map[string]any{
			{"role": "user", "content": "q"},
			{"role": "assistant", "content": "call"},
			{"role": "tool", "content": strings.Repeat("tool-bytes ", 80), "tool_call_id": "c1"},
			{"role": "user", "content": "latest"},
		},
	}
	raw, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(string(raw)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	found := false
	for _, m := range saw {
		mm, _ := m.(map[string]any)
		if mm["role"] == "tool" && mm["content"] == chatCompressionToolPlaceholder {
			found = true
		}
	}
	if !found {
		t.Fatalf("runtime saw messages %v", saw)
	}
	if w.Header().Get("X-Zerollama-Compressed") != "1" {
		t.Fatalf("missing compressed header: %v", w.Header())
	}
}

func TestRuntimeChatProxyStreamAttachesCompression(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"message\":{\"role\":\"assistant\",\"content\":\"x\"},\"done\":false}\n"))
		_, _ = w.Write([]byte("{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"done\":true}\n"))
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION_MODE", "")

	s := &Server{}
	r := gin.New()
	r.POST("/api/chat", s.runtimeChatProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "should not reach"})
	})
	payload := map[string]any{
		"model":  "llama3.2",
		"stream": true,
		"options": map[string]any{
			"num_ctx": 48,
		},
		"messages": []map[string]any{
			{"role": "user", "content": "q"},
			{"role": "assistant", "content": "call"},
			{"role": "tool", "content": strings.Repeat("tool-bytes ", 80), "tool_call_id": "c1"},
			{"role": "user", "content": "latest"},
		},
	}
	raw, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(string(raw)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	if w.Header().Get("X-Zerollama-Compressed") != "1" {
		t.Fatalf("missing compressed header")
	}
	var done map[string]any
	for _, line := range strings.Split(strings.TrimSpace(w.Body.String()), "\n") {
		var obj map[string]any
		if json.Unmarshal([]byte(line), &obj) != nil {
			continue
		}
		if d, _ := obj["done"].(bool); d {
			done = obj
		}
	}
	if done == nil {
		t.Fatalf("no done chunk: %s", w.Body.String())
	}
	comp, _ := done["compression"].(map[string]any)
	if comp == nil || comp["mode"] != "placeholder" {
		t.Fatalf("done compression %v body %s", done["compression"], w.Body.String())
	}
}

func TestRuntimeChatProxyStickyPromptCacheKey(t *testing.T) {
	gin.SetMode(gin.TestMode)
	resetStickyElideForTest()
	t.Cleanup(resetStickyElideForTest)

	var saw []any
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		saw, _ = body["messages"].([]any)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"role": "assistant", "content": "ok"},
			"done":    true,
		})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	t.Setenv("OLLAMA_RUNTIME_ALL", "1")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION_MODE", "")

	s := &Server{}
	r := gin.New()
	r.POST("/api/chat", s.runtimeChatProxy(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"error": "should not reach"})
	})

	late := strings.Repeat("late-tool-bytes ", 200)
	early := strings.Repeat("early-tool ", 20)
	turn1 := map[string]any{
		"model":  "llama3.2",
		"stream": false,
		"options": map[string]any{
			"num_ctx":          200,
			"prompt_cache_key": "hermes:api-chat:1",
		},
		"messages": []map[string]any{
			{"role": "system", "content": "rules"},
			{"role": "user", "content": "q"},
			{"role": "tool", "content": early},
			{"role": "assistant", "content": "next"},
			{"role": "tool", "content": late},
			{"role": "user", "content": "latest"},
		},
	}
	raw1, _ := json.Marshal(turn1)
	req1 := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(string(raw1)))
	req1.Header.Set("Content-Type", "application/json")
	w1 := httptest.NewRecorder()
	r.ServeHTTP(w1, req1)
	if w1.Code != http.StatusOK {
		t.Fatalf("turn1 status %d %s", w1.Code, w1.Body.String())
	}
	if w1.Header().Get("X-Zerollama-Compressed") != "1" {
		t.Fatalf("turn1 should compress")
	}

	turn2 := map[string]any{
		"model":  "llama3.2",
		"stream": false,
		"options": map[string]any{
			"num_ctx":          5000,
			"prompt_cache_key": "hermes:api-chat:1",
		},
		"messages": []map[string]any{
			{"role": "system", "content": "rules"},
			{"role": "user", "content": "q"},
			{"role": "tool", "content": early},
			{"role": "assistant", "content": "next"},
			{"role": "tool", "content": late},
			{"role": "assistant", "content": "more"},
			{"role": "tool", "content": strings.Repeat("newer-tool-bytes ", 200)},
			{"role": "user", "content": "follow-up"},
		},
	}
	raw2, _ := json.Marshal(turn2)
	req2 := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(string(raw2)))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("turn2 status %d %s", w2.Code, w2.Body.String())
	}
	if len(saw) < 5 {
		t.Fatalf("forwarded %d msgs", len(saw))
	}
	m4, _ := saw[4].(map[string]any)
	if content, _ := m4["content"].(string); content != chatCompressionToolPlaceholder {
		t.Fatalf("roomy /api/chat turn2 should keep sticky elide, tool[4]=%q", content)
	}
}
