package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestCachePinHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cachePinMu.Lock()
	cachePinByID = map[string]*cachePinLease{}
	cachePinMu.Unlock()

	s := &Server{}
	r := gin.New()
	r.POST("/api/cache/pin", s.CachePinHandler)
	r.DELETE("/api/cache/pin/:id", s.CacheUnpinHandler)

	ttl := 60
	body, _ := json.Marshal(api.CachePinRequest{
		PromptCacheKey: "hermes:agent:test",
		TTLSeconds:     &ttl,
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/cache/pin", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var resp api.CachePinResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if !resp.CanPin || resp.PinID == "" || resp.PromptCacheKey != "hermes:agent:test" {
		t.Fatalf("%+v", resp)
	}
	if !cacheKeyIsPinned("hermes:agent:test") {
		t.Fatal("expected pin active")
	}
	if cacheKeyIsPinned("other") {
		t.Fatal("other key must not be pinned")
	}

	w2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodDelete, "/api/cache/pin/"+resp.PinID, nil)
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("unpin status=%d body=%s", w2.Code, w2.Body.String())
	}
	if cacheKeyIsPinned("hermes:agent:test") {
		t.Fatal("pin should be gone")
	}
}

func TestCacheWarmHandlerValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/api/cache/warm", s.CacheWarmHandler)

	cases := []struct {
		name string
		body any
		code int
	}{
		{
			name: "missing key",
			body: api.CacheWarmRequest{Model: "x", Prompt: "hi"},
			code: http.StatusBadRequest,
		},
		{
			name: "missing prompt",
			body: api.CacheWarmRequest{Model: "x", PromptCacheKey: "k"},
			code: http.StatusBadRequest,
		},
		{
			name: "missing model",
			body: api.CacheWarmRequest{Prompt: "hi", PromptCacheKey: "k"},
			code: http.StatusBadRequest,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, _ := json.Marshal(tc.body)
			w := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/api/cache/warm", bytes.NewReader(b))
			req.Header.Set("Content-Type", "application/json")
			r.ServeHTTP(w, req)
			if w.Code != tc.code {
				t.Fatalf("status=%d body=%s want %d", w.Code, w.Body.String(), tc.code)
			}
		})
	}
}
