package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/middleware"
)

func assertCloudModelsNotSupportedResponse(t *testing.T, status int, body []byte, upstreamPath string) {
	t.Helper()
	if status != http.StatusForbidden {
		t.Fatalf("expected status 403, got %d (%s)", status, string(body))
	}
	if upstreamPath != "" {
		t.Fatalf("expected no upstream request, got path %q", upstreamPath)
	}
	var got map[string]string
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error: %q", string(body))
	}
	if got["error"] != errCloudModelsNotSupported {
		t.Fatalf("unexpected error: %q", got["error"])
	}
}

func assertCloudNativeAPIUseV1Error(t *testing.T, status int, body []byte) {
	t.Helper()
	if status != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d (%s)", status, string(body))
	}
	var got map[string]string
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error: %q", string(body))
	}
	if got["error"] != errCloudUseOpenAICompat {
		t.Fatalf("unexpected error: %q", got["error"])
	}
}

func assertCloudV1Proxied(t *testing.T, status int, body []byte, capturePath, wantUpstreamPath string) {
	t.Helper()
	if status != http.StatusOK {
		t.Fatalf("expected status 200, got %d (%s)", status, string(body))
	}
	if capturePath != wantUpstreamPath {
		t.Fatalf("expected upstream path %q, got %q", wantUpstreamPath, capturePath)
	}
}

func TestStatusHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := Server{}
	w := createRequest(t, s.StatusHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp api.StatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if !resp.Cloud.Disabled {
		t.Fatalf("expected cloud.disabled true, got false")
	}
	if resp.Cloud.Source != "env" {
		t.Fatalf("expected cloud.source env, got %q", resp.Cloud.Source)
	}
}

func TestCloudDisabledBlocksRemoteOperations(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := Server{}

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "test-cloud",
		RemoteHost: "example.com",
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion"},
		},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("chat remote blocked", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "test-cloud",
			Messages: []api.Message{{Role: "user", Content: "hi"}},
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"`+errCloudModelsNotSupported+`"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})

	t.Run("generate remote blocked", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-cloud",
			Prompt: "hi",
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"`+errCloudModelsNotSupported+`"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})

	t.Run("show remote blocked", func(t *testing.T) {
		w := createRequest(t, s.ShowHandler, api.ShowRequest{
			Model: "test-cloud",
		})
		if w.Code != http.StatusNotFound {
			t.Fatalf("expected status 404, got %d", w.Code)
		}
	})
}

func TestDeleteHandlerNormalizesExplicitSourceSuffixes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	s := Server{}

	tests := []string{
		"gpt-oss:20b:local",
		"gpt-oss:20b:cloud",
		"qwen3:cloud",
	}

	for _, modelName := range tests {
		t.Run(modelName, func(t *testing.T) {
			w := createRequest(t, s.DeleteHandler, api.DeleteRequest{
				Model: modelName,
			})
			if w.Code != http.StatusNotFound {
				t.Fatalf("expected status 404, got %d (%s)", w.Code, w.Body.String())
			}

			var resp map[string]string
			if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
				t.Fatal(err)
			}
			want := "model '" + modelName + "' not found"
			if resp["error"] != want {
				t.Fatalf("unexpected error: got %q, want %q", resp["error"], want)
			}
		})
	}
}

func TestExplicitCloudPassthroughAPIAndV1(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	type upstreamCapture struct {
		path   string
		body   string
		header http.Header
	}

	newUpstream := func(t *testing.T, responseBody string) (*httptest.Server, *upstreamCapture) {
		t.Helper()
		capture := &upstreamCapture{}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			payload, _ := io.ReadAll(r.Body)
			capture.path = r.URL.Path
			capture.body = string(payload)
			capture.header = r.Header.Clone()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(responseBody))
		}))

		return srv, capture
	}

	t.Run("api generate", func(t *testing.T) {
		upstream, _ := newUpstream(t, `{"ok":"api"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello","stream":false}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/generate", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Test-Header", "api-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudNativeAPIUseV1Error(t, resp.StatusCode, body)
	})

	t.Run("api chat", func(t *testing.T) {
		upstream, _ := newUpstream(t, `{"message":{"role":"assistant","content":"ok"},"done":true}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","messages":[{"role":"user","content":"hello"}],"stream":false}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/chat", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudNativeAPIUseV1Error(t, resp.StatusCode, body)
	})

	t.Run("api embed", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"model":"kimi-k2.5:cloud","embeddings":[[0.1,0.2]]}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","input":"hello"}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/embed", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/v1/embeddings" {
			t.Fatalf("expected upstream path /api/v1/embeddings, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}
	})

	t.Run("api embeddings", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"embedding":[0.1,0.2]}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello"}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/embeddings", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/v1/embeddings" {
			t.Fatalf("expected upstream path /api/v1/embeddings, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}
	})

	t.Run("api show", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"details":{"format":"gguf"}}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud"}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/show", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/v1/models/kimi-k2.5" {
			t.Fatalf("expected upstream path /api/v1/models/kimi-k2.5, got %q", capture.path)
		}
	})

	t.Run("v1 chat completions bypasses conversion", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"chatcmpl_test","object":"chat.completion"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"gpt-oss:120b:cloud","messages":[{"role":"user","content":"hi"}],"max_tokens":7}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/chat/completions", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Test-Header", "v1-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/chat/completions")
	})

	t.Run("v1 chat completions bypasses conversion with legacy cloud suffix", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"chatcmpl_test","object":"chat.completion"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"gpt-oss:120b-cloud","messages":[{"role":"user","content":"hi"}],"max_tokens":7}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/chat/completions", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Test-Header", "v1-legacy-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/chat/completions")
	})

	t.Run("v1 messages bypasses conversion", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"msg_1","type":"message"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/messages")
	})

	t.Run("v1 messages bypasses conversion with legacy cloud suffix", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"msg_1","type":"message"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:latest-cloud","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/messages")
	})

	t.Run("v1 messages web_search fallback uses legacy cloud /api/chat path", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"model":"gpt-oss:120b","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"hello"},"done":true}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
				"model":"gpt-oss:120b-cloud",
				"max_tokens":10,
				"messages":[{"role":"user","content":"search the web"}],
				"tools":[{"type":"web_search_20250305","name":"web_search"}],
				"stream":false
			}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages?beta=true", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/messages")
	})

	t.Run("v1 messages web_search fallback frames coalesced jsonl chunks", func(t *testing.T) {
		type upstreamCapture struct {
			path string
		}
		capture := &upstreamCapture{}
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capture.path = r.URL.Path
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.WriteHeader(http.StatusOK)

			combined := strings.Join([]string{
				`{"model":"gpt-oss:120b","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hel"},"done":false}`,
				`{"model":"gpt-oss:120b","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"lo"},"done":true}`,
			}, "\n") + "\n"
			_, _ = w.Write([]byte(combined))
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
					"model":"gpt-oss:120b-cloud",
					"max_tokens":10,
					"stream":true,
					"messages":[{"role":"user","content":"search the web"}],
					"tools":[{"type":"web_search_20250305","name":"web_search"}]
				}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages?beta=true", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/messages")
	})

	t.Run("v1 model retrieve bypasses conversion", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"kimi-k2.5:cloud","object":"model","created":1,"owned_by":"ollama"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, local.URL+"/v1/models/kimi-k2.5:cloud", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Test-Header", "v1-model-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/models/kimi-k2.5")
	})

	t.Run("v1 model retrieve normalizes legacy cloud suffix", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"kimi-k2.5:latest","object":"model","created":1,"owned_by":"ollama"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, local.URL+"/v1/models/kimi-k2.5:latest-cloud", nil)
		if err != nil {
			t.Fatal(err)
		}

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/models/kimi-k2.5:latest")
	})
}

func TestCloudDisabledBlocksExplicitCloudPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/chat/completions", bytes.NewBufferString(`{"model":"kimi-k2.5:cloud","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("expected status 403, got %d (%s)", resp.StatusCode, string(body))
	}

	var got map[string]string
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error body, got: %q", string(body))
	}

	want := internalcloud.DisabledError(cloudErrRemoteInferenceUnavailable)
	if got["error"] != want {
		t.Fatalf("unexpected error message: %q want %q", got["error"], want)
	}
}

func TestCloudPassthroughStreamsPromptly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{"model":"kimi-k2.5:cloud","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/chat", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	assertCloudNativeAPIUseV1Error(t, resp.StatusCode, body)
}

func TestCloudPassthroughSkipsAnthropicWebSearch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	type upstreamCapture struct {
		path string
	}
	capture := &upstreamCapture{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture.path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message"}`))
	}))
	defer upstream.Close()

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	router := gin.New()
	router.POST(
		"/v1/messages",
		cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable),
		cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable),
		middleware.AnthropicMessagesMiddleware(),
		func(c *gin.Context) { c.Status(http.StatusTeapot) },
	)

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{
		"model":"kimi-k2.5:cloud",
		"max_tokens":10,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/messages")
}

func TestCloudPassthroughSkipsAnthropicWebSearchLegacySuffix(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	type upstreamCapture struct {
		path string
	}
	capture := &upstreamCapture{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture.path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message"}`))
	}))
	defer upstream.Close()

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	router := gin.New()
	router.POST(
		"/v1/messages",
		cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable),
		cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable),
		middleware.AnthropicMessagesMiddleware(),
		func(c *gin.Context) { c.Status(http.StatusTeapot) },
	)

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{
		"model":"kimi-k2.5:latest-cloud",
		"max_tokens":10,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	assertCloudV1Proxied(t, resp.StatusCode, body, capture.path, "/api/v1/messages")
}

func TestCloudPassthroughSigningFailureReturnsUnauthorized(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	origSignRequest := cloudProxySignRequest
	origSigninURL := cloudProxySigninURL
	cloudProxySignRequest = func(context.Context, *http.Request) error {
		return errors.New("ssh: no key found")
	}
	cloudProxySigninURL = func() (string, error) {
		return "https://ollama.com/signin/example", nil
	}
	t.Cleanup(func() {
		cloudProxySignRequest = origSignRequest
		cloudProxySigninURL = origSigninURL
	})

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello","stream":false}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/generate", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	assertCloudNativeAPIUseV1Error(t, resp.StatusCode, body)
}

func TestCloudPassthroughSigningFailureWithoutSigninURL(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	origSignRequest := cloudProxySignRequest
	origSigninURL := cloudProxySigninURL
	cloudProxySignRequest = func(context.Context, *http.Request) error {
		return errors.New("ssh: no key found")
	}
	cloudProxySigninURL = func() (string, error) {
		return "", errors.New("key missing")
	}
	t.Cleanup(func() {
		cloudProxySignRequest = origSignRequest
		cloudProxySigninURL = origSigninURL
	})

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello","stream":false}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/generate", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	assertCloudNativeAPIUseV1Error(t, resp.StatusCode, body)
}
