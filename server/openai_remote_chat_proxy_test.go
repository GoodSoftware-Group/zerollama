package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

func TestOpenAIRemoteShouldProxy(t *testing.T) {
	remote := &Model{
		Config: model.ConfigV2{
			ModalityBackends: map[string]string{
				model.ModalityInference: model.BackendOpenAIRemote,
			},
		},
	}
	if !openaiRemoteShouldProxy(remote) {
		t.Fatal("inference=openai-remote should proxy")
	}
	if openaiRemoteShouldProxy(&Model{Config: model.ConfigV2{}}) {
		t.Fatal("plain model must not proxy")
	}
	if openaiRemoteShouldProxy(nil) {
		t.Fatal("nil must not proxy")
	}
}

func TestOpenAIRemoteBaseURLPrecedence(t *testing.T) {
	t.Setenv("ZEROLLAMA_OPENAI_REMOTE_URL", "http://fleet:30000")
	m := &Model{
		Config: model.ConfigV2{
			BackendPaths: map[string]string{"openai_url": "http://per-model:30000"},
		},
	}
	if got := openaiRemoteBaseURL(m); got != "http://per-model:30000" {
		t.Fatalf("per-model url: got %q", got)
	}
	if got := openaiRemoteBaseURL(&Model{}); got != "http://fleet:30000" {
		t.Fatalf("fleet url: got %q", got)
	}
	t.Setenv("ZEROLLAMA_OPENAI_REMOTE_URL", "")
	if got := openaiRemoteBaseURL(&Model{}); got != "" {
		t.Fatalf("empty: got %q", got)
	}
}

func TestOpenAIRemoteUpstreamModel(t *testing.T) {
	m := &Model{
		Config: model.ConfigV2{
			BackendPaths: map[string]string{"openai_model": "glm-5.3-flash"},
		},
	}
	if got := openaiRemoteUpstreamModel(m, "glm-5.3-flash:latest"); got != "glm-5.3-flash" {
		t.Fatalf("explicit openai_model: got %q", got)
	}
	plain := &Model{}
	if got := openaiRemoteUpstreamModel(plain, "glm-5.3-flash:latest"); got != "glm-5.3-flash" {
		t.Fatalf("strip tag: got %q", got)
	}
}

func registerOpenAIRemoteTestModel(t *testing.T, cfg model.ConfigV2) model.Name {
	t.Helper()
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	layer, err := manifest.NewLayer(bytes.NewReader(raw), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatal(err)
	}
	name := model.ParseName("glm-5.3-flash")
	if err := manifest.WriteManifest(name, layer, nil); err != nil {
		t.Fatal(err)
	}
	return name
}

func TestOpenAIRemoteChatProxyMissingURL(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_OPENAI_REMOTE_URL", "")
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	registerOpenAIRemoteTestModel(t, model.ConfigV2{
		Architecture: "amd64",
		OS:           "linux",
		Capabilities: []string{"completion"},
		ModalityBackends: map[string]string{
			model.ModalityInference: model.BackendOpenAIRemote,
		},
	})

	s := &Server{}
	r := gin.New()
	r.POST("/v1/chat/completions", s.openaiRemoteChatCompletionsProxy(), func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"fallback": true})
	})

	body := `{"model":"glm-5.3-flash","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestOpenAIRemoteOllamaChatRequest(t *testing.T) {
	m := &Model{
		Config: model.ConfigV2{
			BackendPaths: map[string]string{"openai_model": "glm-5.3-flash"},
		},
	}
	n := 32
	stream := false
	req := api.ChatRequest{
		Model: "glm-5.3-flash:latest",
		Messages: []api.Message{
			{Role: "user", Content: "hi"},
		},
		Stream:  &stream,
		Options: map[string]any{"num_predict": n},
	}
	oreq, err := openaiRemoteOllamaChatRequest(m, req)
	if err != nil {
		t.Fatal(err)
	}
	if oreq.Model != "glm-5.3-flash" {
		t.Fatalf("model=%q", oreq.Model)
	}
	if oreq.Stream {
		t.Fatal("upstream must be non-stream")
	}
	if oreq.MaxTokens == nil || *oreq.MaxTokens != 32 {
		t.Fatalf("max_tokens=%v", oreq.MaxTokens)
	}
}

func TestOpenAIRemoteChatAPIProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"role":"assistant","content":"hola"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))
	}))
	t.Cleanup(up.Close)

	registerOpenAIRemoteTestModel(t, model.ConfigV2{
		Architecture: "amd64",
		OS:           "linux",
		Capabilities: []string{"completion"},
		ModalityBackends: map[string]string{
			model.ModalityInference: model.BackendOpenAIRemote,
		},
		BackendPaths: map[string]string{
			"openai_url":   up.URL,
			"openai_model": "glm-5.3-flash",
		},
	})

	s := &Server{}
	r := gin.New()
	r.POST("/api/chat", s.openaiRemoteChatProxy(), func(c *gin.Context) {
		t.Fatal("must not fall through")
	})

	body := `{"model":"glm-5.3-flash","messages":[{"role":"user","content":"hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var out api.ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if !out.Done || out.Message.Content != "hola" {
		t.Fatalf("out=%+v", out)
	}
}

func TestOpenAIRemoteChatProxyForward(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	var gotModel string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		var req openai.ChatCompletionRequest
		_ = json.Unmarshal(b, &req)
		gotModel = req.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(up.Close)

	registerOpenAIRemoteTestModel(t, model.ConfigV2{
		Architecture: "amd64",
		OS:           "linux",
		Capabilities: []string{"completion"},
		ModalityBackends: map[string]string{
			model.ModalityInference: model.BackendOpenAIRemote,
		},
		BackendPaths: map[string]string{
			"openai_url":   up.URL,
			"openai_model": "glm-5.3-flash",
		},
	})

	s := &Server{}
	r := gin.New()
	r.POST("/v1/chat/completions", s.openaiRemoteChatCompletionsProxy(), func(c *gin.Context) {
		t.Fatal("must not fall through")
	})

	body := `{"model":"glm-5.3-flash:latest","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	if gotModel != "glm-5.3-flash" {
		t.Fatalf("upstream model=%q", gotModel)
	}
	if !bytes.Contains(w.Body.Bytes(), []byte(`"ok"`)) {
		t.Fatalf("body=%s", w.Body.String())
	}
}
