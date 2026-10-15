package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/types/model"
)

const routeToolsTemplate = `
{{- if .Tools }}{{ .Tools }}{{ end }}
{{- range .Messages }}{{ .Role }}: {{ .Content }}{{ end }}`

const routePlainTemplate = `{{- range .Messages }}{{ .Role }}: {{ .Content }}{{ end }}`

func newRouteTestServer(t *testing.T) *Server {
	t.Helper()
	gin.SetMode(gin.TestMode)
	resetRouteMemory()
	t.Cleanup(resetRouteMemory)
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := &mockRunner{CompletionResponse: llm.CompletionResponse{
		Done: true, DoneReason: llm.DoneReasonStop, PromptEvalCount: 1, PromptEvalDuration: 1, EvalCount: 1, EvalDuration: 1,
	}}
	s := &Server{sched: &Scheduler{
		pending:         newPendingQueue(4),
		finishedReqCh:   make(chan *LlmRequest, 4),
		expiredCh:       make(chan *runnerRef, 4),
		unloadedCh:      make(chan any, 4),
		loaded:          make(map[string]*runnerRef),
		newServerFn:     newMockServer(mock),
		getGpuFn:        getGpuFn,
		getSystemInfoFn: getSystemInfoFn,
		waitForRecovery: 250 * time.Millisecond,
		loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
			req.successCh <- &runnerRef{llama: mock}
			return false
		},
	}}
	go s.sched.Run(t.Context())
	return s
}

func createRouteTestModel(t *testing.T, s *Server, name, template string) {
	t.Helper()
	createRouteTestModelKV(t, s, name, template, nil)
}

// createRouteTestModelKV creates a tiny llama GGUF tag; extra KV (e.g. llama.pooling_type)
// changes the capabilities the router sees.
func createRouteTestModelKV(t *testing.T, s *Server, name, template string, extra ggml.KV) {
	t.Helper()
	tensor := func(n string) *ggml.Tensor {
		return &ggml.Tensor{Name: n, Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))}
	}
	kv := ggml.KV{
		"general.architecture":          "llama",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(8192),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}
	for k, v := range extra {
		kv[k] = v
	}
	_, digest := createBinFile(t, kv, []*ggml.Tensor{
		tensor("token_embd.weight"), tensor("blk.0.attn_norm.weight"), tensor("blk.0.ffn_down.weight"),
		tensor("blk.0.ffn_gate.weight"), tensor("blk.0.ffn_up.weight"), tensor("blk.0.ffn_norm.weight"),
		tensor("blk.0.attn_k.weight"), tensor("blk.0.attn_output.weight"), tensor("blk.0.attn_q.weight"),
		tensor("blk.0.attn_v.weight"), tensor("output.weight"),
	})
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    name,
		Files:    map[string]string{"file.gguf": digest},
		Template: template,
		Stream:   &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create %s: status %d: %s", name, w.Code, w.Body.String())
	}
}

func routeChat(t *testing.T, s *Server, req api.ChatRequest) (int, http.Header, string) {
	t.Helper()
	req.Stream = &stream
	w := createRequest(t, s.ChatHandler, req)
	return w.Code, w.Header(), w.Body.String()
}

var weatherTool = api.Tool{
	Type: "function",
	Function: api.ToolFunction{
		Name:        "get_weather",
		Description: "Get the weather",
	},
}

func TestAutoRouteChatPicksToolCapableModel(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	code, hdr, body := routeChat(t, s, api.ChatRequest{
		Model:    "auto",
		Messages: []api.Message{{Role: "user", Content: "weather in Paris?"}},
		Tools:    []api.Tool{weatherTool},
	})
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	if got := hdr.Get("X-Zerollama-Router"); got != "auto" {
		t.Fatalf("X-Zerollama-Router = %q", got)
	}
	if got := hdr.Get("X-Zerollama-Router-Chosen"); !strings.HasPrefix(got, "tooly") {
		t.Fatalf("chosen = %q, want tooly", got)
	}
	if hdr.Get("X-Zerollama-Router-Reason") != "only-candidate" {
		t.Fatalf("reason = %q", hdr.Get("X-Zerollama-Router-Reason"))
	}
}

func TestAutoRouteChatNoCandidateExplains(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)

	code, _, body := routeChat(t, s, api.ChatRequest{
		Model:    "auto",
		Messages: []api.Message{{Role: "user", Content: "weather?"}},
		Tools:    []api.Tool{weatherTool},
	})
	if code != http.StatusServiceUnavailable {
		t.Fatalf("status %d: %s", code, body)
	}
	if !strings.Contains(body, "missing capability: tools") {
		t.Fatalf("body should explain rejection: %s", body)
	}
}

func TestAutoRouteDisabledAndShadowed(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)

	t.Run("env off", func(t *testing.T) {
		t.Setenv("ZEROLLAMA_AUTO_ROUTER", "0")
		code, hdr, _ := routeChat(t, s, api.ChatRequest{Model: "auto", Messages: []api.Message{{Role: "user", Content: "hi"}}})
		if code != http.StatusNotFound || hdr.Get("X-Zerollama-Router") != "" {
			t.Fatalf("status %d router %q, want 404 and no routing", code, hdr.Get("X-Zerollama-Router"))
		}
	})

	t.Run("local model named auto wins", func(t *testing.T) {
		createRouteTestModel(t, s, "auto", routePlainTemplate)
		code, hdr, body := routeChat(t, s, api.ChatRequest{Model: "auto", Messages: []api.Message{{Role: "user", Content: "hi"}}})
		if code != http.StatusOK || hdr.Get("X-Zerollama-Router") != "" {
			t.Fatalf("status %d router %q: %s", code, hdr.Get("X-Zerollama-Router"), body)
		}
	})
}

func TestIsImplicitAutoName(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	for name, want := range map[string]bool{
		"auto":                                   true,
		"AUTO":                                   true,
		"auto:latest":                            true,
		"auto:v2":                                false,
		"library/auto":                           true,
		"someone/auto":                           false,
		"example.com/x/auto":                     false,
		"llama3.2":                               false,
		"autonomous:latest":                      false,
		"":                                       false,
		"registry.ollama.ai/library/auto:latest": true,
	} {
		if got := isImplicitAutoName(name); got != want {
			t.Errorf("isImplicitAutoName(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestEnforceRouterSurfaceUsesFallback(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	sig := autoroute.Signals{Surface: autoroute.SurfaceChat, HasTools: true}

	dec := RouterDecision{Router: "agent", Candidate: "plain"}
	if err := enforceRouterSurface(&dec, RouterSpec{Fallback: "tooly"}, sig); err != nil {
		t.Fatal(err)
	}
	if dec.Candidate != "tooly" || !dec.Fallback || dec.Reason != "surface-fallback" {
		t.Fatalf("dec = %+v, want fallback to tooly", dec)
	}

	dec = RouterDecision{Router: "agent", Candidate: "plain"}
	err := enforceRouterSurface(&dec, RouterSpec{}, sig)
	if err == nil || !strings.Contains(err.Error(), "missing capability: tools") {
		t.Fatalf("err = %v", err)
	}

	// Image surface must never accept a chat model.
	dec = RouterDecision{Router: "img", Candidate: "tooly"}
	err = enforceRouterSurface(&dec, RouterSpec{}, autoroute.Signals{Surface: autoroute.SurfaceImage})
	if err == nil || !strings.Contains(err.Error(), string(model.CapabilityImage)) {
		t.Fatalf("image surface err = %v", err)
	}

	// Models outside the local inventory pass through for the handler to report.
	dec = RouterDecision{Router: "agent", Candidate: "not-pulled:7b"}
	if err := enforceRouterSurface(&dec, RouterSpec{}, sig); err != nil || dec.Candidate != "not-pulled:7b" {
		t.Fatalf("unknown model: dec=%+v err=%v", dec, err)
	}
}

func TestGenerateRouteSignalsSurface(t *testing.T) {
	gin.SetMode(gin.TestMode)
	newCtx := func(surface string) *gin.Context {
		c, _ := gin.CreateTestContext(NewRecorder())
		if surface != "" {
			c.Set("zerollama.route.surface", surface)
		}
		return c
	}
	if got := generateRouteSignals(newCtx("image"), &api.GenerateRequest{Prompt: "cat"}).Surface; got != autoroute.SurfaceImage {
		t.Fatalf("middleware image: %q", got)
	}
	if got := generateRouteSignals(newCtx("image_edit"), &api.GenerateRequest{Prompt: "cat"}).Surface; got != autoroute.SurfaceImageEdit {
		t.Fatalf("middleware edit: %q", got)
	}
	if got := generateRouteSignals(newCtx(""), &api.GenerateRequest{Prompt: "cat", Width: 512}).Surface; got != autoroute.SurfaceImage {
		t.Fatalf("native width: %q", got)
	}
	sig := generateRouteSignals(newCtx(""), &api.GenerateRequest{Prompt: strings.Repeat("x", 4000), Options: map[string]any{"num_predict": 64}})
	if sig.Surface != autoroute.SurfaceGenerate || sig.PromptTokens != 1000 || sig.ReplyTokens != 64 {
		t.Fatalf("text sig = %+v", sig)
	}
}

func TestChatRouteSignals(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(NewRecorder())
	think := &api.ThinkValue{Value: true}
	req := &api.ChatRequest{
		Messages: []api.Message{
			{Role: "user", Content: "look", Images: []api.ImageData{[]byte("png")}},
			{Role: "user", Content: "and listen", AudioClips: []api.AudioData{[]byte("wav")}},
		},
		Think: think,
	}
	sig := chatRouteSignals(c, req)
	if !sig.HasImages || !sig.HasAudio || !sig.WantsThink || sig.Text != "and listen" {
		t.Fatalf("sig = %+v", sig)
	}

	req.ThinkFromAlias = true
	if chatRouteSignals(c, req).WantsThink {
		t.Fatal("alias-derived think must not require thinking")
	}
	req.ThinkFromAlias = false
	c.Set("relax_thinking", true)
	if chatRouteSignals(c, req).WantsThink {
		t.Fatal("relax_thinking must not require thinking")
	}
}

func TestRouterDecisionJSONCarriesAuto(t *testing.T) {
	dec := RouterDecision{Router: "auto", Candidate: "m", Reason: "warm", Auto: &autoroute.Decision{Model: "m"}}
	b, err := json.Marshal(dec)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(`"reason":"warm"`)) || !bytes.Contains(b, []byte(`"auto":{`)) {
		t.Fatalf("json = %s", b)
	}
}

func TestImplicitAutoSkipsRouterHelpers(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)
	sig := autoroute.Signals{Surface: autoroute.SurfaceChat, Text: "hi"}

	// plain is only a score classifier ⇒ never an implicit auto pick.
	resetRouterFile(t, `
routers:
  agent:
    classifier: plain
    candidates:
      - {model: tooly, labels: [chat]}
`)
	dec, err := s.decideAuto(t.Context(), "auto", sig, nil)
	if err != nil || !strings.HasPrefix(dec.Candidate, "tooly") {
		t.Fatalf("helper picked: %+v %v", dec, err)
	}
	found := false
	for _, r := range dec.Auto.Rejected {
		if strings.HasPrefix(r.Model, "plain") && strings.Contains(r.Reason, "score classifier") {
			found = true
		}
	}
	if !found {
		t.Fatalf("helper rejection missing: %+v", dec.Auto.Rejected)
	}

	// A helper that a router also serves from stays eligible.
	resetRouterFile(t, `
routers:
  agent:
    classifier: plain
    candidates:
      - {model: plain, labels: [chat]}
      - {model: tooly, labels: [code]}
`)
	dec, _ = s.decideAuto(t.Context(), "auto", sig, nil)
	for _, r := range dec.Auto.Rejected {
		if strings.Contains(r.Reason, "router helper") {
			t.Fatalf("serving helper rejected: %+v", r)
		}
	}

	// Explicit router allow-lists are untouched (only implicit auto filters helpers).
	resetRouterFile(t, `
routers:
  agent:
    classifier: plain
    candidates:
      - {model: tooly, labels: [chat]}
`)
	dec, err = s.decideAuto(t.Context(), "agent", sig, map[string]bool{"plain:latest": true})
	if err != nil || !strings.HasPrefix(dec.Candidate, "plain") {
		t.Fatalf("allow-list path filtered helpers: %+v %v", dec, err)
	}
}

func TestImplicitAutoHelperFilterIsSoft(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	resetRouterFile(t, `
routers:
  agent:
    decider: plain
    classifier: jev
    candidates:
      - {model: gone-model, labels: [chat]}
`)
	dec, err := s.decideAuto(t.Context(), "auto", autoroute.Signals{Surface: autoroute.SurfaceChat, Text: "hi"}, nil)
	if err != nil || !strings.HasPrefix(dec.Candidate, "plain") {
		t.Fatalf("only-helper node must still answer: %+v %v", dec, err)
	}
}

func TestRouteInventoryReportsLayaAsDecisionOnly(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModelKV(t, s, "laya", routePlainTemplate, ggml.KV{"general.architecture": "laya"})

	cands, _, err := routeCandidates()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cands {
		if strings.HasPrefix(c.Name, "laya") && (len(c.Capabilities) != 1 || c.Capabilities[0] != model.CapabilityDecision) {
			t.Fatalf("laya caps = %v, want [decision]", c.Capabilities)
		}
	}
	// A chat turn must never land on the decision encoder.
	dec, err := s.decideAuto(t.Context(), "auto", autoroute.Signals{Surface: autoroute.SurfaceChat, Text: "hi"}, nil)
	if err != nil || !strings.HasPrefix(dec.Candidate, "plain") {
		t.Fatalf("chat routed to %+v (%v)", dec, err)
	}
	// And decisions route to it.
	dec, err = s.decideAuto(t.Context(), "auto", autoroute.Signals{Surface: autoroute.SurfaceDecisions}, nil)
	if err != nil || !strings.HasPrefix(dec.Candidate, "laya") {
		t.Fatalf("decisions routed to %+v (%v)", dec, err)
	}
}
