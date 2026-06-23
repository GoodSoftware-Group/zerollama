package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestInferenceStatusEmptyScheduler(t *testing.T) {
	st := (&Server{}).inferenceStatus(context.Background())
	if st.Ggml.Pending != 0 || st.Ggml.Active != 0 || st.Ggml.Loaded != 0 {
		t.Fatalf("expected zero ggml backlog, got pending=%d active=%d loaded=%d",
			st.Ggml.Pending, st.Ggml.Active, st.Ggml.Loaded)
	}
	if st.Ggml.Loading {
		t.Fatal("expected ggml.loading false with nil scheduler")
	}
	if len(st.Ggml.LoadedModels) != 0 {
		t.Fatalf("expected no loaded models, got %v", st.Ggml.LoadedModels)
	}
	if st.Runtime.Enabled {
		t.Fatal("expected runtime.enabled false without ZEROLLAMA_RUNTIME_URL")
	}
}

func TestInferenceStatusWithRuntimeHealth(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"waiting":         3,
			"running":         1,
			"inference_state": "running",
			"llama_server":    true,
		})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	st := (&Server{}).inferenceStatus(context.Background())
	if !st.Runtime.Enabled || !st.Runtime.Available {
		t.Fatalf("runtime enabled=%v available=%v", st.Runtime.Enabled, st.Runtime.Available)
	}
	if st.Runtime.Waiting == nil || *st.Runtime.Waiting != 3 {
		t.Fatalf("runtime waiting=%v", st.Runtime.Waiting)
	}
	if st.Runtime.Running == nil || *st.Runtime.Running != 1 {
		t.Fatalf("runtime running=%v", st.Runtime.Running)
	}
	if st.Runtime.LlamaLoaded == nil || !*st.Runtime.LlamaLoaded {
		t.Fatal("expected llama_loaded true")
	}
	if st.Runtime.State != "running" {
		t.Fatalf("state=%q", st.Runtime.State)
	}
}

func TestInferenceStatusRuntimeUnavailableOmitsQueue(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	st := (&Server{}).inferenceStatus(context.Background())
	if !st.Runtime.Enabled || st.Runtime.Available {
		t.Fatalf("enabled=%v available=%v", st.Runtime.Enabled, st.Runtime.Available)
	}
	if st.Runtime.Waiting != nil || st.Runtime.Running != nil || st.Runtime.LlamaLoaded != nil {
		t.Fatalf("queue fields should be omitted when unavailable: %+v", st.Runtime)
	}

	raw, err := json.Marshal(st.Runtime)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"waiting", "running", "llama_loaded"} {
		if _, ok := m[key]; ok {
			t.Fatalf("expected %q omitted from JSON, got %v", key, m)
		}
	}
}

func TestStatusHandlerIncludesInference(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := Server{}
	w := createRequest(t, s.StatusHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}

	var resp api.StatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if !resp.Cloud.Disabled {
		t.Fatal("expected cloud disabled")
	}
	if resp.Inference.Ggml.Loaded != 0 {
		t.Fatalf("expected ggml.loaded 0, got %d", resp.Inference.Ggml.Loaded)
	}
	if resp.Inference.Backend.LlamaServer == "" {
		t.Fatal("expected inference.backend.llama_server set")
	}
	if resp.Inference.Backend.GgufPath == "" {
		t.Fatal("expected inference.backend.gguf_path set")
	}
	if resp.Inference.Backend.RuntimeChat == "" {
		t.Fatal("expected inference.backend.runtime_chat set")
	}
	if !resp.Inference.Backend.GgmlLinked {
		t.Fatal("expected ggml_linked true in default build test")
	}
}

func TestStatusHandlerIncludesBackendEdge(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")
	t.Setenv("ZEROLLAMA_EDGE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	t.Setenv("ZEROLLAMA_RUNTIME", "0")

	s := Server{}
	w := createRequest(t, s.StatusHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}

	var resp api.StatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if !resp.Inference.Backend.Edge {
		t.Fatal("expected backend.edge true")
	}
	if resp.Inference.Backend.LlamaServer != "explicit" {
		t.Fatalf("llama_server=%q", resp.Inference.Backend.LlamaServer)
	}
	if resp.Inference.Backend.RuntimeChat != "off" {
		t.Fatalf("runtime_chat=%q", resp.Inference.Backend.RuntimeChat)
	}
	if resp.Inference.Backend.GgufPath != "llama-server" {
		t.Fatalf("gguf_path=%q", resp.Inference.Backend.GgufPath)
	}
}

func TestInferenceFleetSnapshotLoadedModelsMatchCount(t *testing.T) {
	ctx := t.Context()
	sched := InitScheduler(ctx)

	runner := &runnerRef{
		model:    &Model{ShortName: "llama3:latest"},
		modelKey: "test-key",
		llama:    &mockLlm{},
		loadedMeta: api.LoadedModelMetadata{
			NumCtx:   4096,
			ProbedAt: time.Now().UTC(),
		},
	}
	sched.loadedMu.Lock()
	sched.loaded["test-key"] = runner
	sched.loadedMu.Unlock()

	snap := sched.InferenceFleetSnapshot()
	if snap.Loaded != 1 {
		t.Fatalf("loaded=%d", snap.Loaded)
	}
	if len(snap.LoadedModels) != snap.Loaded {
		t.Fatalf("loaded=%d loaded_models=%d (%v)", snap.Loaded, len(snap.LoadedModels), snap.LoadedModels)
	}
	if snap.LoadedModels[0] != "llama3:latest" {
		t.Fatalf("loaded_models=%v", snap.LoadedModels)
	}
}

func TestInferenceFleetSnapshotExcludesLoadingRunners(t *testing.T) {
	ctx := t.Context()
	sched := InitScheduler(ctx)

	ready := &runnerRef{
		model:    &Model{ShortName: "ready:latest"},
		modelKey: "ready-key",
		llama:    &mockLlm{},
		loadedMeta: api.LoadedModelMetadata{
			NumCtx:   4096,
			ProbedAt: time.Now().UTC(),
		},
	}
	loading := &runnerRef{
		model:    &Model{ShortName: "loading:latest"},
		modelKey: "loading-key",
		loading:  true,
		llama:    &mockLlm{},
	}
	sched.loadedMu.Lock()
	sched.loaded["ready-key"] = ready
	sched.loaded["loading-key"] = loading
	sched.loadedMu.Unlock()

	snap := sched.InferenceFleetSnapshot()
	if snap.Loaded != 1 {
		t.Fatalf("loaded=%d want 1 (loading runner excluded)", snap.Loaded)
	}
	if len(snap.LoadedModels) != 1 || snap.LoadedModels[0] != "ready:latest" {
		t.Fatalf("loaded_models=%v", snap.LoadedModels)
	}
}

func TestInferenceStatusLoadedModelNames(t *testing.T) {
	ctx := t.Context()
	s := InitScheduler(ctx)

	runner := &runnerRef{
		model:    &Model{ShortName: "llama3:latest"},
		modelKey: "test-key",
		llama:    &mockLlm{},
		loadedMeta: api.LoadedModelMetadata{
			NumCtx:   4096,
			ProbedAt: time.Now().UTC(),
		},
	}
	s.loadedMu.Lock()
	s.loaded["test-key"] = runner
	s.loadedMu.Unlock()

	st := (&Server{sched: s}).inferenceStatus(ctx)
	if st.Ggml.Loaded != 1 {
		t.Fatalf("loaded=%d", st.Ggml.Loaded)
	}
	if len(st.Ggml.LoadedModels) != st.Ggml.Loaded {
		t.Fatalf("loaded=%d loaded_models=%v", st.Ggml.Loaded, st.Ggml.LoadedModels)
	}
	if st.Ggml.LoadedModels[0] != "llama3:latest" {
		t.Fatalf("loaded_models=%v", st.Ggml.LoadedModels)
	}
	if len(st.Ggml.LoadedModelDetails) != 1 {
		t.Fatalf("loaded_model_details=%+v", st.Ggml.LoadedModelDetails)
	}
	if st.Ggml.LoadedModelDetails[0].Name != "llama3:latest" {
		t.Fatalf("detail name=%q", st.Ggml.LoadedModelDetails[0].Name)
	}
	if st.Ggml.LoadedModelDetails[0].NumCtx != 4096 {
		t.Fatalf("detail num_ctx=%d", st.Ggml.LoadedModelDetails[0].NumCtx)
	}
}
