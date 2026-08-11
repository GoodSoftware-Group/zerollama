package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestInferenceStatusIncludesConfig(t *testing.T) {
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "1")
	t.Setenv("OLLAMA_MAX_QUEUE", "64")
	t.Setenv("OLLAMA_NUM_PARALLEL", "2")
	t.Setenv("OLLAMA_KEEP_ALIVE", "10m")

	st := (&Server{}).inferenceStatus(context.Background())
	if st.Config.MaxLoadedConfigured != 1 {
		t.Fatalf("max_loaded_configured=%d", st.Config.MaxLoadedConfigured)
	}
	if st.Config.MaxLoadedModels != 1 {
		t.Fatalf("max_loaded_models=%d", st.Config.MaxLoadedModels)
	}
	if st.Config.MaxQueue != 64 {
		t.Fatalf("max_queue=%d", st.Config.MaxQueue)
	}
	if st.Config.NumParallel != 2 {
		t.Fatalf("num_parallel=%d", st.Config.NumParallel)
	}
	if st.Config.SameModelMultiCopy {
		t.Fatal("expected same_model_multi_copy false")
	}
	if !st.Config.NumParallelMeansSlots {
		t.Fatal("expected num_parallel_means_slots true")
	}
	if st.Config.ResidencyOwner != "go_sched" {
		t.Fatalf("residency_owner=%q", st.Config.ResidencyOwner)
	}
	if st.Config.KeepAlive == "" {
		t.Fatal("expected keep_alive")
	}
}

func TestStatusHandlerIncludesConfig(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "2")

	s := Server{}
	w := createRequest(t, s.StatusHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	var resp api.StatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Inference.Config.MaxLoadedConfigured != 2 {
		t.Fatalf("config=%+v", resp.Inference.Config)
	}
}

func TestCanLoadHandlerMissingModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/can-load", strings.NewReader(`{}`))
	c.Request.Header.Set("Content-Type", "application/json")
	s.CanLoadHandler(c)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status %d body=%s", w.Code, w.Body.String())
	}
}

func TestCanLoadHeuristicUnknownModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "1")
	t.Setenv("OLLAMA_MAX_QUEUE", "512")

	s := Server{}
	w := createRequest(t, s.CanLoadHandler, api.CanLoadRequest{Model: "missing-model-xyz:latest"})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body=%s", w.Code, w.Body.String())
	}
	var resp api.CanLoadResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Confidence != canLoadConfidenceHeuristic {
		t.Fatalf("confidence=%q", resp.Confidence)
	}
	if resp.Backend != "ggml" {
		t.Fatalf("backend=%q", resp.Backend)
	}
	if resp.CanLoad {
		t.Fatal("expected can_load false for missing model")
	}
	if !strings.Contains(resp.Notes, "heuristic") {
		t.Fatalf("notes=%q", resp.Notes)
	}
}

func TestCanLoadRuntimeExactWithEstimate(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"waiting": 0, "running": 0, "llama_server": false, "inference_state": "idle",
			})
		case "/internal/vram-estimate":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"vram_estimate": map[string]any{
					"gguf": "/tmp/x.gguf",
					"topology": map[string]any{
						"device_count":    2,
						"tensor_parallel": 2,
						"split_mode":      "tensor",
						"tensor_split":    []any{1.0, 1.0},
						"main_gpu":        0,
					},
				},
				"vram_budget": map[string]any{
					"fits": true, "fits_with_margin": true, "suggested_max_num_ctx": 8192,
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	s := Server{}
	w := createRequest(t, s.CanLoadHandler, api.CanLoadRequest{
		Model:   "any:tag",
		Options: map[string]any{"gguf": "/tmp/x.gguf", "num_ctx": 4096},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body=%s", w.Code, w.Body.String())
	}
	var resp api.CanLoadResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Backend != "runtime" || resp.Confidence != canLoadConfidenceExact {
		t.Fatalf("backend=%q confidence=%q", resp.Backend, resp.Confidence)
	}
	if !resp.CanLoad {
		t.Fatalf("expected can_load true, resp=%+v", resp)
	}
	if resp.AlreadyLoaded {
		t.Fatal("expected already_loaded false when runtime has no matching GGUF")
	}
	if resp.SuggestedMaxNumCtx == nil || *resp.SuggestedMaxNumCtx != 8192 {
		t.Fatalf("suggested=%v", resp.SuggestedMaxNumCtx)
	}
	if resp.DeviceCount != 2 || resp.TensorParallel != 2 || resp.SplitMode != "tensor" {
		t.Fatalf("topology device=%d tp=%d split=%q", resp.DeviceCount, resp.TensorParallel, resp.SplitMode)
	}
	if len(resp.TensorSplit) != 2 {
		t.Fatalf("tensor_split=%v", resp.TensorSplit)
	}
}

func TestCanLoadRuntimeFailClosedWhenEstimateMissing(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"waiting": 0, "running": 0, "llama_server": false,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	s := Server{}
	w := createRequest(t, s.CanLoadHandler, api.CanLoadRequest{
		Model:   "any:tag",
		Options: map[string]any{"gguf": "/tmp/missing-estimate.gguf"},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	var resp api.CanLoadResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.CanLoad {
		t.Fatalf("expected fail closed, got %+v", resp)
	}
	if !strings.Contains(resp.Notes, "fail closed") {
		t.Fatalf("notes=%q", resp.Notes)
	}
}

func TestCanLoadRuntimeAlreadyLoadedOnlyOnPathMatch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"waiting": 0, "running": 0, "llama_server": true,
				"model_swap": map[string]any{"loaded_gguf": "/tmp/resident.gguf"},
			})
		case "/internal/vram-estimate":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"vram_estimate": map[string]any{},
				"vram_budget":   map[string]any{"fits": true, "fits_with_margin": true},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	runtimeHealthCacheMu.Lock()
	runtimeHealthCacheURL = ""
	runtimeHealthCacheMu.Unlock()

	s := Server{}
	other := createRequest(t, s.CanLoadHandler, api.CanLoadRequest{
		Model:   "other:tag",
		Options: map[string]any{"gguf": "/tmp/other.gguf"},
	})
	var otherResp api.CanLoadResponse
	if err := json.NewDecoder(other.Body).Decode(&otherResp); err != nil {
		t.Fatal(err)
	}
	if otherResp.AlreadyLoaded {
		t.Fatalf("other GGUF must not report already_loaded: %+v", otherResp)
	}

	same := createRequest(t, s.CanLoadHandler, api.CanLoadRequest{
		Model:   "resident:tag",
		Options: map[string]any{"gguf": "/tmp/resident.gguf"},
	})
	var sameResp api.CanLoadResponse
	if err := json.NewDecoder(same.Body).Decode(&sameResp); err != nil {
		t.Fatal(err)
	}
	if !sameResp.AlreadyLoaded {
		t.Fatalf("expected already_loaded for matching GGUF: %+v", sameResp)
	}
}

func TestIsHostUnstableErrorTight(t *testing.T) {
	if !isHostUnstableError("runner exited unexpectedly") {
		t.Fatal("expected runner exited")
	}
	if isHostUnstableError("failed to configure llama server binary path") {
		t.Fatal("broad 'llama server' must not match")
	}
	if isHostUnstableError("subprocess config invalid") {
		t.Fatal("broad subprocess must not match")
	}
}

func TestGgufPathsEqual(t *testing.T) {
	if !ggufPathsEqual("/tmp/a.gguf", "/tmp/a.gguf") {
		t.Fatal("same path")
	}
	if ggufPathsEqual("/tmp/a.gguf", "/tmp/b.gguf") {
		t.Fatal("different paths")
	}
	if ggufPathsEqual("", "/tmp/a.gguf") {
		t.Fatal("empty")
	}
}

func TestMetricsHandlerPrometheusText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	metricsIncQueueReject()
	s := Server{}
	w := createRequest(t, s.MetricsHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "zerollama_ggml_pending") {
		t.Fatalf("missing gauge: %s", body)
	}
	if !strings.Contains(body, "zerollama_inference_queue_rejects_total") {
		t.Fatalf("missing counter: %s", body)
	}
	ct := w.Header().Get("Content-Type")
	if !strings.Contains(ct, "text/plain") {
		t.Fatalf("content-type=%q", ct)
	}
}

func TestWriteBusyUnavailableRetryAfter(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/generate", nil)
	writeBusyUnavailable(c, ErrMaxQueue.Error())
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d", w.Code)
	}
	if w.Header().Get("Retry-After") != "2" {
		t.Fatalf("Retry-After=%q", w.Header().Get("Retry-After"))
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["retry_after"] != float64(2) {
		t.Fatalf("body=%v", body)
	}
}

func TestClassifyEmptyGenerationTable(t *testing.T) {
	tests := []struct {
		name string
		in   emptyGenInput
		want emptyGenClass
	}{
		{"thinking_only", emptyGenInput{Thinking: "hmm", LoadDone: true, NumPredict: 128}, emptyGenOK},
		{"eval_count", emptyGenInput{EvalCount: 3, LoadDone: true, NumPredict: 128}, emptyGenOK},
		{"short_predict", emptyGenInput{LoadDone: true, NumPredict: 1}, emptyGenEmpty},
		{"empty_normal", emptyGenInput{LoadDone: true, NumPredict: 128}, emptyGenEmpty},
		{"not_loaded", emptyGenInput{LoadDone: false, NumPredict: 128}, emptyGenOK},
		{"runner_exit", emptyGenInput{StreamError: "runner exited unexpectedly"}, emptyGenUnstable},
		{"normal_text", emptyGenInput{Response: "hi", EvalCount: 1, LoadDone: true}, emptyGenOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := classifyEmptyGeneration(tt.in); got != tt.want {
				t.Fatalf("got %v want %v", got, tt.want)
			}
		})
	}
}

func TestEnqueueGenerateStreamErrorIncludesCauseAndTimings(t *testing.T) {
	ch := make(chan any, 4)
	var sentDone bool
	extra := inferenceErrorExtra{
		TotalDuration: 1500 * time.Millisecond,
		LoadDuration:  200 * time.Millisecond,
	}
	enqueueGenerateStreamErrorExtra(ch, "m", &sentDone, "runner exited", 0, extra)
	close(ch)

	var errBody gin.H
	for v := range ch {
		if h, ok := v.(gin.H); ok {
			errBody = h
		}
	}
	if errBody["error"] != "runner exited" {
		t.Fatalf("body=%v", errBody)
	}
	if errBody["cause"] != causeHostUnstable {
		t.Fatalf("cause=%v", errBody["cause"])
	}
	if errBody["total_duration"] == nil {
		t.Fatal("expected total_duration")
	}
}

func TestErrorExtraFromCheckpoints(t *testing.T) {
	start := time.Now().Add(-2 * time.Second)
	loaded := start.Add(100 * time.Millisecond)
	ft := start.Add(500 * time.Millisecond)
	extra := errorExtraFromCheckpoints(start, loaded, ft, true)
	if extra.LoadDuration <= 0 || !extra.HasTTFT || extra.TimeToFirstToken <= 0 {
		t.Fatalf("%+v", extra)
	}
}
