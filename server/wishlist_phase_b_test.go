package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/server/vram"
)

func TestPrepareRuntimeOptsSkipUnload(t *testing.T) {
	// Covered in vram package; keep a smoke that opts type exists for call sites.
	_ = vram.PrepareRuntimeOpts{SkipUnload: true}
}

func TestProposeLoadSerializeRequired(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	w := httptest.NewRecorder()
	c, r := gin.CreateTestContext(w)
	r.POST("/api/propose-load", s.ProposeLoadHandler)

	body := api.ProposeLoadRequest{
		Models: []api.CanLoadRequest{
			{Model: "m1", Options: map[string]any{"gguf": "/tmp/a.gguf"}},
			{Model: "m2", Options: map[string]any{"gguf": "/tmp/b.gguf"}},
		},
	}
	b, _ := json.Marshal(body)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/propose-load", bytes.NewReader(b))
	c.Request.Header.Set("Content-Type", "application/json")
	s.ProposeLoadHandler(c)
	// Without models on disk, entries fail closed — still returns 200 with plan.
	if w.Code != http.StatusOK && w.Code != http.StatusBadRequest {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
}

func TestPinRejectEmpty(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{sched: &Scheduler{}}
	s.sched.mlxGate = *newMLXAgentGate()
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/pin", bytes.NewReader([]byte(`{"models":[]}`)))
	c.Request.Header.Set("Content-Type", "application/json")
	s.PinHandler(c)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status %d", w.Code)
	}
}

func TestPinProtectsModelKeys(t *testing.T) {
	g := newMLXAgentGate()
	g.mu.Lock()
	g.pins["pin_test"] = &pinLease{
		ID:        "pin_test",
		ModelKeys: []string{"digest:protected"},
		ExpiresAt: time.Now().Add(time.Hour),
	}
	g.mu.Unlock()
	keys := g.protectedModelKeys()
	if _, ok := keys["digest:protected"]; !ok {
		t.Fatalf("expected protected key, got %v", keys)
	}
}

func TestPinExpires(t *testing.T) {
	g := newMLXAgentGate()
	g.mu.Lock()
	g.pins["pin_old"] = &pinLease{
		ID:        "pin_old",
		ModelKeys: []string{"digest:gone"},
		ExpiresAt: time.Now().Add(-time.Minute),
	}
	g.mu.Unlock()
	keys := g.protectedModelKeys()
	if _, ok := keys["digest:gone"]; ok {
		t.Fatalf("expired pin should not protect, got %v", keys)
	}
}

func TestUnpin(t *testing.T) {
	g := newMLXAgentGate()
	g.mu.Lock()
	g.pins["pin_x"] = &pinLease{ID: "pin_x", ExpiresAt: time.Now().Add(time.Hour)}
	g.mu.Unlock()
	if !g.deletePin("pin_x") {
		t.Fatal("expected delete")
	}
	if g.deletePin("pin_x") {
		t.Fatal("second delete should fail")
	}
}

func TestDistinctRuntimeGGUFs(t *testing.T) {
	got := distinctRuntimeGGUFs([]pinModelResolved{
		{Backend: "runtime", GGUF: "/tmp/a.gguf"},
		{Backend: "runtime", GGUF: "/tmp/a.gguf"},
		{Backend: "runtime", GGUF: "/tmp/b.gguf"},
		{Backend: "ggml", GGUF: ""},
	})
	if len(got) != 2 {
		t.Fatalf("got %v", got)
	}
}

func TestUnloadAllRunnersKeepsPinned(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	pinnedLLM := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	otherLLM := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.mlxGate = *newMLXAgentGate()
	s.mlxGate.mu.Lock()
	s.mlxGate.pins["pin_keep"] = &pinLease{
		ID:        "pin_keep",
		ModelKeys: []string{"pinned"},
		ExpiresAt: time.Now().Add(time.Hour),
	}
	s.mlxGate.mu.Unlock()

	s.loadedMu.Lock()
	s.loaded["pinned"] = &runnerRef{llama: pinnedLLM, modelKey: "pinned", numParallel: 1}
	s.loaded["other"] = &runnerRef{llama: otherLLM, modelKey: "other", numParallel: 1}
	s.loadedMu.Unlock()

	s.UnloadAllRunners()

	if pinnedLLM.closeCalled {
		t.Fatal("pinned runner must survive UnloadAllRunners")
	}
	if !otherLLM.closeCalled {
		t.Fatal("unpinned runner should unload")
	}
	s.loadedMu.Lock()
	_, still := s.loaded["pinned"]
	_, gone := s.loaded["other"]
	s.loadedMu.Unlock()
	if !still || gone {
		t.Fatalf("loaded map pinned=%v other_present=%v", still, gone)
	}
}

func TestUnloadAllRunnersForcedClearsPins(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	pinnedLLM := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.mlxGate = *newMLXAgentGate()
	s.mlxGate.mu.Lock()
	s.mlxGate.pins["pin_force"] = &pinLease{
		ID:        "pin_force",
		ModelKeys: []string{"pinned"},
		ExpiresAt: time.Now().Add(time.Hour),
	}
	s.mlxGate.mu.Unlock()

	s.loadedMu.Lock()
	s.loaded["pinned"] = &runnerRef{llama: pinnedLLM, modelKey: "pinned", numParallel: 1}
	s.loadedMu.Unlock()

	s.UnloadAllRunnersForced()

	if !pinnedLLM.closeCalled {
		t.Fatal("forced unload must clear pinned runners")
	}
	if s.ggmlRunnersLoaded() {
		t.Fatal("expected no ggml runners after forced unload")
	}
}

func TestPinBudgetAcrossLeases(t *testing.T) {
	g := newMLXAgentGate()
	g.mu.Lock()
	g.pins["a"] = &pinLease{
		ID:        "a",
		ModelKeys: []string{"k1", "k2"},
		ExpiresAt: time.Now().Add(time.Hour),
	}
	if !g.pinWouldExceedBudgetLocked([]string{"k3"}, 2) {
		t.Fatal("expected budget exceed when adding third distinct key")
	}
	if g.pinWouldExceedBudgetLocked([]string{"k1"}, 2) {
		t.Fatal("overlapping key should not exceed budget")
	}
	if g.pinUniqueKeyCountLocked() != 2 {
		t.Fatalf("unique count %d", g.pinUniqueKeyCountLocked())
	}
	g.mu.Unlock()
}

func TestPrepareRuntimeSkipRequiresEmptyGGML(t *testing.T) {
	s := &Server{sched: InitScheduler(t.Context())}
	s.sched.waitForRecovery = 10 * time.Millisecond
	if s.ggmlRunnersLoaded() {
		t.Fatal("expected empty")
	}
	s.sched.loadedMu.Lock()
	s.sched.loaded["x"] = &runnerRef{llama: &mockLlm{}, numParallel: 1}
	s.sched.loadedMu.Unlock()
	if !s.ggmlRunnersLoaded() {
		t.Fatal("expected loaded")
	}
}

func TestPrepareRuntimeVRAMSkipWhenResidentAndEmpty(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"waiting": 0, "running": 0, "llama_server": true,
				"model_swap": map[string]any{"loaded_gguf": "/tmp/resident.gguf"},
			})
		case "/internal/inference/resume":
			w.WriteHeader(http.StatusOK)
		default:
			http.NotFound(w, r)
		}
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	runtimeHealthCacheMu.Lock()
	runtimeHealthCacheURL = ""
	runtimeHealthCacheMu.Unlock()

	s := &Server{sched: InitScheduler(t.Context())}
	s.sched.waitForRecovery = 10 * time.Millisecond
	other := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	// Empty ggml → skip unload path.
	if err := s.prepareRuntimeVRAM(t.Context(), "/tmp/resident.gguf", false); err != nil {
		t.Fatalf("skip path: %v", err)
	}
	if other.closeCalled {
		t.Fatal("no ggml runners should have been touched")
	}

	// With ggml loaded, must unload (not skip) even when GGUF matches.
	s.sched.loadedMu.Lock()
	s.sched.loaded["other"] = &runnerRef{llama: other, modelKey: "other", numParallel: 1}
	s.sched.loadedMu.Unlock()
	if err := s.prepareRuntimeVRAM(t.Context(), "/tmp/resident.gguf", false); err != nil {
		t.Fatalf("unload path: %v", err)
	}
	if !other.closeCalled {
		t.Fatal("expected ggml unload when runners present")
	}
}

func TestPrepareRuntimeVRAMBlocksPinnedGGML(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/internal/inference/resume" {
			t.Fatal("must not resume runtime while pinned ggml remains")
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	s := &Server{sched: InitScheduler(t.Context())}
	s.sched.waitForRecovery = 10 * time.Millisecond
	s.sched.mlxGate = *newMLXAgentGate()
	s.sched.mlxGate.mu.Lock()
	s.sched.mlxGate.pins["p"] = &pinLease{
		ID: "p", ModelKeys: []string{"pinned"}, ExpiresAt: time.Now().Add(time.Hour),
	}
	s.sched.mlxGate.mu.Unlock()
	pinned := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	s.sched.loadedMu.Lock()
	s.sched.loaded["pinned"] = &runnerRef{llama: pinned, modelKey: "pinned", numParallel: 1}
	s.sched.loadedMu.Unlock()

	err := s.prepareRuntimeVRAM(t.Context(), "/tmp/any.gguf", false)
	if !errors.Is(err, ErrRuntimeVRAMPinnedGGML) {
		t.Fatalf("got %v", err)
	}
	if pinned.closeCalled {
		t.Fatal("pin must survive")
	}
}

func TestPrepareRuntimeVRAMBlocksConflictingGGUFPin(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	s := &Server{sched: InitScheduler(t.Context())}
	s.sched.mlxGate = *newMLXAgentGate()
	s.sched.mlxGate.mu.Lock()
	s.sched.mlxGate.pins["p"] = &pinLease{
		ID: "p", RuntimeGGUFs: []string{"/tmp/pinned.gguf"}, ExpiresAt: time.Now().Add(time.Hour),
	}
	s.sched.mlxGate.mu.Unlock()

	if err := s.prepareRuntimeVRAM(t.Context(), "/tmp/other.gguf", false); !errors.Is(err, ErrRuntimeVRAMPinnedGGUF) {
		t.Fatalf("got %v", err)
	}
	if err := s.prepareRuntimeVRAM(t.Context(), "/tmp/pinned.gguf", false); err != nil {
		t.Fatalf("same GGUF should be allowed: %v", err)
	}
	// Exclusive / force bypasses GGUF pin check.
	if err := s.prepareRuntimeVRAM(t.Context(), "/tmp/other.gguf", true); err != nil {
		t.Fatalf("forceUnload: %v", err)
	}
}

func TestPinWouldConflictRuntimeGGUFAcrossLeases(t *testing.T) {
	g := newMLXAgentGate()
	g.mu.Lock()
	g.pins["a"] = &pinLease{
		ID: "a", RuntimeGGUFs: []string{"/tmp/a.gguf"}, ExpiresAt: time.Now().Add(time.Hour),
	}
	if !g.pinWouldConflictRuntimeGGUFLocked([]string{"/tmp/b.gguf"}) {
		t.Fatal("expected conflict")
	}
	if g.pinWouldConflictRuntimeGGUFLocked([]string{"/tmp/a.gguf"}) {
		t.Fatal("same GGUF should not conflict")
	}
	g.mu.Unlock()
}

func TestAbortPrepareRuntimeVRAMFailed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	if !s.abortIfPrepareRuntimeVRAMFailed(c, ErrRuntimeVRAMPinnedGGUF) {
		t.Fatal("expected abort")
	}
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d", w.Code)
	}
	if w.Header().Get("Retry-After") != "2" {
		t.Fatalf("Retry-After=%q", w.Header().Get("Retry-After"))
	}
}
