package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestLlamaCompatSlotsEmpty(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{sched: InitScheduler(t.Context())}
	got := s.llamaCompatSlots()
	if len(got) != 0 {
		t.Fatalf("empty sched: got %d slots, want 0", len(got))
	}
}

func TestLlamaCompatSlotsFromLoadedRunner(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")
	s := &Server{sched: InitScheduler(t.Context())}
	runner := &runnerRef{
		model:     &Model{Name: "qwen3.6:27b-mlx", ShortName: "qwen3.6:27b-mlx", Config: model.ConfigV2{ModelFormat: "safetensors"}},
		modelKey:  "digest:test",
		Options:   &api.Options{Runner: api.Runner{NumCtx: 32768}},
		expiresAt: time.Now().Add(time.Hour),
	}
	s.sched.loadedMu.Lock()
	s.sched.loaded[runner.modelKey] = runner
	s.sched.loadedMu.Unlock()

	got := s.llamaCompatSlots()
	if len(got) < 1 {
		t.Fatal("expected at least one slot")
	}
	if got[0].NCtx != 32768 {
		t.Fatalf("n_ctx=%d, want 32768", got[0].NCtx)
	}
	if got[0].IsProcessing {
		t.Fatal("idle runner should not be processing")
	}

	w := httptest.NewRecorder()
	c, r := gin.CreateTestContext(w)
	r.GET("/slots", s.SlotsHandler)
	c.Request = httptest.NewRequest(http.MethodGet, "/slots", nil)
	r.ServeHTTP(w, c.Request)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var slots []llamaCompatSlot
	if err := json.Unmarshal(w.Body.Bytes(), &slots); err != nil {
		t.Fatal(err)
	}
	if len(slots) < 1 || slots[0].NCtx != 32768 {
		t.Fatalf("handler body=%s", w.Body.String())
	}
}

func TestLlamaCompatSlotsLoadingMarksProcessing(t *testing.T) {
	s := &Server{sched: InitScheduler(t.Context())}
	runner := &runnerRef{
		model:    &Model{Name: "gemma4:26b-optiq", ShortName: "gemma4:26b-optiq", Config: model.ConfigV2{ModelFormat: "safetensors"}},
		modelKey: "digest:loading",
		Options:  &api.Options{Runner: api.Runner{NumCtx: 131072}},
		loading:  true,
	}
	s.sched.loadedMu.Lock()
	s.sched.loaded[runner.modelKey] = runner
	s.sched.loadedMu.Unlock()

	got := s.llamaCompatSlots()
	if len(got) != 1 {
		t.Fatalf("MLX should expose 1 slot, got %d", len(got))
	}
	if !got[0].IsProcessing {
		t.Fatal("loading runner should set is_processing")
	}
	if got[0].NCtx != 131072 {
		t.Fatalf("n_ctx=%d, want 131072", got[0].NCtx)
	}
}
