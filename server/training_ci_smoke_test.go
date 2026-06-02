package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/x/trainingworker"
)

// TestTrainingRoutesRegisteredOnlyWithWorker is CPU-only CI smoke (T4): Gin routes
// exist when the training client is wired, without starting embedded Python or CUDA.
func TestTrainingRoutesRegisteredOnlyWithWorker(t *testing.T) {
	gin.SetMode(gin.TestMode)

	r := gin.New()
	(&Server{}).registerTrainingRoutes(r)
	if len(r.Routes()) != 0 {
		t.Fatalf("expected no routes without training client, got %d", len(r.Routes()))
	}

	r2 := gin.New()
	(&Server{training: &trainingworker.Client{}}).registerTrainingRoutes(r2)
	if len(r2.Routes()) == 0 {
		t.Fatal("expected /api/train routes when training client is set")
	}
	paths := make(map[string]bool)
	for _, rt := range r2.Routes() {
		paths[rt.Path] = true
	}
	for _, want := range []string{"/api/train/jobs", "/api/train/status"} {
		if !paths[want] {
			t.Fatalf("missing route %q in %v", want, paths)
		}
	}
}

// TestTrainHTTPStatusHandlerRuns verifies GET /api/train/status is wired (502 without
// embedded Python is expected in CI — no CUDA / libpython required).
func TestTrainHTTPStatusHandlerRuns(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{training: &trainingworker.Client{}}
	r := gin.New()
	s.registerTrainingRoutes(r)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/train/status", nil)
	r.ServeHTTP(w, req)

	if w.Code == http.StatusNotFound {
		t.Fatalf("status route not registered: %d", w.Code)
	}
	if w.Code != http.StatusBadGateway && w.Code != http.StatusOK {
		t.Fatalf("unexpected status=%d body=%s", w.Code, w.Body.String())
	}
}
