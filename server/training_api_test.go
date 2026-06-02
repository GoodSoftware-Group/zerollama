package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/x/trainingworker"
)

func TestTrainHTTPSubmitRejectsWhenInferenceBusy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pending.Push(&LlmRequest{
		ctx:   ctx,
		model: &Model{ModelPath: "/tmp/m"},
	})

	tw := &trainingworker.Client{}
	tw.SetInferenceSubmitGuard((&Server{sched: sched}).checkTrainingSubmitAllowed)
	s := &Server{sched: sched, training: tw}
	r := gin.New()
	s.registerTrainingRoutes(r)

	body, _ := json.Marshal(map[string]any{"kind": "train", "payload": map[string]any{}})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/train/jobs", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusConflict {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}
