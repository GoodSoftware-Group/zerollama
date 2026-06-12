package server

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestInferencePeekRequestBody(t *testing.T) {
	model, stream := inferencePeekRequestBody([]byte(`{"model":"llama3.2:3b","stream":true,"prompt":"hi"}`))
	if model != "llama3.2:3b" || !stream {
		t.Fatalf("model=%q stream=%v", model, stream)
	}
}

func TestInferenceQueueLogAttrs(t *testing.T) {
	attrs := inferenceQueueLogAttrs(inferenceQueueSnapshot{
		GgmlPending: 2,
		GgmlActive:  1,
		GgmlLoaded:  1,
	})
	if len(attrs) != 8 {
		t.Fatalf("attrs=%v", attrs)
	}
}

func TestInferenceAccessLogMiddleware(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{sched: InitScheduler(t.Context())}
	r := gin.New()
	r.POST("/api/generate", s.inferenceAccessLogMiddleware("/api/generate"), func(c *gin.Context) {
		recordInferenceCompletion(c, "stop", 10, 5)
		c.JSON(200, gin.H{"done": true})
	})

	req := httptest.NewRequest("POST", "/api/generate", strings.NewReader(`{"model":"m","stream":false,"prompt":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("status=%d", w.Code)
	}
}
