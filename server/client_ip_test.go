package server

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/types/model"
)

func TestRequestClientIP(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	var got string
	r.POST("/", func(c *gin.Context) {
		got = requestClientIP(c)
		c.Status(200)
	})
	req := httptest.NewRequest("POST", "/", nil)
	req.RemoteAddr = "203.0.113.9:54321"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if got != "203.0.113.9" {
		t.Fatalf("client_ip=%q", got)
	}
}

func TestInferenceAccessLogIncludesClientIP(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{sched: InitScheduler(t.Context())}
	r := gin.New()
	r.POST("/api/generate", s.inferenceAccessLogMiddleware("/api/generate"), func(c *gin.Context) {
		if ip := clientIPFromContext(c.Request.Context()); ip != "198.51.100.7" {
			t.Errorf("ctx client_ip=%q", ip)
		}
		c.JSON(200, gin.H{"done": true})
	})

	req := httptest.NewRequest("POST", "/api/generate", strings.NewReader(`{"model":"m","stream":false,"prompt":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	req.RemoteAddr = "198.51.100.7:1234"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("status=%d", w.Code)
	}
}

func TestProcessSnapshotLastClientIPUnkeyed(t *testing.T) {
	sched := InitScheduler(t.Context())
	m := &Model{
		ShortName: "openhermes:test",
		Digest:    "abc123",
		Config:    model.ConfigV2{ModelFormat: "gguf"},
	}
	modelKey := schedulerModelKey(m)
	runner := &runnerRef{
		model:     m,
		modelKey:  modelKey,
		loading:   false,
		totalSize: 1,
		vramSize:  1,
	}
	sched.loadedMu.Lock()
	sched.loaded[modelKey] = runner
	sched.loadedMu.Unlock()

	runner.noteLastClientIP("192.168.255.50")
	models := sched.ProcessModelsSnapshot()
	if len(models) != 1 || models[0].Zerollama == nil {
		t.Fatalf("models=%+v", models)
	}
	if models[0].Zerollama.LastClientIP != "192.168.255.50" {
		t.Fatalf("last_client_ip=%q", models[0].Zerollama.LastClientIP)
	}
	if len(models[0].Zerollama.Sessions) != 1 || models[0].Zerollama.Sessions[0].ClientIP != "192.168.255.50" {
		t.Fatalf("sessions=%+v", models[0].Zerollama.Sessions)
	}
}

func TestProcessSessionInfoIncludesClientIP(t *testing.T) {
	slot := &mlxSessionSlot{
		sessionKey: "hermes:agent:1",
		clientIP:   "10.0.0.8",
		projectID:  "",
		hotUntil:   time.Now().Add(time.Minute),
	}
	info := slot.processSessionInfo(time.Now())
	if info.ClientIP != "10.0.0.8" {
		t.Fatalf("client_ip=%q", info.ClientIP)
	}
}
