package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/fleet"
)

func TestAssignHoldRegisterAndStatus(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_SECRET", "hold-secret")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TOKEN", "1")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TTL", "10s")

	now := time.Now().UTC()
	tok, _, _, err := fleet.MintAssignToken("n1", "llama3", now)
	if err != nil {
		t.Fatal(err)
	}

	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/api/fleet/assign-hold", s.AssignHoldHandler)
	r.GET("/api/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, s.statusResponse(c.Request.Context()))
	})

	body, _ := json.Marshal(map[string]string{"token": tok})
	req := httptest.NewRequest(http.MethodPost, "/api/fleet/assign-hold", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("hold status=%d body=%s", w.Code, w.Body.String())
	}

	st := s.inferenceStatus(req.Context())
	if st.Ggml.AssignHolds != 1 {
		t.Fatalf("assign_holds=%d", st.Ggml.AssignHolds)
	}
	if st.Ggml.Pending < 1 {
		t.Fatalf("pending should include hold, got %d", st.Ggml.Pending)
	}

	r2 := gin.New()
	r2.POST("/api/chat", s.assignmentTokenMiddleware(), func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	req2 := httptest.NewRequest(http.MethodPost, "/api/chat", nil)
	req2.Header.Set(fleet.AssignmentTokenHeader, tok)
	w2 := httptest.NewRecorder()
	r2.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("chat status=%d", w2.Code)
	}
	if s.ensureAssignHolds().ActiveCount(time.Now()) != 0 {
		t.Fatal("hold should be consumed")
	}
}

func TestAssignHoldExpired(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_SECRET", "hold-secret")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TOKEN", "1")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TTL", "2s")
	tok, _, _, err := fleet.MintAssignToken("n1", "m", time.Now().Add(-5*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/api/fleet/assign-hold", s.AssignHoldHandler)
	body, _ := json.Marshal(map[string]string{"token": tok})
	req := httptest.NewRequest(http.MethodPost, "/api/fleet/assign-hold", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusConflict {
		t.Fatalf("status=%d want 409 body=%s", w.Code, w.Body.String())
	}
}
