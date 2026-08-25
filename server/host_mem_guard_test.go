package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestHostMemGuardDisabledPasses(t *testing.T) {
	t.Setenv("ZEROLLAMA_HOST_MEM_GUARD", "0")
	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{}
	r.POST("/api/chat", s.hostMemGuard(), func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}
