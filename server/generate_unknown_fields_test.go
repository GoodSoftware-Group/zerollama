package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestGenerateHandler_UnknownFieldRejected(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	body := `{"model":"qwen2.5:0.5b","prompt":"hi","__minefield_unvalidated_field_probe__":true}`
	c.Request = &http.Request{
		Method: http.MethodPost,
		URL:    &url.URL{Path: "/api/generate"},
		Body:   io.NopCloser(strings.NewReader(body)),
		Header: http.Header{"Content-Type": []string{"application/json"}},
	}
	s.GenerateHandler(c)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "__minefield_unvalidated_field_probe__") {
		t.Fatalf("body=%s", w.Body.String())
	}
}
