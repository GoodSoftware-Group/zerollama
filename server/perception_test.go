package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/envconfig"
)

func TestDetectWireOnlyWhenURLUnset(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_DETECT_URL", "")
	require.Empty(t, envconfig.DetectURL())

	s := &Server{}
	r := gin.New()
	r.POST("/v1/detect", s.DetectHandler)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/detect", strings.NewReader(`{"image":"aGVsbG8=","queries":["cup"]}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	var body map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &body))
	require.Equal(t, "wire_only", body["status"])
	require.Empty(t, body["detections"])
	require.NotContains(t, w.Body.String(), "404 page not found")
}

func TestDetectRequiresImage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_DETECT_URL", "")
	s := &Server{}
	r := gin.New()
	r.POST("/v1/detect", s.DetectHandler)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/detect", strings.NewReader(`{"queries":["cup"]}`))
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusBadRequest, w.Code)
}

func TestDetectProxiesToSibling(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var gotPath string
	var gotBody map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","model":"owlv2","detections":[{"label":"cup","score":0.9,"box":[1,2,3,4]}]}`))
	}))
	defer up.Close()
	t.Setenv("ZEROLLAMA_DETECT_URL", up.URL)

	s := &Server{}
	r := gin.New()
	r.POST("/v1/detect", s.DetectHandler)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/detect", strings.NewReader(`{"image":"abc","queries":["cup"],"threshold":0.2}`))
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, "/v1/detect", gotPath)
	require.Equal(t, "abc", gotBody["image"])
	var body map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &body))
	require.Equal(t, "ok", body["status"])
}

func TestSegmentWireOnlyIsNotGraspReady(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_SEGMENT_URL", "")
	s := &Server{}
	r := gin.New()
	r.POST("/v1/segment", s.SegmentHandler)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/segment", strings.NewReader(`{"image":"abc","box":[0,0,10,10]}`))
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	var body map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &body))
	require.Equal(t, "wire_only", body["status"])
	require.Equal(t, false, body["grasp_ready"])
}
