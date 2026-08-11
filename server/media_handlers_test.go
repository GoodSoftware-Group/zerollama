package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/server/media"
	"github.com/ollama/ollama/types/model"
)

func TestMediaHTTPHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	root := t.TempDir()
	setMediaStore(media.New(root, time.Hour, 1<<30))

	s := &Server{}
	r := gin.New()
	r.PUT("/v1/media/:session/:label", s.MediaPutHandler)
	r.HEAD("/v1/media/:session/:label", s.MediaHeadHandler)
	r.GET("/v1/media/:session/:label", s.MediaGetLabelHandler)
	r.DELETE("/v1/media/:session/:label", s.MediaDeleteHandler)
	r.GET("/v1/media/:session", s.MediaListHandler)

	png := tinyPNGBytes()
	req := httptest.NewRequest(http.MethodPut, "/v1/media/anim1/kf0", bytes.NewReader(png))
	req.Header.Set("Content-Type", "image/png")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("PUT status=%d body=%s", w.Code, w.Body.String())
	}
	var put media.PutResult
	if err := json.Unmarshal(w.Body.Bytes(), &put); err != nil {
		t.Fatal(err)
	}
	if put.Digest == "" || put.Kind != media.KindImage {
		t.Fatalf("put result: %+v", put)
	}

	req = httptest.NewRequest(http.MethodPut, "/v1/media/anim1/kf1", bytes.NewReader(png))
	req.Header.Set("Content-Type", "image/png")
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("PUT2 status=%d", w.Code)
	}

	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/v1/media/anim1", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("LIST status=%d", w.Code)
	}
	var list struct {
		Labels []media.LabelInfo `json:"labels"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Labels) != 2 {
		t.Fatalf("labels=%d", len(list.Labels))
	}

	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodHead, "/v1/media/anim1/kf0", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("HEAD status=%d", w.Code)
	}

	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodHead, "/v1/media/anim1/final", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("HEAD missing status=%d", w.Code)
	}

	casEntries, _ := filepath.Glob(filepath.Join(root, "cas", "sha256-*"))
	if len(casEntries) != 1 {
		t.Fatalf("expected 1 cas file, got %d", len(casEntries))
	}
}

func TestBuildWanVideoPayloadKeyframes(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	kfDir := filepath.Join(t.TempDir(), "kf")
	if err := os.MkdirAll(kfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(kfDir, "000.png"), tinyPNGBytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendWan},
		BackendPaths: map[string]string{
			"wan_repo":     filepath.Join(t.TempDir(), "Wan2.2"),
			"wan_ckpt_dir": filepath.Join(t.TempDir(), "ckpt"),
		},
		VideoGeneration: &model.VideoGenerationConfig{
			Profile:    wanProfile22TI2V5B,
			VRAMTier:   "16g",
			Size:       "832x480",
			Frames:     49,
			Steps:      25,
			TimeoutSec: 100,
		},
	}
	_ = os.MkdirAll(cfg.BackendPaths["wan_repo"], 0o755)
	_ = os.MkdirAll(cfg.BackendPaths["wan_ckpt_dir"], 0o755)

	submittedAt := time.Date(2026, 5, 27, 12, 0, 0, 0, time.UTC)
	payload, err := buildWanVideoPayload(cfg, *cfg.VideoGeneration, "wan2.2-ti2v-5b", "motion", nil, submittedAt, kfDir)
	if err != nil {
		t.Fatal(err)
	}
	if payload.Env["VIDEO_KEYFRAME_DIR"] != kfDir {
		t.Fatalf("VIDEO_KEYFRAME_DIR=%q", payload.Env["VIDEO_KEYFRAME_DIR"])
	}
	if payload.Env["WAN_IMAGE"] == "" {
		t.Fatal("expected WAN_IMAGE")
	}
	if payload.Env["VIDEO_CLEANUP_KEYFRAME_DIR"] != "1" {
		t.Fatalf("expected VIDEO_CLEANUP_KEYFRAME_DIR=1, got %q", payload.Env["VIDEO_CLEANUP_KEYFRAME_DIR"])
	}
}

func TestValidateWanKeyframes(t *testing.T) {
	if err := validateWanKeyframes("wan2.1-t2v-1.3b", []string{"a"}); err == nil {
		t.Fatal("expected error for t2v+keyframes")
	}
	if err := validateWanKeyframes("wan2.2-ti2v-5b", []string{"a", "b"}); err != nil {
		t.Fatal(err)
	}
}

func tinyPNGBytes() []byte {
	return []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
		0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
		0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
		0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xfe, 0xd4, 0xef, 0x00, 0x00,
		0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}
}
