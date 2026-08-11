package server

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/trainingworker"
)

func TestClampWanFrames16g(t *testing.T) {
	cfg := model.VideoGenerationConfig{VRAMTier: "16g", Profile: wanProfile22TI2V5B}
	if got := clampWanFrames(cfg, 200, 49); got != wan22MaxFrames16g {
		t.Fatalf("expected cap %d, got %d", wan22MaxFrames16g, got)
	}
	cfg.Profile = wanProfile21T2V13B
	if got := clampWanFrames(cfg, 80, 49); got != 49 {
		t.Fatalf("expected cap 49 for 1.3b, got %d", got)
	}
	// Manifest may raise the 16g ceiling above the profile default.
	cfg.Profile = wanProfile22TI2V5B
	if got := clampWanFrames(cfg, 100, 90); got != 90 {
		t.Fatalf("expected manifest ceiling 90, got %d", got)
	}
}

func TestBuildWanVideoPayload(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)

	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendWan},
		BackendPaths: map[string]string{
			"wan_repo":     filepath.Join(t.TempDir(), "Wan2.1"),
			"wan_ckpt_dir": filepath.Join(t.TempDir(), "ckpt"),
		},
		VideoGeneration: &model.VideoGenerationConfig{
			Profile:    wanProfile21T2V13B,
			VRAMTier:   "16g",
			Size:       "832x480",
			Frames:     49,
			Steps:      25,
			TimeoutSec: 100,
		},
	}
	if err := os.MkdirAll(cfg.BackendPaths["wan_repo"], 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(cfg.BackendPaths["wan_ckpt_dir"], 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfg.BackendPaths["wan_ckpt_dir"], "dummy.pth"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	wantVenv := filepath.Join(filepath.Dir(cfg.BackendPaths["wan_repo"]), "venv")
	if err := os.MkdirAll(filepath.Join(wantVenv, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wantVenv, "bin", "python3"), []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	submittedAt := time.Date(2026, 5, 27, 12, 0, 0, 0, time.UTC)
	payload, err := buildWanVideoPayload(cfg, *cfg.VideoGeneration, "wan2.1-t2v", "a cat on stage", nil, submittedAt, "")
	if err != nil {
		t.Fatal(err)
	}
	if payload.Timeout != 100 {
		t.Fatalf("timeout: got %d want 100", payload.Timeout)
	}
	if payload.Env["WAN_PROMPT"] != "a cat on stage" {
		t.Fatalf("prompt env: %q", payload.Env["WAN_PROMPT"])
	}
	if payload.OutputPath != filepath.Join(videoArtifactRoot(), "{job_id}.mp4") {
		t.Fatalf("output path: %s", payload.OutputPath)
	}
	if payload.Env["WAN_VENV"] != wantVenv {
		t.Fatalf("WAN_VENV: got %q want %q", payload.Env["WAN_VENV"], wantVenv)
	}
	wantPython := filepath.Join(wantVenv, "bin", "python3")
	if payload.PythonBin != wantPython {
		t.Fatalf("python_bin: got %q want %q", payload.PythonBin, wantPython)
	}
	if payload.Env["WAN_PYTHON"] != wantPython {
		t.Fatalf("WAN_PYTHON: got %q want %q", payload.Env["WAN_PYTHON"], wantPython)
	}
	if payload.VideoModel != "wan2.1-t2v" || payload.VideoSize != "832x480" {
		t.Fatalf("metadata: model=%q size=%q", payload.VideoModel, payload.VideoSize)
	}
	if payload.SubmittedAt != submittedAt.Format(time.RFC3339) {
		t.Fatalf("submitted_at: got %q", payload.SubmittedAt)
	}
	if payload.Env["WAN_CONVERT_MODEL_DTYPE"] != "" {
		t.Fatalf("WAN_CONVERT_MODEL_DTYPE should not be set for 2.1 profile")
	}
	if payload.Env["WAN_SUBPROCESS_TIMEOUT"] != "100" {
		t.Fatalf("WAN_SUBPROCESS_TIMEOUT: got %q", payload.Env["WAN_SUBPROCESS_TIMEOUT"])
	}
	if _, ok := payload.Env["WAN_PRECISION"]; ok {
		t.Fatalf("WAN_PRECISION should not be passed (unused by wrapper)")
	}
}

func TestSafeVideoArtifactPath(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	root := videoArtifactRoot()
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}

	f := filepath.Join(root, "abc.mp4")
	if err := os.WriteFile(f, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := safeVideoArtifactPath(f); err != nil {
		t.Fatalf("expected ok: %v", err)
	}
	if _, err := safeVideoArtifactPath("/etc/passwd"); err == nil {
		t.Fatal("expected reject outside root")
	}
}

func TestVideoFromTrainingJobMapping(t *testing.T) {
	job := map[string]any{
		"jobId":       "deadbeef",
		"status":      "running",
		"progress":    42.0,
		"submittedAt": "2026-05-27T12:00:00Z",
		"videoModel":  "wan2.1-t2v",
		"videoSize":   "832x480",
	}
	raw, _ := json.Marshal(job)
	v, err := openai.VideoFromTrainingJob(raw)
	if err != nil {
		t.Fatal(err)
	}
	if v.ID != "deadbeef" {
		t.Fatalf("expected id deadbeef, got %q", v.ID)
	}
	if v.Status != "in_progress" || v.Progress != 42 {
		t.Fatalf("unexpected video: %+v", v)
	}
	if v.Model != "wan2.1-t2v" || v.Size != "832x480" {
		t.Fatalf("metadata: model=%q size=%q", v.Model, v.Size)
	}
	if v.CreatedAt != 1779883200 {
		t.Fatalf("created_at: got %d", v.CreatedAt)
	}

	cancelled, _ := json.Marshal(map[string]any{
		"jobId":  "cafe",
		"status": "cancelled",
		"error":  "user cancelled",
	})
	cv, err := openai.VideoFromTrainingJob(cancelled)
	if err != nil {
		t.Fatal(err)
	}
	if cv.Status != "cancelled" {
		t.Fatalf("cancelled status: got %q", cv.Status)
	}
	if cv.Error == nil || cv.Error.Code != "video_generation_cancelled" {
		t.Fatalf("cancelled error: %+v", cv.Error)
	}
}

func TestCompletedVideoOutputPath(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "x.mp4")
	if err := os.WriteFile(out, []byte("v"), 0o644); err != nil {
		t.Fatal(err)
	}
	// wire shape: resultJson is a JSON-encoded string (not an inline object)
	resultJSON, _ := json.Marshal(map[string]any{"output_path": out, "status": "ok"})
	job := []byte(`{"status":"completed","resultJson":` + jsonString(string(resultJSON)) + `}`)
	p, _, err := completedVideoOutputFromJob(job)
	if err != nil || p != out {
		t.Fatalf("got %q err %v", p, err)
	}
}

func TestDeferredTrainingJobStatusNotFound(t *testing.T) {
	s := &Server{}
	_, err := s.deferredTrainingJobStatusJSON(context.Background(), "defer-deadbeef")
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, trainingworker.ErrJobNotFound) {
		t.Fatalf("expected ErrJobNotFound, got %v", err)
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "training.py")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("training.py not found")
		}
		dir = parent
	}
}
