package server

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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

func TestWanMMGPDefaults(t *testing.T) {
	t.Setenv("ZEROLLAMA_WAN_MMGP", "")
	t.Setenv("ZEROLLAMA_WAN_MMGP_PROFILE", "")
	t.Setenv("ZEROLLAMA_WAN_MMGP_QUANTIZE", "")

	ti2v := model.VideoGenerationConfig{VRAMTier: "16g", Profile: wanProfile22TI2V5B}
	if got := wanMMGP(ti2v); got != "1" {
		t.Fatalf("16g ti2v WAN_MMGP: got %q want 1", got)
	}
	if got := wanMMGPProfile(ti2v); got != "5" {
		t.Fatalf("16g ti2v profile: got %q want 5", got)
	}
	if got := wanMMGPQuantize(ti2v); got != "0" {
		t.Fatalf("quantize default: got %q want 0", got)
	}

	t2v := model.VideoGenerationConfig{VRAMTier: "16g", Profile: wanProfile21T2V13B}
	if got := wanMMGP(t2v); got != "0" {
		t.Fatalf("16g t2v should not default mmgp: got %q", got)
	}

	t.Setenv("ZEROLLAMA_WAN_MMGP", "0")
	if got := wanMMGP(ti2v); got != "0" {
		t.Fatalf("opt-out: got %q", got)
	}
}

func TestBuildWanVideoPayloadTI2VMMGP(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("ZEROLLAMA_WAN_MMGP", "")
	t.Setenv("ZEROLLAMA_WAN_MMGP_PROFILE", "")
	t.Setenv("ZEROLLAMA_WAN_VAE_CPU", "")
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_GIB", "")
	t.Setenv("ZEROLLAMA_WAN_OMP_NUM_THREADS", "2")
	t.Setenv("ZEROLLAMA_WAN_RLIMIT_AS_GIB", "0")

	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendWan},
		BackendPaths: map[string]string{
			"wan_repo":     filepath.Join(t.TempDir(), "Wan2.2"),
			"wan_ckpt_dir": filepath.Join(t.TempDir(), "ckpt"),
		},
		VideoGeneration: &model.VideoGenerationConfig{
			Profile:    wanProfile22TI2V5B,
			VRAMTier:   "16g",
			Size:       "1280x704",
			Frames:     17,
			Steps:      8,
			TimeoutSec: 600,
		},
	}
	if err := os.MkdirAll(cfg.BackendPaths["wan_repo"], 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(cfg.BackendPaths["wan_ckpt_dir"], 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfg.BackendPaths["wan_ckpt_dir"], "dummy.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	wantVenv := filepath.Join(filepath.Dir(cfg.BackendPaths["wan_repo"]), "venv")
	if err := os.MkdirAll(filepath.Join(wantVenv, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wantVenv, "bin", "python3"), []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	payload, err := buildWanVideoPayload(cfg, *cfg.VideoGeneration, "wan2.2-ti2v-5b", "a cat", nil, time.Now().UTC(), "")
	if err != nil {
		t.Fatal(err)
	}
	if payload.Env["WAN_MMGP"] != "1" {
		t.Fatalf("WAN_MMGP: got %q want 1", payload.Env["WAN_MMGP"])
	}
	if payload.Env["WAN_MMGP_PROFILE"] != "5" {
		t.Fatalf("WAN_MMGP_PROFILE: got %q want 5", payload.Env["WAN_MMGP_PROFILE"])
	}
	if payload.Env["WAN_MMGP_QUANTIZE"] != "0" {
		t.Fatalf("WAN_MMGP_QUANTIZE: got %q want 0", payload.Env["WAN_MMGP_QUANTIZE"])
	}
	if payload.Env["WAN_VAE_CPU"] != "0" {
		t.Fatalf("under mmgp WAN_VAE_CPU should be GPU (0): got %q", payload.Env["WAN_VAE_CPU"])
	}
	if payload.Env["WAN_ALLOW_GPU_VAE"] != "1" {
		t.Fatalf("WAN_ALLOW_GPU_VAE: got %q want 1", payload.Env["WAN_ALLOW_GPU_VAE"])
	}
	if payload.Env["OMP_NUM_THREADS"] == "" {
		t.Fatal("expected OMP_NUM_THREADS containment")
	}
	if payload.Env["WAN_CONVERT_MODEL_DTYPE"] != "true" {
		t.Fatalf("WAN_CONVERT_MODEL_DTYPE: got %q", payload.Env["WAN_CONVERT_MODEL_DTYPE"])
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

func TestClampLtxFrames(t *testing.T) {
	cfg := model.VideoGenerationConfig{VRAMTier: "16g", Profile: ltxProfile13BDistill}
	if got := clampLtxFrames(cfg, 20, 17); got != 17 {
		t.Fatalf("snap: got %d want 17", got)
	}
	if got := clampLtxFrames(cfg, 25, 17); got != 25 {
		t.Fatalf("legal: got %d want 25", got)
	}
	if got := clampLtxFrames(cfg, 200, 17); got != 41 {
		t.Fatalf("16g cap: got %d want 41", got)
	}
}

func TestBuildVideoJobPayloadLTX(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("ZEROLLAMA_LTX_DRY_RUN", "")
	t.Setenv("ZEROLLAMA_LTX_MMGP_PROFILE", "")

	repo := t.TempDir()
	ckpt := filepath.Join(repo, "ckpts")
	venv := filepath.Join(t.TempDir(), "venv")
	if err := os.MkdirAll(ckpt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(venv, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(venv, "bin", "python3"), []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"ltxv_0.9.8_13B_distilled_quanto_bf16_int8.safetensors",
		"ltxv_0.9.7_VAE.safetensors",
	} {
		if err := os.WriteFile(filepath.Join(ckpt, name), make([]byte, 2048), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendLTX},
		BackendPaths: map[string]string{
			"wan2gp_repo":     repo,
			"wan2gp_ckpt_dir": ckpt,
			"wan2gp_venv":     venv,
		},
		VideoGeneration: &model.VideoGenerationConfig{
			Profile:    ltxProfile13BDistill,
			VRAMTier:   "16g",
			Size:       "768x512",
			Frames:     17,
			Steps:      6,
			TimeoutSec: 600,
		},
	}

	payload, err := buildVideoJobPayload(model.BackendLTX, cfg, *cfg.VideoGeneration, "ltxv-13b-distilled:16g", "long prompt", nil, time.Now().UTC(), "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(payload.ScriptPath, "ltx_video_generate.py") {
		t.Fatalf("script: %s", payload.ScriptPath)
	}
	if payload.Env["WAN2GP_REPO"] != repo {
		t.Fatalf("repo env: %s", payload.Env["WAN2GP_REPO"])
	}
	if payload.Env["LTX_MODEL_TYPE"] != "ltxv_distilled" {
		t.Fatalf("model_type: %s", payload.Env["LTX_MODEL_TYPE"])
	}
	if payload.Env["LTX_MMGP_PROFILE"] != "5" {
		t.Fatalf("profile: %s", payload.Env["LTX_MMGP_PROFILE"])
	}
}

func TestBuildVideoJobPayloadLTXMissingPaths(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendLTX},
		BackendPaths:     map[string]string{},
		VideoGeneration: &model.VideoGenerationConfig{
			Profile: ltxProfile13BDistill,
			Frames:  17,
			Steps:   6,
		},
	}
	_, err := buildVideoJobPayload(model.BackendLTX, cfg, *cfg.VideoGeneration, "ltxv", "p", nil, time.Now().UTC(), "")
	if err == nil || !strings.Contains(err.Error(), "wan2gp_repo") {
		t.Fatalf("want wan2gp_repo error, got %v", err)
	}
}

func TestBuildVideoJobPayloadH3Tiny(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("ZEROLLAMA_VIDEO_CLI", "")
	t.Setenv("ZEROLLAMA_WAN_CLI", "")
	dir := t.TempDir()
	cli := filepath.Join(dir, "video-cli")
	if err := os.WriteFile(cli, []byte("#!/bin/true\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	ckpt := filepath.Join(dir, "MiniMax-H3")
	if err := os.MkdirAll(filepath.Join(ckpt, "FL2VA", "video_vae", "source"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendH3},
		BackendPaths: map[string]string{
			"video_cli":   cli,
			"h3_ckpt_dir": ckpt,
		},
		VideoGeneration: &model.VideoGenerationConfig{
			Profile:    h3ProfileTinyT2VA,
			Frames:     5,
			Steps:      2,
			TimeoutSec: 60,
		},
	}
	payload, err := buildVideoJobPayload(model.BackendH3, cfg, *cfg.VideoGeneration, "minimax-h3-tiny:lab", "a fox", nil, time.Now().UTC(), "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(payload.ScriptPath, "wan_c_generate.py") {
		t.Fatalf("script: %s", payload.ScriptPath)
	}
	if payload.Env["VIDEO_FAMILY"] != "h3" {
		t.Fatalf("family: %s", payload.Env["VIDEO_FAMILY"])
	}
	if payload.Env["WAN_CKPT_DIR"] != ckpt {
		t.Fatalf("ckpt: %s", payload.Env["WAN_CKPT_DIR"])
	}
	if payload.Env["WAN_FRAMES"] != "5" || payload.Env["WAN_STEPS"] != "2" {
		t.Fatalf("tiny geometry frames=%s steps=%s", payload.Env["WAN_FRAMES"], payload.Env["WAN_STEPS"])
	}
	if payload.Env["WAN_SIZE"] != "32x32" || payload.VideoSize != "32x32" {
		t.Fatalf("tiny size=%s video_size=%s", payload.Env["WAN_SIZE"], payload.VideoSize)
	}
}

func TestBuildVideoJobPayloadH3RelativeCLI(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("ZEROLLAMA_VIDEO_CLI", "")
	t.Setenv("ZEROLLAMA_WAN_CLI", "")
	cli := filepath.Join(root, "x", "video-c", "video-cli")
	if _, err := os.Stat(cli); err != nil {
		t.Skip("video-cli not built")
	}
	ckpt := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ckpt, "FL2VA", "video_vae", "source"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendH3},
		BackendPaths: map[string]string{
			"video_cli":   "x/video-c/video-cli",
			"h3_ckpt_dir": ckpt,
		},
		VideoGeneration: &model.VideoGenerationConfig{Profile: h3ProfileTinyT2VA, Steps: 2},
	}
	payload, err := buildVideoJobPayload(model.BackendH3, cfg, *cfg.VideoGeneration, "minimax-h3-tiny:lab", "a fox", nil, time.Now().UTC(), "")
	if err != nil {
		t.Fatal(err)
	}
	if payload.Env["VIDEO_CLI"] != cli {
		t.Fatalf("cli: %s want %s", payload.Env["VIDEO_CLI"], cli)
	}
}

func TestResolveVideoGenerationConfigH3(t *testing.T) {
	m := &Model{
		Config: model.ConfigV2{
			VideoGeneration: &model.VideoGenerationConfig{Profile: h3ProfileTinyT2VA},
		},
	}
	cfg, err := resolveVideoGenerationConfig(m, openai.VideoCreateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Frames != h3TinyFrames || cfg.Steps != h3TinyDefaultSteps || cfg.Size != h3TinySize {
		t.Fatalf("got frames=%d steps=%d size=%s", cfg.Frames, cfg.Steps, cfg.Size)
	}
	if cfg.DitLayers != h3DefaultDitLayers {
		t.Fatalf("dit_layers=%d want %d", cfg.DitLayers, h3DefaultDitLayers)
	}
}

func TestResolveVideoGenerationConfigH3768(t *testing.T) {
	m := &Model{
		Config: model.ConfigV2{
			VideoGeneration: &model.VideoGenerationConfig{Profile: h3Profile768T2VA},
		},
	}
	cfg, err := resolveVideoGenerationConfig(m, openai.VideoCreateRequest{Size: "64x64"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Frames != h3TinyFrames || cfg.Steps != h3TinyDefaultSteps || cfg.Size != h3768Size {
		t.Fatalf("got frames=%d steps=%d size=%s", cfg.Frames, cfg.Steps, cfg.Size)
	}
	if cfg.DitLayers != h3DefaultDitLayers {
		t.Fatalf("dit_layers=%d want %d", cfg.DitLayers, h3DefaultDitLayers)
	}
}

func TestBuildVideoJobPayloadH3768(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("ZEROLLAMA_VIDEO_CLI", "")
	t.Setenv("ZEROLLAMA_WAN_CLI", "")
	dir := t.TempDir()
	cli := filepath.Join(dir, "video-cli")
	if err := os.WriteFile(cli, []byte("#!/bin/true\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	ckpt := filepath.Join(dir, "MiniMax-H3")
	if err := os.MkdirAll(filepath.Join(ckpt, "FL2VA", "video_vae", "source"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendH3},
		BackendPaths: map[string]string{
			"video_cli":   cli,
			"h3_ckpt_dir": ckpt,
		},
		VideoGeneration: &model.VideoGenerationConfig{
			Profile:    h3Profile768T2VA,
			Size:       h3768Size,
			Frames:     5,
			Steps:      2,
			TimeoutSec: 0,
		},
	}
	payload, err := buildVideoJobPayload(model.BackendH3, cfg, *cfg.VideoGeneration, "minimax-h3-768:lab", "a fox", nil, time.Now().UTC(), "")
	if err != nil {
		t.Fatal(err)
	}
	if payload.Env["WAN_SIZE"] != h3768Size || payload.VideoSize != h3768Size {
		t.Fatalf("size=%s video_size=%s", payload.Env["WAN_SIZE"], payload.VideoSize)
	}
	if payload.Env["WAN_FRAMES"] != "5" || payload.Timeout != h3768TimeoutSec {
		t.Fatalf("frames=%s timeout=%d", payload.Env["WAN_FRAMES"], payload.Timeout)
	}
	if payload.Env["VIDEO_H3_LAYERS"] != strconv.Itoa(h3DefaultDitLayers) {
		t.Fatalf("layers=%s", payload.Env["VIDEO_H3_LAYERS"])
	}
}

func TestBuildVideoJobPayloadH3MissingCLI(t *testing.T) {
	root := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("ZEROLLAMA_VIDEO_CLI", "")
	t.Setenv("ZEROLLAMA_WAN_CLI", "")
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityVideoGeneration: model.BackendH3},
		BackendPaths:     map[string]string{},
		VideoGeneration:  &model.VideoGenerationConfig{Profile: h3ProfileTinyT2VA},
	}
	_, err := buildVideoJobPayload(model.BackendH3, cfg, *cfg.VideoGeneration, "h3", "p", nil, time.Now().UTC(), "")
	if err == nil || !strings.Contains(err.Error(), "video_cli") {
		t.Fatalf("want video_cli error, got %v", err)
	}
}
