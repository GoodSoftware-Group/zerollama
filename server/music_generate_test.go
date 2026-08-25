package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

func music3TestRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(wd, ".."))
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", root)
	t.Setenv("ZEROLLAMA_MUSIC_PYTHON", "")
	return root
}

func TestBuildMusic3JobPayload(t *testing.T) {
	root := music3TestRoot(t)
	tmp := t.TempDir()
	dur := 30.0
	steps := 30
	seed := int64(7)
	tokens := 250
	req := openai.SpeechCreateRequest{
		Model:        "minimax-music3:lab",
		Input:        "[Verse]\nHi",
		Instructions: "Warm acoustic pop",
		Duration:     &dur,
		Steps:        &steps,
		Seed:         &seed,
		MaxNewTokens: &tokens,
	}
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalitySpeech: model.BackendMusic3},
		BackendPaths:     map[string]string{"music3_mlx_model": tmp},
		MusicGeneration:  &model.MusicGenerationConfig{DurationSec: 10, Steps: 30, TimeoutSec: 120},
	}
	p, err := buildMusic3JobPayload(cfg, req, time.Unix(1, 0).UTC())
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(p.ScriptPath) || filepath.Base(p.ScriptPath) != "music3_mlx_generate.py" {
		t.Fatalf("script %q", p.ScriptPath)
	}
	if p.Env["MUSIC3_LYRICS"] != "[Verse]\nHi" {
		t.Fatalf("lyrics %q", p.Env["MUSIC3_LYRICS"])
	}
	if p.Env["MUSIC3_DURATION"] != "30" {
		t.Fatalf("explicit duration must win over max_new_tokens, got %q", p.Env["MUSIC3_DURATION"])
	}
	if p.Env["MUSIC3_OUTPUT_PATH"] != filepath.Join(videoArtifactRoot(), "{job_id}.wav") {
		t.Fatalf("output %q", p.Env["MUSIC3_OUTPUT_PATH"])
	}
	if p.Env["MUSIC3_SEED"] != "7" {
		t.Fatalf("seed %q", p.Env["MUSIC3_SEED"])
	}
	if p.Timeout != 120 {
		t.Fatalf("timeout %d", p.Timeout)
	}
	venvPy := filepath.Join(root, ".venv-music", "bin", "python")
	if _, err := os.Stat(venvPy); err == nil && p.PythonBin != venvPy {
		t.Fatalf("python %q want %q", p.PythonBin, venvPy)
	}

	req.Duration = nil
	p2, err := buildMusic3JobPayload(cfg, req, time.Unix(1, 0).UTC())
	if err != nil {
		t.Fatal(err)
	}
	if p2.Env["MUSIC3_DURATION"] != "10" {
		t.Fatalf("max_new_tokens 250 → 10s, got %q", p2.Env["MUSIC3_DURATION"])
	}
}

func TestResolveMusic3PythonOverride(t *testing.T) {
	root := music3TestRoot(t)
	t.Setenv("ZEROLLAMA_MUSIC_PYTHON", "/opt/custom/python")
	if got := resolveMusic3Python(root); got != "/opt/custom/python" {
		t.Fatalf("got %q", got)
	}
}

func TestExpandRunScriptJobIDParity(t *testing.T) {
	// Mirrors training.py _expand_run_script_job_id: every env string, not only WAN_*.
	env := map[string]string{
		"WAN_OUTPUT_PATH":    "/m/{job_id}.mp4",
		"MUSIC3_OUTPUT_PATH": "/m/{job_id}.wav",
	}
	jobID := "abc123"
	for k, v := range env {
		if strings.Contains(v, "{job_id}") {
			env[k] = strings.ReplaceAll(v, "{job_id}", jobID)
		}
	}
	if env["MUSIC3_OUTPUT_PATH"] != "/m/abc123.wav" || env["WAN_OUTPUT_PATH"] != "/m/abc123.mp4" {
		t.Fatalf("%v", env)
	}
}
