package modality

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestGenerateExternalImageSDEnv(t *testing.T) {
	t.Setenv("OLLAMA_EXTERNAL_IMAGE_BIN", filepath.Join(t.TempDir(), "missing"))
	trueVal := true
	falseVal := false
	cfg := model.ConfigV2{
		BackendPaths: map[string]string{
			"sd_cli":   "~/bin/sd-cli",
			"sd_model": "~/models/test.gguf",
		},
		ImageGeneration: &model.ImageGenerationConfig{
			Steps:       15,
			CFGScale:    6.5,
			Sampler:     "euler",
			DiffusionFA: &trueVal,
			VAEOnCPU:    &trueVal,
			VAETiling:   &falseVal,
		},
	}
	env := sdExternalImageEnv(cfg)
	m := map[string]string{}
	for _, e := range env {
		k, v, _ := splitEnvPair(e)
		m[k] = v
	}
	if m["OLLAMA_SD_STEPS"] != "15" {
		t.Fatalf("steps = %q", m["OLLAMA_SD_STEPS"])
	}
	if m["OLLAMA_SD_DIFFUSION_FA"] != "1" {
		t.Fatalf("diffusion_fa = %q", m["OLLAMA_SD_DIFFUSION_FA"])
	}
	if m["OLLAMA_SD_VAE_TILING"] != "0" {
		t.Fatalf("vae_tiling = %q", m["OLLAMA_SD_VAE_TILING"])
	}
}

func TestGenerateExternalImageHook(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "out.png")
	// Minimal PNG
	png := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
		0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
		0x54, 0x08, 0xd7, 0x63, 0xf8, 0x0f, 0x00, 0x01,
		0x01, 0x01, 0x00, 0x18, 0xdd, 0x8d, 0xb4, 0x00,
		0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
		0x42, 0x60, 0x82,
	}
	script := filepath.Join(dir, "fake_sd.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\ncp \""+filepath.Join(dir, "seed.png")+"\" \"$OLLAMA_IMAGE_OUTPUT\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "seed.png"), png, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_EXTERNAL_IMAGE_BIN", script)
	_ = out
	got, err := GenerateExternalImage(context.Background(), model.ConfigV2{}, "test", 512, 512, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 8 || got[0] != 0x89 {
		t.Fatalf("expected png bytes, got len=%d", len(got))
	}
}