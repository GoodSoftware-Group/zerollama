package modality

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestIsExternalImageBackend(t *testing.T) {
	if !IsExternalImageBackend(model.BackendExternalImage) {
		t.Fatal("external-image")
	}
	if !IsExternalImageBackend(model.BackendOpenVINOImage) {
		t.Fatal("openvino-image")
	}
	if IsExternalImageBackend(model.BackendMLXImagegen) {
		t.Fatal("mlx should be false")
	}
}

func TestOvExternalImageEnv(t *testing.T) {
	cfg := model.ConfigV2{
		BackendPaths: map[string]string{
			"ov_model_dir": "/data/sd15-ov",
			"ov_python":    "/usr/bin/python3",
			"ov_device":    "GPU",
		},
		ImageGeneration: &model.ImageGenerationConfig{
			Steps:    15,
			Width:    512,
			Height:   512,
			CFGScale: 7.5,
		},
	}
	env := ovExternalImageEnv(cfg)
	m := map[string]string{}
	for _, e := range env {
		k, v, _ := splitEnvPair(e)
		m[k] = v
	}
	if m["OLLAMA_OV_MODEL_DIR"] != "/data/sd15-ov" {
		t.Fatalf("model_dir=%q", m["OLLAMA_OV_MODEL_DIR"])
	}
	if m["OLLAMA_OV_STEPS"] != "15" {
		t.Fatalf("steps=%q", m["OLLAMA_OV_STEPS"])
	}
	if m["OLLAMA_OV_DEVICE"] != "GPU" {
		t.Fatalf("device=%q", m["OLLAMA_OV_DEVICE"])
	}
}

func splitEnvPair(s string) (string, string, bool) {
	for i := 0; i < len(s); i++ {
		if s[i] == '=' {
			return s[:i], s[i+1:], true
		}
	}
	return "", "", false
}
