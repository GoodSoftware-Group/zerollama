package modality

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// GenerateExternalImage runs the configured external image program with environment variables set:
//
//	OLLAMA_IMAGE_PROMPT, OLLAMA_IMAGE_WIDTH, OLLAMA_IMAGE_HEIGHT, OLLAMA_IMAGE_SEED, OLLAMA_IMAGE_OUTPUT
//
// Backends:
//   - stable-diffusion.cpp: OLLAMA_SD_* from backend_paths sd_cli/sd_model (scripts/image/sd_external_image.sh)
//   - OpenVINO GenAI: OLLAMA_OV_* from backend_paths ov_model_dir/ov_python (scripts/image/ov_external_image.sh)
//
// WHY subprocess: diffusion stacks (UNet/VAE) are not ggml chat runners; isolating them keeps
// scheduler state and VRAM accounting for chat models unchanged. WHY per-model bin: Vulkan SD
// uses fleet OLLAMA_EXTERNAL_IMAGE_BIN; OpenVINO overrides via backend_paths.external_image_bin.
//
// The program must write a PNG file to OLLAMA_IMAGE_OUTPUT.
func GenerateExternalImage(ctx context.Context, cfg model.ConfigV2, prompt string, width, height int32, seed int64) ([]byte, error) {
	ctx, cancel := externalImageCtx(ctx)
	defer cancel()

	bin, err := resolveExternalImageBin(cfg)
	if err != nil {
		return nil, err
	}
	tmpDir, err := os.MkdirTemp("", "ollama-extimg-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	outPath := filepath.Join(tmpDir, "out.png")

	env := append(os.Environ(),
		"OLLAMA_IMAGE_PROMPT="+prompt,
		fmt.Sprintf("OLLAMA_IMAGE_WIDTH=%d", width),
		fmt.Sprintf("OLLAMA_IMAGE_HEIGHT=%d", height),
		fmt.Sprintf("OLLAMA_IMAGE_SEED=%d", seed),
		"OLLAMA_IMAGE_OUTPUT="+outPath,
	)
	env = append(env, externalImagePresetEnv(cfg)...)

	cmd := exec.CommandContext(ctx, bin)
	cmd.Env = env
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("external image: deadline exceeded or cancelled: %w", ctx.Err())
		}
		return nil, fmt.Errorf("external image (%s): %w: %s", bin, err, stderr.String())
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		return nil, fmt.Errorf("read generated image: %w", err)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return nil, fmt.Errorf("external image produced empty output at %s", outPath)
	}
	return data, nil
}

func resolveExternalImageBin(cfg model.ConfigV2) (string, error) {
	if p := expandHome(PathFor(cfg, "external_image_bin")); p != "" {
		return p, nil
	}
	backend := BackendFor(cfg, model.ModalityImage)
	if backend == model.BackendOpenVINOImage {
		for _, candidate := range []string{
			"/usr/lib/ollama-zerollama/ov_external_image.sh",
		} {
			if st, err := os.Stat(candidate); err == nil && !st.IsDir() {
				return candidate, nil
			}
		}
	}
	bin := envconfig.ExternalImageBin()
	if bin == "" {
		return "", fmt.Errorf("OLLAMA_EXTERNAL_IMAGE_BIN is not set (required for modality_backends.image=%q)", backend)
	}
	return bin, nil
}

func externalImagePresetEnv(cfg model.ConfigV2) []string {
	var env []string
	env = append(env, sdExternalImageEnv(cfg)...)
	env = append(env, ovExternalImageEnv(cfg)...)
	return env
}

func sdExternalImageEnv(cfg model.ConfigV2) []string {
	var env []string
	if p := expandHome(PathFor(cfg, "sd_cli")); p != "" {
		env = append(env, "OLLAMA_SD_CLI="+p)
	}
	if p := expandHome(PathFor(cfg, "sd_model")); p != "" {
		env = append(env, "OLLAMA_SD_MODEL="+p)
	}
	appendImageGenerationEnv(&env, cfg.ImageGeneration, "OLLAMA_SD")
	return env
}

func ovExternalImageEnv(cfg model.ConfigV2) []string {
	var env []string
	if p := expandHome(PathFor(cfg, "ov_model_dir")); p != "" {
		env = append(env, "OLLAMA_OV_MODEL_DIR="+p)
	}
	if p := expandHome(PathFor(cfg, "ov_python")); p != "" {
		env = append(env, "OLLAMA_OV_PYTHON="+p)
	}
	if p := expandHome(PathFor(cfg, "ov_generate_py")); p != "" {
		env = append(env, "OLLAMA_OV_GENERATE_PY="+p)
	}
	device := PathFor(cfg, "ov_device")
	if device == "" {
		device = "GPU"
	}
	env = append(env, "OLLAMA_OV_DEVICE="+device)
	appendImageGenerationEnv(&env, cfg.ImageGeneration, "OLLAMA_OV")
	return env
}

func appendImageGenerationEnv(env *[]string, ig *model.ImageGenerationConfig, prefix string) {
	if ig == nil {
		return
	}
	if ig.Steps > 0 {
		*env = append(*env, prefix+"_STEPS="+strconv.Itoa(ig.Steps))
	}
	if ig.Width > 0 {
		*env = append(*env, prefix+"_DEFAULT_WIDTH="+strconv.Itoa(ig.Width))
	}
	if ig.Height > 0 {
		*env = append(*env, prefix+"_DEFAULT_HEIGHT="+strconv.Itoa(ig.Height))
	}
	if ig.CFGScale > 0 {
		*env = append(*env, prefix+"_CFG_SCALE="+strconv.FormatFloat(ig.CFGScale, 'f', -1, 64))
	}
	if prefix == "OLLAMA_SD" {
		if ig.Sampler != "" {
			*env = append(*env, prefix+"_SAMPLER="+ig.Sampler)
		}
		if ig.DiffusionFA != nil {
			*env = append(*env, prefix+"_DIFFUSION_FA="+boolStr(*ig.DiffusionFA))
		}
		if ig.VAEOnCPU != nil {
			*env = append(*env, prefix+"_VAE_ON_CPU="+boolStr(*ig.VAEOnCPU))
		}
		if ig.VAETiling != nil {
			*env = append(*env, prefix+"_VAE_TILING="+boolStr(*ig.VAETiling))
		}
	}
}

func expandHome(p string) string {
	if p == "" {
		return ""
	}
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~/"))
		}
	}
	return p
}

func boolStr(v bool) string {
	if v {
		return "1"
	}
	return "0"
}
