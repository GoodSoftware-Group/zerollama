package modality

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/ollama/ollama/envconfig"
)

// GenerateExternalImage runs OLLAMA_EXTERNAL_IMAGE_BIN with environment variables set:
//
//	OLLAMA_IMAGE_PROMPT, OLLAMA_IMAGE_WIDTH, OLLAMA_IMAGE_HEIGHT, OLLAMA_IMAGE_SEED, OLLAMA_IMAGE_OUTPUT
//
// The program must write a PNG file to OLLAMA_IMAGE_OUTPUT.
func GenerateExternalImage(ctx context.Context, prompt string, width, height int32, seed int64) ([]byte, error) {
	ctx, cancel := externalImageCtx(ctx)
	defer cancel()

	bin := envconfig.ExternalImageBin()
	if bin == "" {
		return nil, fmt.Errorf("OLLAMA_EXTERNAL_IMAGE_BIN is not set (required for modality_backends.image=external-image)")
	}
	tmpDir, err := os.MkdirTemp("", "ollama-extimg-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	outPath := filepath.Join(tmpDir, "out.png")
	cmd := exec.CommandContext(ctx, bin)
	cmd.Env = append(os.Environ(),
		"OLLAMA_IMAGE_PROMPT="+prompt,
		fmt.Sprintf("OLLAMA_IMAGE_WIDTH=%d", width),
		fmt.Sprintf("OLLAMA_IMAGE_HEIGHT=%d", height),
		fmt.Sprintf("OLLAMA_IMAGE_SEED=%d", seed),
		"OLLAMA_IMAGE_OUTPUT="+outPath,
	)
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
