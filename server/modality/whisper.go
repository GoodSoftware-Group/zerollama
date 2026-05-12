package modality

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// TranscribeWhisper runs a whisper.cpp-compatible CLI. Expected invocation shape:
//
//	<bin> -m <model.ggml> -f <audio.ext> -otxt -of <outbase> [-l <lang>]
//
// The input file extension is chosen from the original upload name and/or magic bytes.
func TranscribeWhisper(ctx context.Context, cfg model.ConfigV2, audio []byte, originalFilename, language string) (string, error) {
	ctx, cancel := whisperCtx(ctx)
	defer cancel()

	bin := envconfig.WhisperBin()
	modelPath := PathFor(cfg, "whisper_model")
	if modelPath == "" {
		modelPath = envconfig.WhisperModelPath()
	}
	if modelPath == "" {
		return "", fmt.Errorf("set backend_paths.whisper_model in the model config or OLLAMA_WHISPER_MODEL")
	}

	ext := AudioInputExt(originalFilename, audio)

	tmpDir, err := os.MkdirTemp("", "ollama-whisper-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmpDir)

	inPath := filepath.Join(tmpDir, "input."+ext)
	if err := os.WriteFile(inPath, audio, 0o600); err != nil {
		return "", err
	}
	outBase := filepath.Join(tmpDir, "out")
	args := []string{"-m", modelPath, "-f", inPath, "-otxt", "-of", outBase}
	if language != "" {
		args = append(args, "-l", language)
	}
	if extra := envconfig.WhisperExtraArgs(); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}
	cmd := exec.CommandContext(ctx, bin, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return "", fmt.Errorf("whisper: deadline exceeded or cancelled: %w", ctx.Err())
		}
		return "", fmt.Errorf("whisper (%s): %w: %s", bin, err, stderr.String())
	}
	txtData, err := os.ReadFile(outBase + ".txt")
	if err != nil {
		return "", fmt.Errorf("read whisper output: %w", err)
	}
	return strings.TrimSpace(string(txtData)), nil
}
