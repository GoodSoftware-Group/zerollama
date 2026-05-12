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

// SpeechPiper runs Piper TTS. Resolves ONNX path from backend_paths.piper_model, or
// backend_paths.piper_voice_<voice> when voice is set (voice name sanitized to [a-z0-9]+).
// Optional speed maps to Piper --length_scale (inverse: faster speech => smaller scale).
func SpeechPiper(ctx context.Context, cfg model.ConfigV2, text, voice string, speed *float64) ([]byte, string, error) {
	ctx, cancel := piperCtx(ctx)
	defer cancel()

	modelPath := resolvePiperModelPath(cfg, voice)
	if modelPath == "" {
		return nil, "", fmt.Errorf("set backend_paths.piper_model or piper_voice_<name> for the requested voice")
	}
	bin := envconfig.PiperBin()

	tmpDir, err := os.MkdirTemp("", "ollama-piper-*")
	if err != nil {
		return nil, "", err
	}
	defer os.RemoveAll(tmpDir)

	outPath := filepath.Join(tmpDir, "out.wav")
	args := []string{"--model", modelPath, "--output_file", outPath}
	if p := PathFor(cfg, "piper_config"); p != "" {
		args = append(args, "--config", p)
	}
	if speed != nil && *speed > 0 {
		// OpenAI speed: 0.25–4.0, default 1.0. Piper length_scale: higher = slower.
		scale := 1.0 / *speed
		scale = max(0.25, min(4.0, scale))
		args = append(args, "--length_scale", fmt.Sprintf("%.4f", scale))
	}
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Stdin = strings.NewReader(text)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, "", fmt.Errorf("piper: deadline exceeded or cancelled: %w", ctx.Err())
		}
		return nil, "", fmt.Errorf("piper (%s): %w: %s", bin, err, stderr.String())
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		return nil, "", fmt.Errorf("read piper output: %w", err)
	}
	return data, "audio/wav", nil
}

func resolvePiperModelPath(cfg model.ConfigV2, voice string) string {
	base := PathFor(cfg, "piper_model")
	voice = strings.TrimSpace(strings.ToLower(voice))
	if voice == "" {
		return base
	}
	key := "piper_voice_" + sanitizeVoiceKey(voice)
	if p := PathFor(cfg, key); p != "" {
		return p
	}
	return base
}

func sanitizeVoiceKey(v string) string {
	var b strings.Builder
	for _, r := range v {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_':
			b.WriteRune(r)
		}
	}
	return b.String()
}
