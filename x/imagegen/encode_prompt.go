package imagegen

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
)

// ExecuteEncodePrompt runs Qwen3 text encoding in a fresh process.
// WHY subprocess on CUDA: Contiguous(mmap) TE weights do not return VRAM on
// ReleaseStruct in-process, so DiT load OOMs on 16GB. Child exit frees the card.
func ExecuteEncodePrompt(args []string) error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: envconfig.LogLevel()})))

	fs := flag.NewFlagSet("imagegen-encode-prompt", flag.ExitOnError)
	modelName := fs.String("model", "", "model name")
	prompt := fs.String("prompt", "", "positive prompt")
	negative := fs.String("negative-prompt", "", "optional negative prompt")
	output := fs.String("output", "", "path to write positive embedding bin")
	negOutput := fs.String("negative-output", "", "path to write negative embedding bin (CFG)")
	maxLen := fs.Int("max-len", 512, "tokenizer max length before pad-to-32")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelName == "" || *prompt == "" || *output == "" {
		return fmt.Errorf("--model, --prompt, and --output are required")
	}
	if *negative != "" && *negOutput == "" {
		return fmt.Errorf("--negative-output is required with --negative-prompt")
	}

	_ = os.Setenv("MLX_USE_CUDA_GRAPHS", "false")
	_ = os.Setenv("MLX_DISABLE_COMPILE", "1")

	stdout := os.Stdout
	os.Stdout = os.Stderr
	defer func() { os.Stdout = stdout }()

	if err := mlx.InitMLX(); err != nil {
		return fmt.Errorf("mlx init: %w", err)
	}
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}

	m := &zimage.Model{}
	if err := m.EncodePromptToFiles(*modelName, *prompt, *negative, *output, *negOutput, *maxLen); err != nil {
		return err
	}

	os.Stdout = stdout
	return json.NewEncoder(os.Stdout).Encode(struct {
		OK bool `json:"ok"`
	}{OK: true})
}
