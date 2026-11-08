package modality

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/whispercli"
	"github.com/ollama/ollama/types/model"
)

// WhisperAhead is a whisper.cpp process started before the clip is committed.
// The model loads while the caller still holds the audio. Finish writes the WAV.
type WhisperAhead struct {
	proc *whispercli.Ahead
	dir  string
}

// StartWhisperAhead launches the CLI with stdin as the clip. language empty
// leaves the CLI language default.
func StartWhisperAhead(ctx context.Context, cfg model.ConfigV2, language string) (*WhisperAhead, error) {
	modelPath := strings.TrimSpace(PathFor(cfg, "whisper_model"))
	if modelPath == "" {
		modelPath = strings.TrimSpace(envconfig.WhisperModelPath())
	}
	if modelPath == "" {
		return nil, fmt.Errorf("set backend_paths.whisper_model in the model config or OLLAMA_WHISPER_MODEL")
	}
	dir, err := os.MkdirTemp("", "ollama-whisper-live-*")
	if err != nil {
		return nil, err
	}
	outBase := filepath.Join(dir, "out")
	args := whispercli.TextArgs(modelPath, "-", outBase, language)
	if extra := strings.TrimSpace(envconfig.WhisperExtraArgs()); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}
	proc, err := whispercli.StartAhead(ctx, envconfig.WhisperBin(), args, outBase)
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	return &WhisperAhead{proc: proc, dir: dir}, nil
}

// Finish writes the WAV and returns the transcript text.
func (w *WhisperAhead) Finish(ctx context.Context, wav []byte) (string, error) {
	if w == nil || w.proc == nil {
		return "", fmt.Errorf("whisper is not running")
	}
	return w.proc.Finish(ctx, wav)
}

// Close stops the process and removes its output directory.
func (w *WhisperAhead) Close() {
	if w == nil {
		return
	}
	w.proc.Close()
	_ = os.RemoveAll(w.dir)
}
