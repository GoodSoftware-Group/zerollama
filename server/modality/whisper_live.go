package modality

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/whispercli"
	"github.com/ollama/ollama/types/model"
)

// WhisperLive is one whisper-server for one duplex call.
type WhisperLive struct {
	mu   sync.Mutex
	proc *whispercli.Server
}

// StartWhisperLive launches whisper-server for this call when the binary is
// installed. Nil means the caller should use the CLI ahead path.
func StartWhisperLive(ctx context.Context, cfg model.ConfigV2) *WhisperLive {
	bin := whispercli.LookServer(envconfig.WhisperServer())
	if bin == "" {
		return nil
	}
	modelPath := strings.TrimSpace(PathFor(cfg, "whisper_model"))
	if modelPath == "" {
		modelPath = strings.TrimSpace(envconfig.WhisperModelPath())
	}
	if modelPath == "" {
		return nil
	}
	proc, err := whispercli.StartServer(ctx, bin, modelPath)
	if err != nil {
		return nil
	}
	return &WhisperLive{proc: proc}
}

// Close stops the process.
func (w *WhisperLive) Close() {
	if w == nil {
		return
	}
	w.mu.Lock()
	proc := w.proc
	w.mu.Unlock()
	proc.Close()
}

// Ready is true when /health succeeded and the binary has not been retired.
func (w *WhisperLive) Ready() bool {
	if w == nil {
		return false
	}
	w.mu.Lock()
	proc := w.proc
	w.mu.Unlock()
	return proc != nil && proc.Ready()
}

// Retired is true after a binary that cannot read a posted WAV.
func (w *WhisperLive) Retired() bool {
	if w == nil {
		return true
	}
	w.mu.Lock()
	proc := w.proc
	w.mu.Unlock()
	return proc == nil || proc.Retired()
}

// Ensure waits for the live server, or restarts it after a crash.
func (w *WhisperLive) Ensure(ctx context.Context) error {
	if w == nil {
		return fmt.Errorf("whisper-server is not running")
	}
	w.mu.Lock()
	proc := w.proc
	w.mu.Unlock()
	if proc == nil {
		return fmt.Errorf("whisper-server is not running")
	}
	if proc.Retired() {
		return fmt.Errorf("whisper-server retired")
	}
	if proc.Ready() {
		return nil
	}
	if err := proc.Wait(ctx); err == nil && proc.Ready() {
		return nil
	}
	if proc.Retired() {
		return fmt.Errorf("whisper-server retired")
	}
	w.mu.Lock()
	if w.proc != proc {
		w.mu.Unlock()
		return w.Ensure(ctx)
	}
	w.mu.Unlock()
	if err := proc.Restart(ctx); err != nil {
		return err
	}
	return proc.Wait(ctx)
}

// Transcribe posts one clip. ErrBadAudio retires this process.
func (w *WhisperLive) Transcribe(ctx context.Context, wav []byte, language string) (string, error) {
	if w == nil {
		return "", fmt.Errorf("whisper-server is not running")
	}
	w.mu.Lock()
	proc := w.proc
	w.mu.Unlock()
	if proc == nil {
		return "", fmt.Errorf("whisper-server is not running")
	}
	return proc.Transcribe(ctx, wav, language)
}

// FinishCLI starts a one-shot CLI process for this clip.
func FinishCLI(ctx context.Context, cfg model.ConfigV2, wav []byte, language string) (string, error) {
	ahead, err := StartWhisperAhead(ctx, cfg, language)
	if err != nil {
		return "", err
	}
	defer ahead.Close()
	return ahead.Finish(ctx, wav)
}

// WarmLive returns a finish func that posts to the live server when ready.
// Otherwise it starts the CLI ahead of the clip. A binary that cannot read a
// posted WAV falls back to the CLI for this clip and later WarmLive calls.
// A crash restarts the server for the next clip.
func WarmLive(ctx context.Context, live *WhisperLive, cfg model.ConfigV2, language string) (func(context.Context, []byte) (string, error), error) {
	if live != nil && !live.Retired() {
		if err := live.Ensure(ctx); err == nil && live.Ready() {
			return func(ctx context.Context, wav []byte) (string, error) {
				text, err := live.Transcribe(ctx, wav, language)
				if err == nil {
					return text, nil
				}
				if errors.Is(err, whispercli.ErrBadAudio) || live.Retired() {
					return FinishCLI(ctx, cfg, wav, language)
				}
				if !live.Ready() {
					if err := live.Ensure(ctx); err == nil && live.Ready() {
						text, err := live.Transcribe(ctx, wav, language)
						if err == nil {
							return text, nil
						}
						if errors.Is(err, whispercli.ErrBadAudio) || live.Retired() {
							return FinishCLI(ctx, cfg, wav, language)
						}
					}
					return FinishCLI(ctx, cfg, wav, language)
				}
				return "", err
			}, nil
		}
	}
	ahead, err := StartWhisperAhead(ctx, cfg, language)
	if err != nil {
		return nil, err
	}
	return func(ctx context.Context, wav []byte) (string, error) {
		defer ahead.Close()
		return ahead.Finish(ctx, wav)
	}, nil
}
