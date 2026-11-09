package qwen3asr

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

// Shared keeps one Qwen3-ASR process for a weight directory across requests.
type Shared struct {
	mu       sync.Mutex
	dir      string
	proc     *Server
	booting  bool
	bootDone chan struct{}
}

// defaultShared is the process used by Transcribe.
var defaultShared = &Shared{}

// Transcribe posts wav to the resident process. language is a whisper-style
// code (en) or empty. ErrUnavailable means the caller should use Whisper.
// A clip the model rejects, a cancelled wait, or a failed restart is an
// ordinary error.
func Transcribe(ctx context.Context, modelDir string, wav []byte, language string) (string, error) {
	return defaultShared.Transcribe(ctx, modelDir, wav, language)
}

// Transcribe starts or reuses the process for modelDir.
func (s *Shared) Transcribe(ctx context.Context, modelDir string, wav []byte, language string) (string, error) {
	if s == nil {
		return "", ErrUnavailable
	}
	modelDir = strings.TrimSpace(modelDir)
	if modelDir == "" {
		return "", ErrUnavailable
	}
	if st, err := os.Stat(modelDir); err != nil || !st.IsDir() {
		return "", ErrUnavailable
	}
	name := LanguageName(language)

	proc, err := s.ensure(ctx, modelDir)
	if err != nil {
		return "", err
	}
	text, err := proc.Transcribe(ctx, wav, name)
	if err == nil {
		return text, nil
	}
	if errors.Is(err, ErrUnavailable) {
		return "", err
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return "", err
	}
	if errors.Is(err, ErrBusy) {
		return s.waitBusy(ctx, proc, wav, name)
	}
	// Process may have crashed after a successful start. Restart once.
	if !proc.Ready() {
		s.mu.Lock()
		same := s.proc == proc && s.dir == modelDir
		s.mu.Unlock()
		if same {
			if rerr := proc.Restart(ctx); rerr != nil {
				s.drop(proc)
				// A restart failure is not a missing install; do not fall back to Whisper.
				// Use %v so ErrUnavailable from a cold wait is not wrapped for fallback.
				return "", fmt.Errorf("qwen3-asr restart failed: %v", rerr)
			}
			return proc.Transcribe(ctx, wav, name)
		}
	}
	return "", err
}

// waitBusy retries while another clip holds the GPU. A guess that was
// cancelled still runs to the end of that call; the committed clip waits
// for it instead of stacking another thread.
func (s *Shared) waitBusy(ctx context.Context, proc *Server, wav []byte, name string) (string, error) {
	deadline := time.Now().Add(3 * time.Second)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return "", ctx.Err()
		case <-timer.C:
		}
		text, err := proc.Transcribe(ctx, wav, name)
		if !errors.Is(err, ErrBusy) {
			return text, err
		}
		if time.Now().After(deadline) {
			return "", err
		}
	}
}

func (s *Shared) ensure(ctx context.Context, modelDir string) (*Server, error) {
	for {
		s.mu.Lock()
		if s.booting {
			done := s.bootDone
			s.mu.Unlock()
			if done == nil {
				continue
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-done:
			}
			continue
		}
		if s.proc != nil && s.dir == modelDir {
			proc := s.proc
			s.mu.Unlock()
			if err := proc.Wait(ctx); err != nil {
				// A killed or crashed child keeps startErr set. Wait would
				// return that forever, so drop it and boot another process.
				if proc.Ready() || proc.Loading() {
					return nil, err
				}
				s.mu.Lock()
				same := s.proc == proc && !s.booting
				if same {
					s.proc = nil
					s.dir = ""
				}
				s.mu.Unlock()
				if same {
					proc.Close()
				}
				continue
			}
			return proc, nil
		}

		old := s.proc
		s.proc = nil
		s.dir = ""
		s.booting = true
		done := make(chan struct{})
		s.bootDone = done
		s.mu.Unlock()

		if old != nil {
			old.Close()
		}

		proc, err := StartServer(ctx, modelDir)
		if err == nil {
			err = proc.Wait(ctx)
		}
		if err != nil {
			if proc != nil {
				// Keep a still-loading process when the caller cancelled so the
				// next request can Wait for the same load.
				if (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) && proc.Loading() {
					s.mu.Lock()
					s.proc = proc
					s.dir = modelDir
					s.booting = false
					close(done)
					s.bootDone = nil
					s.mu.Unlock()
					return nil, err
				}
				proc.Close()
			}
			s.mu.Lock()
			s.booting = false
			close(done)
			s.bootDone = nil
			s.mu.Unlock()
			return nil, err
		}

		s.mu.Lock()
		s.proc = proc
		s.dir = modelDir
		s.booting = false
		close(done)
		s.bootDone = nil
		s.mu.Unlock()
		return proc, nil
	}
}

func (s *Shared) drop(proc *Server) {
	if s == nil || proc == nil {
		return
	}
	s.mu.Lock()
	if s.proc == proc {
		s.proc = nil
		s.dir = ""
	}
	s.mu.Unlock()
	proc.Close()
}

// Close stops the shared process.
func (s *Shared) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	proc := s.proc
	s.proc = nil
	s.dir = ""
	s.mu.Unlock()
	if proc != nil {
		proc.Close()
	}
}
