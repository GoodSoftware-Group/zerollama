package piperphrase

import (
	"context"
	"strings"
	"sync"
)

// Shared is one Piper process for a binary and its flags. The same flags
// reuse the process. A different flag set, or a failed utterance, starts another.
type Shared struct {
	mu   sync.Mutex
	key  string
	proc *Proc
}

// Close stops the process.
func (s *Shared) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	proc := s.proc
	s.proc = nil
	s.key = ""
	s.mu.Unlock()
	if proc != nil {
		proc.Close()
	}
}

// Speak writes one WAV utterance on a process started without --output_raw.
func (s *Shared) Speak(ctx context.Context, bin string, args []string, text, wavPath string) ([]byte, error) {
	proc, err := s.procFor(bin, args)
	if err != nil {
		return nil, err
	}
	data, err := proc.Speak(ctx, text, wavPath)
	if err != nil {
		s.drop(proc)
	}
	return data, err
}

// SpeakPCM emits s16le mono from a process started with --output_raw.
func (s *Shared) SpeakPCM(ctx context.Context, bin string, args []string, text string, onChunk func([]byte) error) error {
	proc, err := s.procFor(bin, args)
	if err != nil {
		return err
	}
	if err := proc.SpeakPCM(ctx, text, onChunk); err != nil {
		s.drop(proc)
		return err
	}
	return nil
}

func (s *Shared) procFor(bin string, args []string) (*Proc, error) {
	key := bin + "\x00" + strings.Join(args, "\x00")
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.proc != nil && s.key == key {
		return s.proc, nil
	}
	old := s.proc
	s.proc = nil
	s.key = ""
	if old != nil {
		old.Close()
	}
	proc, err := Start(bin, args)
	if err != nil {
		return nil, err
	}
	s.proc = proc
	s.key = key
	return proc, nil
}

func (s *Shared) drop(proc *Proc) {
	s.mu.Lock()
	same := s.proc == proc
	if same {
		s.proc = nil
		s.key = ""
	}
	s.mu.Unlock()
	if same && proc != nil {
		proc.Close()
	}
}
