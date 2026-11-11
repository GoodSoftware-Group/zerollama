package piperphrase

import (
	"context"
	"strings"
	"sync"
)

// Shared keeps one Piper process per binary+flag set. Concurrent requests with
// different voices/flags each get their own process; switching flags no longer
// kills a process that another request is still reading.
type Shared struct {
	mu    sync.Mutex
	slots map[string]*leasedProc
}

type leasedProc struct {
	proc *Proc
	refs int
	dead bool
}

// Close stops every process.
func (s *Shared) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	slots := s.slots
	s.slots = nil
	s.mu.Unlock()
	for _, sl := range slots {
		if sl != nil && sl.proc != nil {
			sl.proc.Close()
		}
	}
}

// Speak writes one WAV utterance on a process started without --output_raw.
func (s *Shared) Speak(ctx context.Context, bin string, args []string, text, wavPath string) ([]byte, error) {
	key := slotKey(bin, args)
	proc, err := s.acquire(key, bin, args)
	if err != nil {
		return nil, err
	}
	defer s.release(key)
	data, err := proc.Speak(ctx, text, wavPath)
	if err != nil {
		s.retire(key, proc)
	}
	return data, err
}

// SpeakPCM emits s16le mono from a process started with --output_raw.
func (s *Shared) SpeakPCM(ctx context.Context, bin string, args []string, text string, onChunk func([]byte) error) error {
	key := slotKey(bin, args)
	proc, err := s.acquire(key, bin, args)
	if err != nil {
		return err
	}
	defer s.release(key)
	if err := proc.SpeakPCM(ctx, text, onChunk); err != nil {
		s.retire(key, proc)
		return err
	}
	return nil
}

func slotKey(bin string, args []string) string {
	return bin + "\x00" + strings.Join(args, "\x00")
}

func (s *Shared) acquire(key, bin string, args []string) (*Proc, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.slots == nil {
		s.slots = make(map[string]*leasedProc)
	}
	if sl, ok := s.slots[key]; ok && sl != nil && sl.proc != nil && !sl.dead {
		sl.refs++
		return sl.proc, nil
	}
	proc, err := Start(bin, args)
	if err != nil {
		return nil, err
	}
	s.slots[key] = &leasedProc{proc: proc, refs: 1}
	return proc, nil
}

func (s *Shared) release(key string) {
	s.mu.Lock()
	sl := s.slots[key]
	if sl == nil {
		s.mu.Unlock()
		return
	}
	if sl.refs > 0 {
		sl.refs--
	}
	closeNow := sl.dead && sl.refs == 0
	if closeNow {
		delete(s.slots, key)
	}
	proc := sl.proc
	s.mu.Unlock()
	if closeNow && proc != nil {
		proc.Close()
	}
}

// retire marks a failed process so new acquires start a fresh one. The process
// is closed once the last reader releases.
func (s *Shared) retire(key string, proc *Proc) {
	s.mu.Lock()
	sl := s.slots[key]
	if sl == nil || sl.proc != proc {
		s.mu.Unlock()
		return
	}
	sl.dead = true
	closeNow := sl.refs == 0
	if closeNow {
		delete(s.slots, key)
	}
	s.mu.Unlock()
	if closeNow {
		proc.Close()
	}
}
