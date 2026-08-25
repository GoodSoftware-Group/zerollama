package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

// ErrLoadCooldown is returned while a model is in failed-load cooldown (LA18).
var ErrLoadCooldown = errors.New("model load in cooldown")

// LoadCooldownError carries retry timing for 503 Retry-After.
type LoadCooldownError struct {
	Key      string
	Until    time.Time
	Failures int
	Cause    error
}

func (e *LoadCooldownError) Error() string {
	wait := time.Until(e.Until).Truncate(time.Millisecond)
	if wait < 0 {
		wait = 0
	}
	if e.Cause != nil {
		return fmt.Sprintf("%v: retry after %s (%d consecutive failures): %v", ErrLoadCooldown, wait, e.Failures, e.Cause)
	}
	return fmt.Sprintf("%v: retry after %s (%d consecutive failures)", ErrLoadCooldown, wait, e.Failures)
}

func (e *LoadCooldownError) Unwrap() error { return e.Cause }

func (e *LoadCooldownError) Is(target error) bool {
	return target == ErrLoadCooldown
}

func (e *LoadCooldownError) RetryAfter() time.Duration {
	d := time.Until(e.Until)
	if d < time.Second {
		return time.Second
	}
	return d
}

type loadCooldownEntry struct {
	until    time.Time
	failures int
	cause    error
}

func (s *Scheduler) cooldownErr(key string) error {
	if key == "" || envconfig.LoadCooldownInitial() <= 0 {
		return nil
	}
	s.loadCooldownMu.Lock()
	defer s.loadCooldownMu.Unlock()
	ent := s.loadCooldown[key]
	if ent == nil {
		return nil
	}
	now := time.Now()
	if now.After(ent.until) {
		return nil
	}
	return &LoadCooldownError{Key: key, Until: ent.until, Failures: ent.failures, Cause: ent.cause}
}

func (s *Scheduler) clearLoadCooldown(key string) {
	if key == "" {
		return
	}
	s.loadCooldownMu.Lock()
	delete(s.loadCooldown, key)
	s.loadCooldownMu.Unlock()
}

func (s *Scheduler) noteLoadFailure(key string, err error) {
	if key == "" || err == nil || skipLoadCooldown(err) {
		return
	}
	initial := envconfig.LoadCooldownInitial()
	if initial <= 0 {
		return
	}
	capDur := envconfig.LoadCooldownMax()
	s.loadCooldownMu.Lock()
	defer s.loadCooldownMu.Unlock()
	if s.loadCooldown == nil {
		s.loadCooldown = make(map[string]*loadCooldownEntry)
	}
	ent := s.loadCooldown[key]
	if ent == nil {
		ent = &loadCooldownEntry{}
		s.loadCooldown[key] = ent
	}
	ent.failures++
	ent.cause = err
	mult := 1 << min(ent.failures-1, 30)
	delay := time.Duration(int64(initial) * int64(mult))
	if delay > capDur || delay <= 0 {
		delay = capDur
	}
	ent.until = time.Now().Add(delay)
	slog.Warn("model load cooldown",
		"key", key,
		"failures", ent.failures,
		"retry_after", delay,
		"error", err,
	)
}

func skipLoadCooldown(err error) bool {
	if err == nil {
		return true
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	if errors.Is(err, ErrLoadCooldown) || errors.Is(err, ErrMaxQueue) {
		return true
	}
	if errors.Is(err, ErrDarwinMetalContention) || errors.Is(err, ErrRuntimeInferenceModel) {
		return true
	}
	if errors.Is(err, ErrEdgeGgmlRunnerDisabled) || errors.Is(err, llm.ErrGgmlRunnerUnlinked) {
		return true
	}
	return false
}

func retryAfterSeconds(d time.Duration) int {
	sec := int(math.Ceil(d.Seconds()))
	if sec < 1 {
		return 1
	}
	return sec
}
