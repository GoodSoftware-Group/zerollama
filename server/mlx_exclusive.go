package server

import (
	"errors"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/envconfig"
)

// ErrMLXExclusiveBusy is returned when another MLX runner is fulfillment-protected
// and cannot be evicted for exclusive residency.
var ErrMLXExclusiveBusy = errors.New(
	"another MLX model is pinned or in an exclusive session; " +
		"finish that session or unload it before loading a second safetensors model",
)

// findOtherMLXRunner returns another loaded safetensors/MLX runner that should
// yield before pending loads. Darwin default is one MLX resident at a time
// (ZEROLLAMA_MLX_EXCLUSIVE) so dual large models do not jetsam mid-turn.
// If the only peer is fulfillment-protected, returns (nil, ErrMLXExclusiveBusy).
func (s *Scheduler) findOtherMLXRunner(pending *LlmRequest) (*runnerRef, error) {
	if pending == nil || pending.model == nil || !pending.model.IsMLX() {
		return nil, nil
	}
	if !envconfig.MLXExclusive() {
		return nil, nil
	}
	pendingKey := schedulerModelKey(pending.model)
	protected := s.mlxGate.protectedModelKeys()

	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	var protectedPeer *runnerRef
	for key, r := range s.loaded {
		if r == nil || key == pendingKey {
			continue
		}
		if r.model == nil || !r.model.IsMLX() {
			continue
		}
		if _, skip := protected[r.modelKey]; skip {
			protectedPeer = r
			continue
		}
		slog.Info("mlx exclusive: will unload other MLX before load",
			"pending", pending.model.ShortName,
			"evict", r.model.ShortName,
			"evict_key", r.modelKey,
		)
		return r, nil
	}
	if protectedPeer != nil {
		name := protectedPeer.modelKey
		if protectedPeer.model != nil && protectedPeer.model.ShortName != "" {
			name = protectedPeer.model.ShortName
		}
		return nil, fmt.Errorf("%w (held: %s)", ErrMLXExclusiveBusy, name)
	}
	return nil, nil
}
