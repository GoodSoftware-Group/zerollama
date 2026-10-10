package mlx

import (
	"testing"
	"time"
)

// Regression: Sweep used to call free() while holding arraysMu; free() also
// locks arraysMu, so the first unpinned free deadlocked the MLX load path
// (hang after MoE LoadWeights, never reaching "mlx load eval starting").
func TestSweepDoesNotDeadlockOnUnpinned(t *testing.T) {
	withMLXThread(t, func() {
		keep := FromValue(1)
		drop := FromValue(2)
		Pin(keep)
		defer Unpin(keep)

		done := make(chan struct{})
		go func() {
			Sweep()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("Sweep deadlocked holding arraysMu (free re-lock)")
		}
		if !keep.Valid() {
			t.Fatal("pinned array was swept")
		}
		if drop.Valid() {
			t.Fatal("unpinned array survived Sweep")
		}
		_ = drop
	})
}
