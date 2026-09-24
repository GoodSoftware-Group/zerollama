package server

import (
	"errors"
	"fmt"
	"strings"
)

// ErrUmaGPULease is returned when mlxrunner cannot take HOLD_GPU (broker busy,
// wedged dead-owner lease, or wait timeout). HTTP 503 — not "model not found".
var ErrUmaGPULease = errors.New(
	"gpu lease unavailable (HOLD_GPU): Metal broker is busy or wedged — " +
		"unload other Metal/MLX runners, check /tmp/uma_daemon.sock (STATUS/JOBS/QUEUE), " +
		"RELEASE dead holders, then retry",
)

// ErrMLXRunnerJetsam is returned when the mlxrunner subprocess was SIGKILL'd
// (common under dual large MLX on Apple UMA). HTTP 503 — unload peers and retry.
var ErrMLXRunnerJetsam = errors.New(
	"mlx runner killed during load (likely UMA/jetsam): " +
		"unload other Metal/MLX models (Darwin keeps one MLX resident by default) and retry",
)

// classifyTransientMLXError wraps known transient Metal/MLX load failures so
// clients get 503 + clear text and load cooldown does not fire.
func classifyTransientMLXError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrUmaGPULease) || errors.Is(err, ErrMLXRunnerJetsam) {
		return err
	}
	s := err.Error()
	switch {
	case strings.Contains(s, "HOLD_GPU") || strings.Contains(s, "uma lease begin"):
		return fmt.Errorf("%w: %v", ErrUmaGPULease, err)
	case strings.Contains(s, "signal: killed"):
		return fmt.Errorf("%w: %v", ErrMLXRunnerJetsam, err)
	default:
		return err
	}
}

func mlxScheduleErrorCode(err error) string {
	switch {
	case errors.Is(err, ErrUmaGPULease):
		return "gpu_lease"
	case errors.Is(err, ErrMLXRunnerJetsam):
		return "mlx_jetsam"
	case errors.Is(err, ErrMLXExclusiveBusy):
		return "mlx_exclusive"
	case errors.Is(err, ErrLoadCooldown):
		return "load_cooldown"
	case errors.Is(err, ErrDarwinMetalContention):
		return "metal_contention"
	default:
		return ""
	}
}
