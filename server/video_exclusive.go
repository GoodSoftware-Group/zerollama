package server

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/server/vram"
)

const (
	videoExclusiveModelKey     = gpuMediaModelKey
	videoExclusivePollInterval = 500 * time.Millisecond
	// Cap how long we keep exclusive if status polling fails (Wan timeouts are ≤1h typically).
	videoExclusiveWatchLimit = 3 * time.Hour
)

// videoExclusiveRequested is true unless the client opts out with fulfillment none/off.
//
// WHY default on: Wan TI2V needs nearly the whole 16GB card; background chat reloading
// llama-server mid-job is what made nvidia-smi look “idle at 260MiB” while DiT OOMed.
func videoExclusiveRequested(opts map[string]any) bool {
	z := zerollamaBlockFromOptions(opts)
	name := strings.ToLower(strings.TrimSpace(stringFromMap(z, "fulfillment")))
	if name == "" {
		return true
	}
	if name == "none" || name == "off" {
		return false
	}
	return true
}

// acquireVideoExclusiveGPU holds fulfillment=exclusive + training VRAM block until the
// run_script job reaches a terminal status. Safe to call for already-tracked job IDs.
func (s *Server) acquireVideoExclusiveGPU(ctx context.Context, jobID string) {
	jobID = strings.TrimSpace(jobID)
	if s == nil || jobID == "" || strings.HasPrefix(jobID, "defer-") {
		return
	}

	s.videoExclusiveMu.Lock()
	if s.videoExclusiveJobs == nil {
		s.videoExclusiveJobs = make(map[string]struct{})
	}
	if _, exists := s.videoExclusiveJobs[jobID]; exists {
		s.videoExclusiveMu.Unlock()
		return
	}
	first := len(s.videoExclusiveJobs) == 0
	s.videoExclusiveJobs[jobID] = struct{}{}
	s.videoExclusiveMu.Unlock()

	if first {
		s.beginVideoExclusiveHold(ctx, jobID)
	}
	go s.watchVideoExclusiveJob(jobID)
}

func (s *Server) beginVideoExclusiveHold(ctx context.Context, jobID string) {
	if s == nil {
		return
	}
	sessionKey := "video:exclusive:" + jobID
	var releaseFulfill func()
	if s.sched != nil {
		// Wait briefly for any other fulfillment; then claim exclusive.
		waitCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		_ = s.sched.mlxGate.waitForFulfillment(waitCtx, videoExclusiveModelKey, sessionKey, fulfillmentBenchmark)
		cancel()
		releaseFulfill = s.sched.mlxGate.beginFulfillment(videoExclusiveModelKey, sessionKey, fulfillmentBenchmark)
		// Force-evict chat runners (including pins) — soft unload leaves gemma resident.
		vram.PrepareForTraining(ctx, s.sched)
	} else {
		releaseFulfill = func() {}
		vram.PrepareForTraining(ctx, nil)
	}

	s.videoExclusiveMu.Lock()
	s.videoExclusiveRelease = releaseFulfill
	s.videoExclusiveMu.Unlock()

	// WHY force block even when OccupiesGPU already true: keep SetTrainingGPUBusy /
	// PauseNewLoads latched across health TTL gaps while Wan's child still holds CUDA.
	s.syncTrainingVRAMCoordination(ctx, true)
	s.updateSchedInferencePauses(ctx)

	slog.Info("video exclusive GPU hold begin",
		"job_id", jobID,
		"session_key", sessionKey,
	)
}

func (s *Server) releaseVideoExclusiveGPU(jobID string) {
	jobID = strings.TrimSpace(jobID)
	if s == nil || jobID == "" {
		return
	}

	s.videoExclusiveMu.Lock()
	if _, ok := s.videoExclusiveJobs[jobID]; !ok {
		s.videoExclusiveMu.Unlock()
		return
	}
	delete(s.videoExclusiveJobs, jobID)
	last := len(s.videoExclusiveJobs) == 0
	releaseFulfill := s.videoExclusiveRelease
	if last {
		s.videoExclusiveRelease = nil
	}
	s.videoExclusiveMu.Unlock()

	if !last {
		return
	}
	if releaseFulfill != nil {
		releaseFulfill()
	}
	ctx := context.Background()
	// Recompute from training health — do not blindly resume if a train job remains.
	busy := false
	if s.training != nil {
		_, busy = s.training.OccupiesGPU(ctx)
	}
	s.syncTrainingVRAMCoordination(ctx, busy)
	s.updateSchedInferencePauses(ctx)
	slog.Info("video exclusive GPU hold end", "job_id", jobID, "training_still_busy", busy)
}

func (s *Server) videoExclusiveActive() bool {
	if s == nil {
		return false
	}
	s.videoExclusiveMu.Lock()
	defer s.videoExclusiveMu.Unlock()
	return len(s.videoExclusiveJobs) > 0
}

func (s *Server) watchVideoExclusiveJob(jobID string) {
	defer s.releaseVideoExclusiveGPU(jobID)
	deadline := time.Now().Add(videoExclusiveWatchLimit)
	ticker := time.NewTicker(videoExclusivePollInterval)
	defer ticker.Stop()
	for {
		if time.Now().After(deadline) {
			slog.Warn("video exclusive hold timed out", "job_id", jobID)
			return
		}
		s.videoExclusiveMu.Lock()
		_, tracked := s.videoExclusiveJobs[jobID]
		s.videoExclusiveMu.Unlock()
		if !tracked {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		raw, err := s.videoJobStatusJSON(ctx, jobID)
		cancel()
		if err == nil && videoJobStatusTerminal(raw) {
			return
		}
		<-ticker.C
	}
}

func videoJobStatusTerminal(raw []byte) bool {
	var wrap struct {
		Job struct {
			Status string `json:"status"`
		} `json:"job"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(wrap.Job.Status)) {
	case "completed", "failed", "cancelled", "canceled", "error":
		return true
	default:
		return false
	}
}
