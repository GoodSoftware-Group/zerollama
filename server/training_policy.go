package server

import (
	"context"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/runtimeclient"
	"github.com/ollama/ollama/server/vram"
)

const trainingGPUPolicyInterval = 400 * time.Millisecond

// runTrainingGPUPolicyMonitor keeps ggml loads paused while training holds the GPU.
func (s *Server) runTrainingGPUPolicyMonitor(ctx context.Context) {
	if s == nil || s.training == nil {
		return
	}
	s.enforceTrainingGPUPolicy(ctx)

	ticker := time.NewTicker(trainingGPUPolicyInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			shutdown := context.Background()
			s.syncTrainingVRAMCoordination(shutdown, false)
			s.finalizeInferenceCoordination(shutdown)
			return
		case <-ticker.C:
			s.enforceTrainingGPUPolicy(ctx)
		}
	}
}

func (s *Server) enforceTrainingGPUPolicy(ctx context.Context) {
	if !envconfig.BlockInferenceDuringTraining() {
		s.syncTrainingVRAMCoordination(ctx, false)
	} else {
		busy := s.trainingOccupiesGPU(ctx)
		s.syncTrainingVRAMCoordination(ctx, busy)
	}
	s.syncRuntimeCoordinationAndPauses(ctx)
}

// syncTrainingVRAMCoordination evicts runtime + ggml inference when training holds the GPU.
func (s *Server) syncTrainingVRAMCoordination(ctx context.Context, block bool) {
	if s == nil {
		return
	}
	s.trainingVRAMMu.Lock()
	defer s.trainingVRAMMu.Unlock()
	if block == s.trainingVRAMBlocked {
		return
	}
	s.trainingVRAMBlocked = block
	runtimeclient.SetTrainingGPUBusy(ctx, block)
	if block {
		vram.ReleaseRuntimeVRAM(ctx)
		if s.sched != nil {
			s.sched.UnloadAllRunners()
		}
		return
	}
	runtimeclient.ResumeInference(ctx)
}

// trainingOccupiesGPU reports whether training currently blocks inference.
func (s *Server) trainingOccupiesGPU(ctx context.Context) bool {
	if s == nil || s.training == nil {
		return false
	}
	_, busy := s.training.OccupiesGPU(ctx)
	if !s.training.LastGPUHealthOK() && envconfig.BlockInferenceFailClosed() {
		return true
	}
	return busy
}

// trainingBlocksInference reports whether new runtime inference should be rejected.
func (s *Server) trainingBlocksInference(ctx context.Context) bool {
	if !envconfig.BlockInferenceDuringTraining() {
		return false
	}
	return s.trainingOccupiesGPU(ctx)
}

func (s *Server) abortIfTrainingBusy(c *gin.Context) bool {
	if !s.trainingBlocksInference(c.Request.Context()) {
		return false
	}
	ctx := c.Request.Context()
	s.syncTrainingVRAMCoordination(ctx, true)
	s.syncRuntimeCoordinationAndPauses(ctx)
	c.AbortWithStatusJSON(503, gin.H{
		"error": "inference paused while training is using the GPU or a training model remains loaded; retry after training finishes or the model is unloaded",
	})
	return true
}
