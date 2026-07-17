package server

import (
	"context"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/runtimeclient"
)

// coordinationWorkloadFields returns ggml scheduler + runtime queue counts for the Python mirror.
func (s *Server) coordinationWorkloadFields(ctx context.Context, h runtimeHealthSnapshot) map[string]any {
	if s == nil {
		return nil
	}
	var st InferenceWorkloadStatus
	s.schedWorkload(&st)
	out := map[string]any{
		"sched_pending": st.SchedulerPending,
		"sched_active":  st.SchedulerActive,
		"sched_loaded":  st.SchedulerLoaded,
	}
	if h.ok {
		out["runtime_waiting"] = h.waiting
		out["runtime_running"] = h.running
		out["runtime_llama_loaded"] = h.llamaLoaded
	}
	return out
}

// pushRuntimeCoordination mirrors Go training/defer policy and inference backlog to Python /health.
func (s *Server) pushRuntimeCoordination(ctx context.Context) {
	s.pushRuntimeCoordinationFromHealth(ctx, runtimeInferenceHealth(ctx))
}

func (s *Server) pushRuntimeCoordinationFromHealth(ctx context.Context, h runtimeHealthSnapshot) {
	if s == nil {
		return
	}
	s.setRuntimeFifoOldest(h.fifoOldest)
	s.trainingVRAMMu.Lock()
	blocked := s.trainingVRAMBlocked
	s.trainingVRAMMu.Unlock()
	snap := map[string]any{
		"block_inference_during_training": envconfig.BlockInferenceDuringTraining(),
		"training_gpu_blocked":            blocked,
		"ggml_loads_paused":               s.inferencePausesGgml(ctx, h),
	}
	if s.trainingDefer != nil {
		for k, v := range s.trainingDefer.coordinationStats() {
			snap[k] = v
		}
	}
	for k, v := range s.coordinationWorkloadFields(ctx, h) {
		snap[k] = v
	}
	for k, v := range s.fifoCoordinationFields() {
		snap[k] = v
	}
	if peers := lmcacheBlobPeersForCoordination(); len(peers) > 0 {
		snap["lmcache_blob_peers"] = peers
	}
	runtimeclient.PushGoCoordination(ctx, snap)
}

// syncRuntimeCoordinationAndPauses probes runtime /health once per tick for mirror + ggml pause.
func (s *Server) syncRuntimeCoordinationAndPauses(ctx context.Context) {
	h := runtimeInferenceHealth(ctx)
	s.updateSchedInferencePausesFromHealth(ctx, h)
	s.pushRuntimeCoordinationFromHealth(ctx, h)
}

// finalizeInferenceCoordination resumes ggml loads and pushes a cleared mirror on daemon shutdown
// so Python admission does not treat ggml_loads_paused as true until GO_COORDINATION_TTL_S.
func (s *Server) finalizeInferenceCoordination(ctx context.Context) {
	if s == nil {
		return
	}
	if s.sched != nil {
		s.sched.ResumeLoads()
	}
	h := runtimeInferenceHealth(ctx)
	s.pushRuntimeCoordinationFinal(ctx, h)
}

func (s *Server) pushRuntimeCoordinationFinal(ctx context.Context, h runtimeHealthSnapshot) {
	if s == nil {
		return
	}
	snap := map[string]any{
		"block_inference_during_training": envconfig.BlockInferenceDuringTraining(),
		"training_gpu_blocked":            false,
		"ggml_loads_paused":               false,
	}
	if s.trainingDefer != nil {
		for k, v := range s.trainingDefer.coordinationStats() {
			snap[k] = v
		}
	}
	for k, v := range s.coordinationWorkloadFields(ctx, h) {
		snap[k] = v
	}
	for k, v := range s.fifoCoordinationFields() {
		snap[k] = v
	}
	if peers := lmcacheBlobPeersForCoordination(); len(peers) > 0 {
		snap["lmcache_blob_peers"] = peers
	}
	runtimeclient.PushGoCoordination(ctx, snap)
}

const runtimeCoordinationInterval = 400 * time.Millisecond

// runRuntimeCoordinationPusher refreshes the Python mirror when the training GPU
// monitor is not running (runtime-only or OLLAMA_BLOCK_INFERENCE_DURING_TRAINING=off).
func (s *Server) runRuntimeCoordinationPusher(ctx context.Context) {
	if s == nil {
		return
	}
	s.syncRuntimeCoordinationAndPauses(ctx)
	ticker := time.NewTicker(runtimeCoordinationInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			s.finalizeInferenceCoordination(context.Background())
			return
		case <-ticker.C:
			s.syncRuntimeCoordinationAndPauses(ctx)
		}
	}
}
