// inference_workload aggregates ggml scheduler load and runtime /health for training
// submit policy. Separate from the VRAM broker: handoff is proactive before loads;
// idle-wait is reactive at submit — the main single-GPU failure is training into a loaded GGUF.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
)

var (
	// ErrInferenceBacklogActive is returned when training submit is blocked by inference load.
	ErrInferenceBacklogActive = errors.New("inference backlog active")
	// ErrRuntimeHealthProbeFailed is returned when idle-wait cannot read runtime /health.
	ErrRuntimeHealthProbeFailed = errors.New("runtime health probe failed")
)

var inferenceHealthClient = &http.Client{Timeout: 2 * time.Second}

const runtimeHealthCacheTTL = 500 * time.Millisecond // Why: training submit idle-wait and ggml load paths probe /health multiple times per second; cache avoids loopback RTT on every check.

var (
	runtimeHealthCacheMu sync.Mutex
	runtimeHealthCached  runtimeHealthSnapshot
	runtimeHealthCacheURL string
	runtimeHealthCachedAt time.Time
)

// InferenceWorkloadStatus summarizes ggml scheduler and Python runtime load.
type InferenceWorkloadStatus struct {
	SchedulerPending   int
	SchedulerActive    int
	SchedulerLoaded    int
	RuntimeWaiting     int
	RuntimeRunning     int
	RuntimeState       string
	RuntimeLlamaLoaded bool
}

func (st InferenceWorkloadStatus) busy() bool {
	if st.SchedulerPending > 0 || st.SchedulerActive > 0 {
		return true
	}
	if envconfig.TrainingWaitGgmlLoaded() && st.SchedulerLoaded > 0 {
		return true
	}
	if st.RuntimeWaiting > 0 || st.RuntimeRunning > 0 {
		return true
	}
	// Loaded llama-server holds VRAM even with an empty runtime queue.
	return st.RuntimeLlamaLoaded
}

type runtimeHealthSnapshot struct {
	waiting     int
	running     int
	state       string
	llamaLoaded bool
	fifoOldest  uint64
	ok          bool // false when runtime URL is set but /health could not be read
}

func (s *Server) schedWorkload(st *InferenceWorkloadStatus) {
	if s == nil || s.sched == nil {
		return
	}
	st.SchedulerPending, st.SchedulerActive, st.SchedulerLoaded = s.sched.InferenceBacklog()
}

func (s *Server) inferenceWorkloadStatus(ctx context.Context) InferenceWorkloadStatus {
	var st InferenceWorkloadStatus
	s.schedWorkload(&st)
	h := runtimeInferenceHealth(ctx)
	if h.ok {
		st.RuntimeWaiting = h.waiting
		st.RuntimeRunning = h.running
		st.RuntimeState = h.state
		st.RuntimeLlamaLoaded = h.llamaLoaded
	}
	return st
}

func (s *Server) checkTrainingSubmitAllowed(ctx context.Context) error {
	if !envconfig.TrainingWaitInferenceIdle() {
		return nil
	}
	var st InferenceWorkloadStatus
	s.schedWorkload(&st)
	h := runtimeInferenceHealth(ctx)
	if !h.ok && runtimeHealthProbeRequired() {
		if envconfig.TrainingWaitFailClosed() {
			return fmt.Errorf(
				"%w: cannot read %s/health; unset ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED to allow submit when probe fails",
				ErrRuntimeHealthProbeFailed,
				strings.TrimSuffix(strings.TrimSpace(effectiveRuntimeURL()), "/"),
			)
		}
	} else if h.ok {
		st.RuntimeWaiting = h.waiting
		st.RuntimeRunning = h.running
		st.RuntimeState = h.state
		st.RuntimeLlamaLoaded = h.llamaLoaded
	}
	if st.busy() {
		return fmt.Errorf(
			"%w (ggml pending=%d active=%d loaded=%d; runtime waiting=%d running=%d llama_server=%v state=%q); "+
				"retry when idle, unload inference models, or unset ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE",
			ErrInferenceBacklogActive,
			st.SchedulerPending,
			st.SchedulerActive,
			st.SchedulerLoaded,
			st.RuntimeWaiting,
			st.RuntimeRunning,
			st.RuntimeLlamaLoaded,
			st.RuntimeState,
		)
	}
	return nil
}

func runtimeHealthProbeRequired() bool {
	return strings.TrimSpace(effectiveRuntimeURL()) != ""
}

func runtimeInferenceHealth(ctx context.Context) runtimeHealthSnapshot {
	base := strings.TrimSpace(effectiveRuntimeURL())
	if base == "" {
		return runtimeHealthSnapshot{ok: true}
	}

	// Short TTL cache — see runtimeHealthCacheTTL. Stale by 500ms is acceptable for
	// training submit gating; fresh enough to detect runtime llama_server unload.
	runtimeHealthCacheMu.Lock()
	if runtimeHealthCacheURL == base && time.Since(runtimeHealthCachedAt) < runtimeHealthCacheTTL {
		snap := runtimeHealthCached
		runtimeHealthCacheMu.Unlock()
		return snap
	}
	runtimeHealthCacheMu.Unlock()

	snap := fetchRuntimeInferenceHealth(ctx, base)

	runtimeHealthCacheMu.Lock()
	runtimeHealthCached = snap
	runtimeHealthCacheURL = base
	runtimeHealthCachedAt = time.Now()
	runtimeHealthCacheMu.Unlock()
	return snap
}

func fetchRuntimeInferenceHealth(ctx context.Context, base string) runtimeHealthSnapshot {
	url := strings.TrimSuffix(base, "/") + "/health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return runtimeHealthSnapshot{}
	}
	resp, err := inferenceHealthClient.Do(req)
	if err != nil {
		return runtimeHealthSnapshot{}
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return runtimeHealthSnapshot{}
	}
	var body struct {
		Waiting            int    `json:"waiting"`
		Running            int    `json:"running"`
		InferenceState     string `json:"inference_state"`
		LlamaServer        bool   `json:"llama_server"`
		FifoRuntimeOldest  uint64 `json:"fifo_runtime_oldest"`
	}
	if json.NewDecoder(resp.Body).Decode(&body) != nil {
		return runtimeHealthSnapshot{}
	}
	return runtimeHealthSnapshot{
		waiting:     body.Waiting,
		running:     body.Running,
		state:       body.InferenceState,
		llamaLoaded: body.LlamaServer,
		fifoOldest:  body.FifoRuntimeOldest,
		ok:          true,
	}
}

// TrainingSubmitConflict reports whether err should map to HTTP 409 for training submit.
func TrainingSubmitConflict(err error) bool {
	return errors.Is(err, ErrInferenceBacklogActive) ||
		errors.Is(err, ErrRuntimeHealthProbeFailed) ||
		errors.Is(err, ErrTrainingOutsideWindow)
}

// TrainingSubmitMisconfigured reports operator/config errors that should not use HTTP 409.
func TrainingSubmitMisconfigured(err error) bool {
	return errors.Is(err, ErrTrainingWindowMisconfigured)
}
