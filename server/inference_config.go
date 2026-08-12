package server

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

// gpuCountCacheTTL: fleet pollers hit /api/status often; full GPU discovery is expensive.
const gpuCountCacheTTL = 30 * time.Second

var (
	gpuCountCacheMu    sync.Mutex
	gpuCountCached     int
	gpuCountCachedAt   time.Time
	gpuCountCacheValid bool
)

// cachedGPUCount returns discover.GPUDevices length with a short TTL.
// Why: effective MAX_LOADED when env is 0 needs GPU count; re-enumerating every poll hurt Mac/CUDA.
func cachedGPUCount(ctx context.Context) int {
	gpuCountCacheMu.Lock()
	defer gpuCountCacheMu.Unlock()
	if gpuCountCacheValid && time.Since(gpuCountCachedAt) < gpuCountCacheTTL {
		return gpuCountCached
	}
	gpus := discover.GPUDevices(ctx, nil)
	n := len(gpus)
	if n < 1 {
		n = 1
	}
	gpuCountCached = n
	gpuCountCachedAt = time.Now()
	gpuCountCacheValid = true
	return n
}

// effectiveMaxLoadedModels mirrors sched.go: when OLLAMA_MAX_LOADED_MODELS is 0,
// default is defaultModelsPerGPU × GPU count (at least 1 GPU worth).
// Why expose both configured and effective on status: clients guessed "0 means unlimited".
func effectiveMaxLoadedModels(ctx context.Context) uint {
	configured := envconfig.MaxRunners()
	if configured > 0 {
		return configured
	}
	return uint(defaultModelsPerGPU * cachedGPUCount(ctx))
}

func formatDurationConfig(d time.Duration) string {
	if d >= time.Duration(1<<62) {
		return "-1"
	}
	return d.String()
}

func residencyOwnerForStatus(runtimeEnabled bool) string {
	if !runtimeEnabled {
		return "go_sched"
	}
	// Runtime may own GGUF chat while ggml still loads vision/legacy — mixed when
	// a runtime URL/embed is configured.
	return "mixed"
}

// inferenceConfigStatus builds inference.config for GET /api/status.
// Why: env folklore (SSH + printenv) does not scale for Orient/Decide progressive probes.
func (s *Server) inferenceConfigStatus(ctx context.Context, runtime api.RuntimeStatus) api.InferenceConfigStatus {
	cfg := api.InferenceConfigStatus{
		NumParallel:           envconfig.NumParallel(),
		NumParallelAuto:       envconfig.GgmlAutoParallelEnabled() && !envconfig.NumParallelExplicit(),
		MaxLoadedConfigured:   envconfig.MaxRunners(),
		MaxLoadedModels:       effectiveMaxLoadedModels(ctx),
		MaxQueue:              envconfig.MaxQueue(),
		KeepAlive:             formatDurationConfig(envconfig.KeepAlive()),
		LoadTimeout:           formatDurationConfig(envconfig.LoadTimeout()),
		SameModelMultiCopy:    false, // advertised honestly — no silent multi-copy of same tag
		ResidencyOwner:        residencyOwnerForStatus(runtime.Enabled),
		NumParallelMeansSlots: true, // NUM_PARALLEL ≠ max concurrent models
	}
	if runtime.Enabled {
		mq := runtimeMaxQueueFromEnv()
		cfg.RuntimeMaxQueue = &mq
	}
	req, resolved, applied := envconfig.InferenceProfileStatus()
	cfg.InferenceProfile = req
	if cfg.InferenceProfile == "" {
		cfg.InferenceProfile = "(default)"
	}
	cfg.InferenceProfileResolved = resolved
	cfg.InferenceProfileApplied = applied
	cfg.GpuProfileID = llm.LastGpuProfileID()
	return cfg
}

func runtimeMaxQueueFromEnv() uint {
	const defaultRuntimeMaxQueue = 512
	raw := envconfig.Var("ZEROLLAMA_RUNTIME_MAX_QUEUE")
	if raw == "" {
		return defaultRuntimeMaxQueue
	}
	n, err := strconv.ParseUint(raw, 10, 32)
	if err != nil || n == 0 {
		return defaultRuntimeMaxQueue
	}
	return uint(n)
}
