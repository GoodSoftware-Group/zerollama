package envconfig

import (
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync"
)

// Inference profile collapses the L1/L3/FORK/graphs flag soup into one workload lane.
//
// WHY: operators should not memorize ZEROLLAMA_GPU_PROFILE + LLAMA_CACHE + RADIX +
// FORK + GGML_CUDA_USE_GRAPHS. Resources + calibrated GPU JSON already answer "what
// fits" and "what's fast on this card"; this layer only picks the product lane.
//
// Values:
//
//	auto       — default for production GPU serve: throughput lane from resources
//	throughput — L1 on, FORK off, L3 cache on, Radix off, CUDA graphs off (Linux)
//	agent      — throughput + L3_PROFILE=agent (Radix / kv_unified) when RUNTIME_CONFIG unset
//	vram       — throughput + FORK_AUTO_VRAM (TBQ only when ctx ≥ threshold)
//	off        — do not apply soft defaults from this layer
const (
	InferenceProfileAuto       = "auto"
	InferenceProfileThroughput = "throughput"
	InferenceProfileAgent      = "agent"
	InferenceProfileVRAM       = "vram"
	InferenceProfileOff        = "off"
)

var (
	inferenceProfileMu       sync.Mutex
	inferenceProfileResolved string
	inferenceProfileApplied  []string
)

// InferenceProfileRequested returns the raw env (empty = treat as auto at apply time
// only when callers opt in; ApplyInferenceProfileDefaults uses defaultAuto).
func InferenceProfileRequested() string {
	return strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_INFERENCE_PROFILE")))
}

// InferenceProfileStatus returns what ApplyInferenceProfileDefaults last resolved.
func InferenceProfileStatus() (requested, resolved string, applied []string) {
	inferenceProfileMu.Lock()
	defer inferenceProfileMu.Unlock()
	req := InferenceProfileRequested()
	appliedCopy := append([]string(nil), inferenceProfileApplied...)
	return req, inferenceProfileResolved, appliedCopy
}

// ApplyInferenceProfileDefaults sets process env for the chosen lane when the
// operator has not already set each variable. Safe to call multiple times.
//
// defaultAuto: when ZEROLLAMA_INFERENCE_PROFILE is unset, use "auto" (production
// GPU path) vs leave untouched (tests / Mac ggml-only). Serve wiring passes true.
func ApplyInferenceProfileDefaults(defaultAuto bool) {
	raw := InferenceProfileRequested()
	resolved := raw
	if resolved == "" {
		if !defaultAuto {
			return
		}
		resolved = InferenceProfileAuto
	}
	switch resolved {
	case InferenceProfileOff, "0", "false", "no":
		inferenceProfileMu.Lock()
		inferenceProfileResolved = InferenceProfileOff
		inferenceProfileApplied = nil
		inferenceProfileMu.Unlock()
		return
	case InferenceProfileAuto:
		resolved = resolveAutoInferenceProfile()
	case InferenceProfileThroughput, InferenceProfileAgent, InferenceProfileVRAM:
		// keep
	default:
		slog.Warn("unknown ZEROLLAMA_INFERENCE_PROFILE; treating as auto", "value", raw)
		resolved = resolveAutoInferenceProfile()
	}

	applied := applyInferenceLane(resolved)
	inferenceProfileMu.Lock()
	inferenceProfileResolved = resolved
	inferenceProfileApplied = applied
	inferenceProfileMu.Unlock()

	if len(applied) > 0 {
		slog.Info("inference profile applied",
			"requested", nullIfEmpty(raw),
			"resolved", resolved,
			"applied", strings.Join(applied, ","),
		)
	}
}

func nullIfEmpty(s string) string {
	if s == "" {
		return "(default)"
	}
	return s
}

// resolveAutoInferenceProfile picks a lane from resources without requiring flags.
// Single NVIDIA GPU → throughput (L1 calibrated). Multi-GPU stays throughput too;
// agent is opt-in (shared system prompts). Darwin → throughput soft defaults only
// (Metal path still owns ggml; L1 JSON applies when Phase 17/runtime used).
func resolveAutoInferenceProfile() string {
	// Future: detect agent fleets via OLLAMA_NUM_PARALLEL / embed hints.
	// Today auto ≡ throughput: calibrated L1 + safe CUDA defaults, not Radix.
	_ = runtime.GOOS
	return InferenceProfileThroughput
}

func applyInferenceLane(lane string) []string {
	var applied []string
	set := func(key, val string) {
		if strings.TrimSpace(os.Getenv(key)) != "" {
			return
		}
		_ = os.Setenv(key, val)
		applied = append(applied, key+"="+val)
	}

	// Shared base: L1 throughput path (stock KV, no fork-for-speed).
	set("ZEROLLAMA_GPU_PROFILE", "1")
	set("ZEROLLAMA_LLAMA_FORK", "0")
	set("ZEROLLAMA_LLAMA_CACHE", "1")
	if runtime.GOOS == "linux" {
		// sm_120 graphs still stale after slot clear — keep off until validated.
		set("GGML_CUDA_USE_GRAPHS", "0")
	}

	switch lane {
	case InferenceProfileAgent:
		// Prefer one YAML over six RADIX_* exports (see docs/runtime-env.md).
		set("ZEROLLAMA_L3_PROFILE", "agent")
		set("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
	case InferenceProfileVRAM:
		// FORK stays 0 for tok/s; AUTO_VRAM enables TBQ only at long ctx.
		set("ZEROLLAMA_LLAMA_FORK_AUTO_VRAM", "1")
		set("ZEROLLAMA_LLAMA_FORK_PROFILE", "vram")
	}

	return applied
}
