// Package llm — Flash-MoE helpers for anemll-flash-llama.cpp llama-server.
//
// Why separate from ggml Metal: Flash-MoE slot-bank streaming and -fit VRAM budgeting
// live in anemll's forked llama-server (--moe-* flags). Mac ggml stays the default
// for in-RAM models (~+7% decode vs upstream llama-server on M4 Max).
//
// Activation requires ZEROLLAMA_FLASH_MOE=1 or a configured sidecar path (env or
// manifest moe_sidecar). Prefetch and slot-bank knobs alone do not enable Flash-MoE.
package llm

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// flashMoEConfig holds resolved Flash-MoE llama-server flags (anemll-flash-llama.cpp).
type flashMoEConfig struct {
	Active           bool
	Sidecar          string
	Mode             string
	SlotBank         int
	TopK             int
	PrefetchTemporal bool
}

// resolveFlashMoEConfig merges env defaults with per-model manifest overrides.
// Active only when explicitly enabled or a sidecar path is set — why: avoid forcing
// -ub 1 / -fit on for tuning-only manifest fields like moe_prefetch_temporal: false.
func resolveFlashMoEConfig(opts api.Options) flashMoEConfig {
	cfg := flashMoEConfig{
		Active:           envconfig.FlashMoEEnabled(),
		Sidecar:          envconfig.FlashMoESidecar(),
		Mode:             envconfig.FlashMoEMode(),
		SlotBank:         envconfig.FlashMoESlotBank(),
		TopK:             envconfig.FlashMoETopK(),
		PrefetchTemporal: envconfig.FlashMoEPrefetchTemporal(),
	}

	if v := strings.TrimSpace(opts.MoeSidecar); v != "" {
		cfg.Sidecar = v
	}
	if v := strings.TrimSpace(opts.MoeMode); v != "" {
		cfg.Mode = v
	}
	if opts.MoeSlotBank > 0 {
		cfg.SlotBank = opts.MoeSlotBank
	}
	if opts.MoeTopK > 0 {
		cfg.TopK = opts.MoeTopK
	}
	if opts.MoePrefetchTemporal != nil {
		cfg.PrefetchTemporal = *opts.MoePrefetchTemporal
	}

	cfg.Active = envconfig.FlashMoEEnabled() || cfg.Sidecar != ""

	if cfg.Active && cfg.Mode == "" {
		cfg.Mode = "slot-bank"
	}
	return cfg
}

// appendFlashMoEArgs adds anemll --moe-* flags when a sidecar is configured.
// Also applies -fit on and -ub 1 — why: required by anemll for dense/slot VRAM
// balance and correct MoE GPU prefill (see docs/flash-moe.md).
func appendFlashMoEArgs(params []string, opts api.Options) []string {
	cfg := resolveFlashMoEConfig(opts)
	if !cfg.Active || cfg.Sidecar == "" {
		return params
	}

	params = append(params,
		"--moe-mode", cfg.Mode,
		"--moe-sidecar", cfg.Sidecar,
	)
	if cfg.SlotBank > 0 {
		params = append(params, "--moe-slot-bank", strconv.Itoa(cfg.SlotBank))
	}
	if cfg.TopK > 0 {
		params = append(params, "--moe-topk", strconv.Itoa(cfg.TopK))
	}
	if cfg.PrefetchTemporal {
		params = append(params, "--moe-prefetch-temporal")
	}

	params = append(params, "-fit", "on")
	params = setLlamaServerUbatch(params, 1)

	return params
}

// setLlamaServerUbatch rewrites an existing -ub value or appends one.
// Why in-place rewrite: appendBatchArgs may have set -ub earlier in startLlamaServer.
func setLlamaServerUbatch(params []string, ubatch int) []string {
	val := strconv.Itoa(ubatch)
	for i := 0; i < len(params)-1; i++ {
		if params[i] == "-ub" {
			params[i+1] = val
			return params
		}
	}
	return append(params, "-ub", val)
}

// preferFlashMoELlamaServer is true when the operator opted into Flash-MoE env or
// pinned a fork binary path — why: select anemll llama-server without sidecar flags.
func preferFlashMoELlamaServer() bool {
	return envconfig.FlashMoEEnabled() || envconfig.FlashMoELlamaServerBin() != ""
}

// FindFlashMoELlamaServer returns the anemll-flash-llama.cpp llama-server binary when built.
func FindFlashMoELlamaServer() (string, error) {
	if bin := flashMoELlamaServerOverride(); bin != "" {
		return bin, nil
	}
	for _, path := range flashMoELlamaServerCandidates(defaultLlamaCppBinarySearch()) {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
	return "", os.ErrNotExist
}

func flashMoELlamaServerOverride() string {
	if override := envconfig.FlashMoELlamaServerBin(); override != "" {
		if st, err := os.Stat(override); err == nil && !st.IsDir() {
			return override
		}
	}
	return ""
}

// flashMoELlamaServerCandidates filters Phase 17 binary search to flash-moe build dirs only.
func flashMoELlamaServerCandidates(search llamaCppBinarySearch) []string {
	seen := map[string]bool{}
	var out []string
	add := func(path string) {
		path = filepath.Clean(path)
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		out = append(out, path)
	}

	for _, path := range llamaCppBinaryCandidates("llama-server", search) {
		if strings.Contains(path, "flash-moe-llama-server") {
			add(path)
		}
	}
	return out
}
