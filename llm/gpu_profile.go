// Package llm — L1 per-GPU llama-server profiles for Phase 17 Go → llama-server.
//
// WHY: Python runtime/configs/gpu/*.json already calibrate throughput flags
// (batch, ubatch, n_parallel, q8_0 KV, FA). Production Linux defaults to
// Go → llama-server (ZEROLLAMA_LLAMA_SERVER=auto), which previously ignored
// those JSON profiles and fell back to f16 KV / -b 512 / -np from VRAM fit.
// This module loads the same JSON so Go launches share L1 knobs.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

// stockCacheFallback matches runtime/gpu_profiles.py _STOCK_CACHE_FALLBACK.
const stockCacheFallback = "q8_0"

// forkOnlyCacheTypes — eliza fork KV enums rejected by stock llama-server.
var forkOnlyCacheTypes = map[string]struct{}{
	"qjl1_256":   {},
	"q4_polar":   {},
	"tbq3_0":     {},
	"tbq4_0":     {},
	"tbq3_tcq":   {},
	"turbo3_0":   {},
	"turbo4_0":   {},
	"turbo3_tcq": {},
}

// GpuProfile holds throughput knobs from runtime/configs/gpu/*.json.
type GpuProfile struct {
	ID                 string
	Name               string
	Source             string // match | bucket | env
	NParallel          int
	BatchSize          int
	UBatchSize         int
	CacheTypeK         string
	CacheTypeV         string
	FlashAttn          bool
	DraftMax           int
	DraftMin           int
	DraftPMin          float64
	CacheTypesFallback bool
	EmitCtxSize        bool
}

type gpuProfileIndex struct {
	Configs []struct {
		ID   string `json:"id"`
		File string `json:"file"`
	} `json:"configs"`
	FallbackBuckets []struct {
		MaxVRAMGB     float64 `json:"max_vram_gb"`
		ConfigID      *string `json:"config_id"`
		Label         string  `json:"label"`
		ParallelScale float64 `json:"parallel_scale"`
	} `json:"fallback_buckets"`
}

type gpuProfileFile struct {
	ID                 string         `json:"id"`
	Name               string         `json:"name"`
	MatchNames         []string       `json:"match_names"`
	VRAMGB             float64        `json:"vram_gb"`
	LlamaServerFlags   map[string]any `json:"llama_server_flags"`
	ForkOnlyFlags      map[string]any `json:"_fork_only_llama_server_flags"`
	ElizaForkFlags     map[string]any `json:"_eliza_fork_llama_server_flags"`
	ElizaForkVRAMFlags map[string]any `json:"_eliza_fork_vram_llama_server_flags"`
}

// GpuProfilesEnabled mirrors Python gpu_profiles_enabled (default on).
func GpuProfilesEnabled() bool {
	env := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_GPU_PROFILE")))
	switch env {
	case "0", "false", "no", "off":
		return false
	case "1", "true", "yes", "on":
		return true
	}
	return true
}

func llamaForkEnabled() bool {
	env := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_LLAMA_FORK")))
	switch env {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func gpuProfilesDir() string {
	if root := strings.TrimSpace(os.Getenv("ZEROLLAMA_REPO")); root != "" {
		p := filepath.Join(root, "runtime", "configs", "gpu")
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			return p
		}
	}
	// Common CT / lab layouts.
	for _, cand := range []string{
		filepath.Join("runtime", "configs", "gpu"),
		filepath.Join("..", "runtime", "configs", "gpu"),
	} {
		if abs, err := filepath.Abs(cand); err == nil {
			if st, err := os.Stat(abs); err == nil && st.IsDir() {
				return abs
			}
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		p := filepath.Join(home, "zerollama", "runtime", "configs", "gpu")
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			return p
		}
	}
	return ""
}

func loadGPUProfileJSON(path string) (*gpuProfileFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg gpuProfileFile
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func loadGPUProfileIndex(dir string) (*gpuProfileIndex, error) {
	b, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		return nil, err
	}
	var idx gpuProfileIndex
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, err
	}
	return &idx, nil
}

func sanitizeCacheType(raw string) (string, bool) {
	key := strings.ToLower(strings.TrimSpace(raw))
	if key == "" {
		return "", false
	}
	if _, ok := forkOnlyCacheTypes[key]; ok {
		return stockCacheFallback, true
	}
	if strings.HasPrefix(key, "tbq") || strings.HasPrefix(key, "qjl") || strings.Contains(key, "polar") || strings.HasPrefix(key, "turbo") {
		return stockCacheFallback, true
	}
	return key, false
}

func flagsFromGPUConfig(cfg *gpuProfileFile, forkEnabled bool) (map[string]any, bool) {
	base := map[string]any{}
	for k, v := range cfg.LlamaServerFlags {
		base[k] = v
	}
	if !forkEnabled {
		fallback := false
		for _, field := range []string{"cache_type_k", "cache_type_v"} {
			raw, _ := base[field].(string)
			if raw == "" {
				continue
			}
			safe, fb := sanitizeCacheType(raw)
			base[field] = safe
			fallback = fallback || fb
		}
		delete(base, "ctx_checkpoints")
		delete(base, "ctx_checkpoint_interval")
		return base, fallback
	}
	for k, v := range cfg.ForkOnlyFlags {
		base[k] = v
	}
	profile := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_LLAMA_FORK_PROFILE")))
	if profile == "" {
		profile = "vram"
	}
	switch profile {
	case "speed", "qjl", "polar", "qjl_polar":
		for k, v := range cfg.ElizaForkFlags {
			base[k] = v
		}
	default:
		if len(cfg.ElizaForkVRAMFlags) > 0 {
			for k, v := range cfg.ElizaForkVRAMFlags {
				base[k] = v
			}
		} else {
			for k, v := range cfg.ElizaForkFlags {
				base[k] = v
			}
		}
	}
	return base, false
}

func profileFromFlags(cfg *gpuProfileFile, flags map[string]any, source string, cacheFB bool) *GpuProfile {
	p := &GpuProfile{
		ID:                 cfg.ID,
		Name:               cfg.Name,
		Source:             source,
		CacheTypesFallback: cacheFB,
		EmitCtxSize:        true,
		FlashAttn:          truthy(flags["flash_attn"]),
	}
	if env := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_GPU_PROFILE_CTX"))); env == "0" || env == "false" || env == "no" || env == "off" {
		p.EmitCtxSize = false
	}
	p.NParallel = intFromAny(flags["n_parallel"])
	p.BatchSize = intFromAny(flags["batch_size"])
	p.UBatchSize = intFromAny(flags["ubatch_size"])
	if s, _ := flags["cache_type_k"].(string); s != "" {
		p.CacheTypeK = s
	}
	if s, _ := flags["cache_type_v"].(string); s != "" {
		p.CacheTypeV = s
	}
	p.DraftMax = intFromAny(flags["draft_max"])
	p.DraftMin = intFromAny(flags["draft_min"])
	p.DraftPMin = floatFromAny(flags["draft_p_min"])
	return p
}

func truthy(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case string:
		s := strings.ToLower(strings.TrimSpace(t))
		return s == "1" || s == "true" || s == "yes" || s == "on"
	case float64:
		return t != 0
	case int:
		return t != 0
	default:
		return false
	}
}

func intFromAny(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	case json.Number:
		i, _ := t.Int64()
		return int(i)
	case string:
		i, _ := strconv.Atoi(strings.TrimSpace(t))
		return i
	default:
		return 0
	}
}

func floatFromAny(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(strings.TrimSpace(t), 64)
		return f
	default:
		return 0
	}
}

func nameMatches(gpuName string, matchNames []string) bool {
	n := strings.ToLower(strings.TrimSpace(gpuName))
	if n == "" {
		return false
	}
	for _, m := range matchNames {
		m = strings.ToLower(strings.TrimSpace(m))
		if m == "" {
			continue
		}
		if strings.Contains(n, m) || strings.Contains(m, n) {
			return true
		}
	}
	return false
}

func primaryGPUIdentity(gpus []ml.DeviceInfo) (name string, vramGB float64) {
	if len(gpus) > 0 {
		g := gpus[0]
		name = g.Description
		if name == "" {
			name = g.Name
		}
		if g.TotalMemory > 0 {
			vramGB = float64(g.TotalMemory) / (1024 * 1024 * 1024)
		}
	}
	// WHY nvidia-smi fallback: bootstrap / lab / edge discovery can hand NewLlamaServerRunner
	// an empty DeviceInfo list (or zero TotalMemory) while CUDA inference still works.
	// Python L1 uses nvidia-smi for the same reason — without this, rtx-5080 never matches
	// and Phase 17 silently keeps f16 / -b 512.
	if name == "" || vramGB <= 0 {
		if n, v := nvidiaSMIGPUIdentity(); n != "" || v > 0 {
			if name == "" {
				name = n
			}
			if vramGB <= 0 {
				vramGB = v
			}
		}
	}
	return name, vramGB
}

// nvidiaSMIGPUIdentity returns the first GPU name + VRAM GiB from nvidia-smi.
func nvidiaSMIGPUIdentity() (name string, vramGB float64) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return "", 0
	}
	line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	if line == "" {
		return "", 0
	}
	parts := strings.SplitN(line, ",", 2)
	name = strings.TrimSpace(parts[0])
	if len(parts) > 1 {
		mb, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err == nil && mb > 0 {
			vramGB = mb / 1024.0
		}
	}
	return name, vramGB
}

// NvidiaSMIFreeVRAMBytes sums free VRAM across GPUs via nvidia-smi (bytes).
// WHY: Go discover can return an empty DeviceInfo list while CUDA still works;
// auto -np fit then sees free_vram=0 and collapses to np=1. Python admission
// already probes nvidia-smi — share that fallback for Phase 17 fit.
func NvidiaSMIFreeVRAMBytes() uint64 {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	var total uint64
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		mb, err := strconv.ParseFloat(line, 64)
		if err != nil || mb <= 0 {
			continue
		}
		total += uint64(mb * 1024 * 1024)
	}
	return total
}

// SelectGpuProfile loads runtime/configs/gpu and picks a profile for gpus.
// Returns nil when profiles are disabled or no match.
func SelectGpuProfile(gpus []ml.DeviceInfo) *GpuProfile {
	if !GpuProfilesEnabled() {
		return nil
	}
	dir := gpuProfilesDir()
	if dir == "" {
		return nil
	}
	idx, err := loadGPUProfileIndex(dir)
	if err != nil {
		slog.Debug("gpu profile: index load failed", "error", err, "dir", dir)
		return nil
	}
	forkOn := llamaForkEnabled()

	if forced := strings.TrimSpace(os.Getenv("ZEROLLAMA_GPU_PROFILE_ID")); forced != "" {
		for _, entry := range idx.Configs {
			if entry.ID != forced && !strings.EqualFold(entry.ID, forced) {
				continue
			}
			cfg, err := loadGPUProfileJSON(filepath.Join(dir, entry.File))
			if err != nil {
				slog.Warn("gpu profile: forced id load failed", "id", forced, "error", err)
				return nil
			}
			flags, fb := flagsFromGPUConfig(cfg, forkOn)
			return profileFromFlags(cfg, flags, "env", fb)
		}
		slog.Warn("gpu profile: ZEROLLAMA_GPU_PROFILE_ID not found", "id", forced)
	}

	name, vramGB := primaryGPUIdentity(gpus)
	for _, entry := range idx.Configs {
		cfg, err := loadGPUProfileJSON(filepath.Join(dir, entry.File))
		if err != nil {
			continue
		}
		if nameMatches(name, cfg.MatchNames) {
			flags, fb := flagsFromGPUConfig(cfg, forkOn)
			return profileFromFlags(cfg, flags, "match", fb)
		}
	}

	if vramGB <= 0 {
		return nil
	}
	for _, bucket := range idx.FallbackBuckets {
		if vramGB > bucket.MaxVRAMGB {
			continue
		}
		if bucket.ConfigID == nil || *bucket.ConfigID == "" {
			return nil
		}
		var cfg *gpuProfileFile
		for _, entry := range idx.Configs {
			if entry.ID == *bucket.ConfigID {
				cfg, err = loadGPUProfileJSON(filepath.Join(dir, entry.File))
				break
			}
		}
		if cfg == nil || err != nil {
			return nil
		}
		flags, fb := flagsFromGPUConfig(cfg, forkOn)
		if scale := bucket.ParallelScale; scale > 0 && scale != 1.0 {
			if n := intFromAny(flags["n_parallel"]); n > 0 {
				flags = copyMap(flags)
				flags["n_parallel"] = max(1, int(float64(n)*scale))
			}
		}
		p := profileFromFlags(cfg, flags, "bucket", fb)
		return p
	}
	return nil
}

func copyMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// ApplyGpuProfileToLaunch mutates launch opts for Phase 17 llama-server.
//
// Precedence (mirrors Python):
//  1. Explicit OLLAMA_KV_CACHE_TYPE / request KvCacheType / OLLAMA_FLASH_ATTENTION win
//  2. Profile fills throughput knobs when enabled
//  3. Embedding keeps np=1; vision does not raise -np above scheduler fit
//  4. Never raises numParallel above the scheduler-fitted value (VRAM-safe)
//  5. Caps numParallel to profile.n_parallel when profile is below the fitted/env value
//  6. Does not apply profile ctx_size (scheduler owns NumCtx; honor PROFILE_CTX=0 spirit)
func ApplyGpuProfileToLaunch(launch *llamaServerLaunchConfig, profile *GpuProfile) {
	if launch == nil || profile == nil {
		return
	}
	if launch.embedding {
		return
	}

	opts := launch.opts
	kvExplicit := strings.TrimSpace(opts.KvCacheType) != "" || strings.TrimSpace(envconfig.KvCacheType()) != ""
	faUserSet := envconfig.FlashAttention(false) == envconfig.FlashAttention(true)

	if !kvExplicit {
		kt := profile.CacheTypeK
		if kt == "" {
			kt = profile.CacheTypeV
		}
		if kt != "" {
			launch.kvCacheType = kt
			opts.KvCacheType = kt
		}
	}

	if profile.BatchSize > 0 {
		opts.NumBatch = profile.BatchSize
	}
	if profile.UBatchSize > 0 {
		launch.ubatchSize = profile.UBatchSize
	} else if profile.BatchSize > 0 {
		launch.ubatchSize = profile.BatchSize
	}

	if profile.FlashAttn && !faUserSet {
		launch.forceFlashAttn = true
	}

	// Cap -np to calibrated profile; never raise above scheduler fit.
	vision := len(launch.projectors) > 0
	if !vision && profile.NParallel > 0 && launch.numParallel > profile.NParallel {
		launch.numParallel = profile.NParallel
	}

	// Draft bounds only when speculative/MTP is actually on — avoid applying
	// L1 draft_max/p_min to plain text loads.
	specOn := launch.config.EnableMTP ||
		strings.TrimSpace(launch.config.SpecType) != "" ||
		strings.TrimSpace(launch.config.DraftModelPath) != ""
	if specOn {
		if profile.DraftMax > 0 {
			opts.DraftNumPredict = profile.DraftMax
		}
		if profile.DraftPMin > 0 {
			launch.draftPMin = profile.DraftPMin
		}
	}

	launch.opts = opts
	launch.gpuProfileID = profile.ID
	RememberGpuProfileID(profile.ID)

	slog.Info("applied L1 GPU profile to llama-server launch",
		"profile", profile.ID,
		"profile_source", profile.Source,
		"num_parallel", launch.numParallel,
		"num_batch", opts.NumBatch,
		"ubatch", launch.ubatchSize,
		"kv_cache_type", launch.kvCacheType,
		"flash_attn_on", launch.forceFlashAttn,
	)
}

// GpuProfileKVAndBatch returns profile KV/batch for VRAM estimates when unset.
// Used by the scheduler's ggmlLoadProfileFor so auto -np fit matches launch KV.
func GpuProfileKVAndBatch(gpus []ml.DeviceInfo, opts api.Options) (kv string, batch int, nParallel int, ok bool) {
	p := SelectGpuProfile(gpus)
	if p == nil {
		return "", 0, 0, false
	}
	kvExplicit := strings.TrimSpace(opts.KvCacheType) != "" || strings.TrimSpace(envconfig.KvCacheType()) != ""
	if !kvExplicit {
		kv = p.CacheTypeK
		if kv == "" {
			kv = p.CacheTypeV
		}
	}
	if p.BatchSize > 0 {
		batch = p.BatchSize
	}
	if p.NParallel > 0 {
		nParallel = p.NParallel
	}
	return kv, batch, nParallel, true
}

// FormatGpuProfileSummary is a compact debug string.
func FormatGpuProfileSummary(p *GpuProfile) string {
	if p == nil {
		return "none"
	}
	return fmt.Sprintf("%s(%s) np=%d b=%d ub=%d kv=%s/%s fa=%v",
		p.ID, p.Source, p.NParallel, p.BatchSize, p.UBatchSize, p.CacheTypeK, p.CacheTypeV, p.FlashAttn)
}

var (
	lastGpuProfileMu sync.Mutex
	lastGpuProfileID string
)

// RememberGpuProfileID stores the last L1 profile id applied on a Go llama-server launch.
func RememberGpuProfileID(id string) {
	lastGpuProfileMu.Lock()
	defer lastGpuProfileMu.Unlock()
	lastGpuProfileID = id
}

// LastGpuProfileID returns the last applied L1 profile id (empty if none yet).
func LastGpuProfileID() string {
	lastGpuProfileMu.Lock()
	defer lastGpuProfileMu.Unlock()
	return lastGpuProfileID
}
