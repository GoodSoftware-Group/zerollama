package mlxrunner

import (
	"log/slog"
	"os"
	"slices"
	"strconv"
	"strings"
)

const (
	defaultPrefillChunk       = 8192
	defaultPrefillSnapshotInt = 32768
	defaultMTPMaxPromptTokens = 32768
	longPrefillChunkCap       = 4096
)

type prefillConfig struct {
	chunkSize               int
	snapshotInterval        int // 0 disables MTP interior snapshots during prefill
	clearCacheEvery         int // clear MLX allocator cache every N chunks (0 = never)
	materializeEvery        int // eval KV state every N prefill chunks
	chunkSizeFromEnv        bool
	clearCacheEveryFromEnv  bool
	materializeEveryFromEnv bool
	snapshotIntervalFromEnv bool
}

func loadPrefillBaseConfig() prefillConfig {
	cfg := prefillConfig{
		chunkSize:        defaultPrefillChunk,
		snapshotInterval: defaultPrefillSnapshotInt,
		clearCacheEvery:  4,
		materializeEvery: 4,
	}
	if v := positiveEnvIntKey("OLLAMA_MLX_PREFILL_CHUNK"); v > 0 {
		cfg.chunkSize = v
		cfg.chunkSizeFromEnv = true
	}
	if raw := os.Getenv("OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL"); raw != "" {
		cfg.snapshotIntervalFromEnv = true
		if raw == "0" || raw == "off" || raw == "false" {
			cfg.snapshotInterval = 0
		} else if v, err := strconv.Atoi(raw); err == nil && v > 0 {
			cfg.snapshotInterval = v
		} else {
			slog.Warn("invalid MLX env setting", "key", "OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL", "value", raw)
		}
	}
	if raw := os.Getenv("OLLAMA_MLX_PREFILL_CLEAR_CACHE_EVERY"); raw != "" {
		cfg.clearCacheEveryFromEnv = true
		if v := positiveEnvIntKey("OLLAMA_MLX_PREFILL_CLEAR_CACHE_EVERY"); v >= 0 {
			cfg.clearCacheEvery = v
		}
	}
	if v := positiveEnvIntKey("OLLAMA_MLX_PREFILL_MATERIALIZE_EVERY"); v > 0 {
		cfg.materializeEvery = v
		cfg.materializeEveryFromEnv = true
	}
	return cfg
}

func positiveEnvIntKey(key string) int {
	raw := os.Getenv(key)
	if raw == "" {
		return 0
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v < 0 {
		slog.Warn("invalid MLX env setting", "key", key, "value", raw)
		return 0
	}
	return v
}

func defaultPrefillConfig(promptLen, slidingWindow int) prefillConfig {
	cfg := effectivePrefillConfig(promptLen, loadPrefillBaseConfig())
	if slidingWindow > 0 {
		cfg = capPrefillChunkForRotatingKV(cfg, slidingWindow)
	}
	return cfg
}

// effectivePrefillConfig tightens chunking and memory hygiene for prompts longer
// than the MTP snapshot threshold. Large batched forwards spike unified memory
// when rotating KV caches linearize past the sliding window.
func effectivePrefillConfig(promptLen int, cfg prefillConfig) prefillConfig {
	if promptLen <= defaultMTPMaxPromptTokens {
		return cfg
	}
	out := cfg
	if !cfg.clearCacheEveryFromEnv {
		out.clearCacheEvery = 1
	}
	if !cfg.materializeEveryFromEnv {
		out.materializeEvery = 1
	}
	if !cfg.chunkSizeFromEnv && out.chunkSize > longPrefillChunkCap {
		out.chunkSize = longPrefillChunkCap
	}
	return out
}

// capPrefillChunkForRotatingKV limits batched prefill to 2× sliding_window (Gemma4
// OptiQ: 2048 when window=1024). Larger chunks spike unified memory on M4 Max.
func capPrefillChunkForRotatingKV(cfg prefillConfig, slidingWindow int) prefillConfig {
	if cfg.chunkSizeFromEnv || slidingWindow <= 0 {
		return cfg
	}
	maxChunk := 2 * slidingWindow
	if cfg.chunkSize <= maxChunk {
		return cfg
	}
	out := cfg
	out.chunkSize = maxChunk
	return out
}

// capPrefillChunkForWorkingSet tightens chunk hygiene when unified memory is
// already most of the recommended working set (mlx-serve long-prompt OOM
// path). Env chunk/materialize/clear overrides still win.
func capPrefillChunkForWorkingSet(cfg prefillConfig, active, recommended int) prefillConfig {
	if recommended <= 0 || active <= 0 || active < recommended*3/4 {
		return cfg
	}
	out := cfg
	if !cfg.chunkSizeFromEnv && out.chunkSize > 1024 {
		out.chunkSize = 1024
	}
	if !cfg.materializeEveryFromEnv {
		out.materializeEvery = 1
	}
	if !cfg.clearCacheEveryFromEnv {
		out.clearCacheEvery = 1
	}
	return out
}

// prefillConfigForCachedTail relaxes allocator hygiene when most of a long prompt
// is already resident and only a short tail remains to evaluate.
// Deprecated: use tunePrefillConfig.
func prefillConfigForCachedTail(cfg prefillConfig, cachedPrefix, remaining int) prefillConfig {
	return tunePrefillConfig(cfg, prefillTuneInput{
		total:          cachedPrefix + remaining,
		cachedPrefix:   cachedPrefix,
		remaining:      remaining,
		promptCacheKey: "",
	})
}

type prefillTuneInput struct {
	total, cachedPrefix, remaining int
	promptCacheKey                 string
	fastPath                       bool
	sameBranch                     bool
}

// tunePrefillConfig picks prefill chunk hygiene automatically from cache shape.
// Env overrides (FromEnv flags) still win when set.
func tunePrefillConfig(cfg prefillConfig, in prefillTuneInput) prefillConfig {
	if in.total <= 0 {
		return cfg
	}
	out := cfg
	ratio := float64(in.cachedPrefix) / float64(in.total)
	agentKey := strings.TrimSpace(in.promptCacheKey) != ""

	// Long-prompt trie restore with a short tail (M15): relax aggressive per-chunk hygiene.
	if in.cachedPrefix >= defaultMTPMaxPromptTokens && in.remaining <= 4096 {
		if !out.materializeEveryFromEnv {
			out.materializeEvery = 4
		}
		if !out.clearCacheEveryFromEnv {
			out.clearCacheEvery = 4
		}
	}

	// Hot agent continuation: resident KV + short tail — avoid per-chunk eval/clear.
	// sameBranch covers trie restore without fast_path (still skipped page-in/out).
	if agentKey && (in.fastPath || in.sameBranch || (ratio >= 0.80 && in.remaining <= 8192)) {
		if !out.materializeEveryFromEnv && out.materializeEvery < 8 {
			out.materializeEvery = 8
		}
		if !out.clearCacheEveryFromEnv && out.clearCacheEvery < 8 {
			out.clearCacheEvery = 8
		}
	}

	// Trie-heavy agent restore with a very small delta to evaluate.
	if agentKey && ratio >= 0.90 && in.remaining <= 1024 {
		if !out.materializeEveryFromEnv && out.materializeEvery < 16 {
			out.materializeEvery = 16
		}
		if !out.clearCacheEveryFromEnv && out.clearCacheEvery < 16 {
			out.clearCacheEvery = 16
		}
	}

	return out
}

func prefillSnapshotInterval(cfg prefillConfig, promptLen int) int {
	if !cfg.snapshotIntervalFromEnv && promptLen > defaultMTPMaxPromptTokens {
		return 0
	}
	return cfg.snapshotInterval
}

// trieSnapshotInterval controls KV snapshot capture during prefill for the prefix
// trie. MTP disables interior snapshots on long prompts, but the trie still
// needs branch points so agent turn-2 can restore rotating caches.
func trieSnapshotInterval(cfg prefillConfig, promptLen int, slidingWindow int, promptCacheKey string) int {
	if cfg.snapshotIntervalFromEnv {
		return cfg.snapshotInterval
	}
	if promptLen > defaultMTPMaxPromptTokens {
		return 8192
	}
	// Agent threads and rotating KV need interior restore points even when the
	// prompt is shorter than the default MTP snapshot interval (32k). Without
	// them, capTrieMatchForRestore can only rewind to the start of a multi-k
	// edge and KV restore fails on Gemma4 OptiQ.
	//
	// Use 1× sliding_window (not 2×): finer boundaries reduce partial-restore
	// loss when message truncation rewrites the prompt prefix between turns.
	if strings.TrimSpace(promptCacheKey) != "" || slidingWindow > 0 {
		if slidingWindow > 0 {
			iv := slidingWindow
			if iv < 1024 {
				iv = 1024
			}
			return iv
		}
		return 8192
	}
	return cfg.snapshotInterval
}

func prefillSnapshotOffsets(promptLen, interval int) []int {
	if interval <= 0 || promptLen <= interval {
		return nil
	}
	var offsets []int
	for offset := interval; offset < promptLen; offset += interval {
		offsets = append(offsets, offset)
	}
	const preThinking = 4
	if end := promptLen - preThinking; end > 0 && end > interval {
		if len(offsets) == 0 || offsets[len(offsets)-1] != end {
			offsets = append(offsets, end)
		}
	}
	return offsets
}

// buildTriePrefillSnapshotOffsets schedules trie restore points during prefill.
func buildTriePrefillSnapshotOffsets(cfg prefillConfig, promptLen int, promptCacheKey string, slidingWindow int) []int {
	offsets := prefillSnapshotOffsets(promptLen, trieSnapshotInterval(cfg, promptLen, slidingWindow, promptCacheKey))
	if key := strings.TrimSpace(promptCacheKey); key != "" {
		if end := promptLen - 1; end > 0 {
			offsets = appendUniqueSnapshotOffset(offsets, end)
		}
	}
	return offsets
}

func appendUniqueSnapshotOffset(offsets []int, v int) []int {
	if slices.Contains(offsets, v) {
		return offsets
	}
	offsets = append(offsets, v)
	slices.Sort(offsets)
	return offsets
}
