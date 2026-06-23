package mlxrunner

import (
	"log/slog"
	"os"
	"strconv"
)

const (
	defaultPrefillChunk       = 8192
	defaultPrefillSnapshotInt = 32768
	defaultMTPMaxPromptTokens = 32768
)

type prefillConfig struct {
	chunkSize               int
	snapshotInterval        int // 0 disables prefill snapshots
	snapshotIntervalFromEnv bool
	clearCacheEvery         int // clear MLX allocator cache every N chunks (0 = never during prefill)
	clearCacheEveryFromEnv  bool
	materializeEvery        int // eval KV state every N prefill chunks
	materializeEveryFromEnv bool
	chunkSizeFromEnv        bool
}

const longPrefillChunkCap = 4096

func loadPrefillConfig() prefillConfig {
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

// effectivePrefillConfig tightens chunking and memory hygiene for prompts longer
// than the MTP snapshot threshold. Large batched forwards (8192 tokens) spike
// unified memory when rotating KV caches linearize past the sliding window.
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

func prefillSnapshotInterval(cfg prefillConfig, promptLen int) int {
	if !cfg.snapshotIntervalFromEnv && promptLen > defaultMTPMaxPromptTokens {
		return 0
	}
	return cfg.snapshotInterval
}

// trieSnapshotInterval controls KV snapshot capture during prefill for the prefix
// trie. MTP disables interior snapshots on long prompts, but the trie still
// needs branch points so agent turn-2 can restore rotating caches at offsets
// below the leaf (Restore rejects clamping when snap.toOffset > maxSize).
func trieSnapshotInterval(cfg prefillConfig, promptLen int) int {
	if cfg.snapshotIntervalFromEnv {
		return cfg.snapshotInterval
	}
	if promptLen > defaultMTPMaxPromptTokens {
		return 8192
	}
	return cfg.snapshotInterval
}

func mtpMaxPromptTokens() int {
	if v := positiveEnvIntKey("OLLAMA_MLX_MTP_MAX_PROMPT_TOKENS"); v > 0 {
		return v
	}
	return defaultMTPMaxPromptTokens
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
		// Only schedule the near-end snapshot when snapshots are enabled and
		// it does not duplicate the last interval boundary.
		if len(offsets) == 0 || offsets[len(offsets)-1] != end {
			offsets = append(offsets, end)
		}
	}
	return offsets
}
