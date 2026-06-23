package mlxrunner

import "testing"

func TestPrefillSnapshotOffsets(t *testing.T) {
	t.Parallel()
	if got := prefillSnapshotOffsets(4096, 8192); len(got) != 0 {
		t.Fatalf("short prompt: got %v want none", got)
	}
	offsets := prefillSnapshotOffsets(100_000, 32_768)
	if len(offsets) < 2 {
		t.Fatalf("expected multiple offsets, got %v", offsets)
	}
	if offsets[0] != 32_768 {
		t.Fatalf("first offset = %d want 32768", offsets[0])
	}
	if offsets := prefillSnapshotOffsets(100_000, 0); len(offsets) != 0 {
		t.Fatalf("interval 0: got %v want none", offsets)
	}
}

func TestMTPMaxPromptTokensDefault(t *testing.T) {
	t.Setenv("OLLAMA_MLX_MTP_MAX_PROMPT_TOKENS", "")
	if got := mtpMaxPromptTokens(); got != defaultMTPMaxPromptTokens {
		t.Fatalf("default = %d want %d", got, defaultMTPMaxPromptTokens)
	}
	t.Setenv("OLLAMA_MLX_MTP_MAX_PROMPT_TOKENS", "65536")
	if got := mtpMaxPromptTokens(); got != 65536 {
		t.Fatalf("override = %d want 65536", got)
	}
}

func TestPrefillSnapshotIntervalLongPrompt(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL", "")
	cfg := loadPrefillConfig()
	if got := prefillSnapshotInterval(cfg, 50_000); got != 0 {
		t.Fatalf("long prompt without env = %d want 0", got)
	}
	if got := prefillSnapshotInterval(cfg, 10_000); got != defaultPrefillSnapshotInt {
		t.Fatalf("short prompt = %d want %d", got, defaultPrefillSnapshotInt)
	}

	t.Setenv("OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL", "16384")
	cfg = loadPrefillConfig()
	if got := prefillSnapshotInterval(cfg, 50_000); got != 16384 {
		t.Fatalf("explicit env on long prompt = %d want 16384", got)
	}
}

func TestTrieSnapshotIntervalLongPrompt(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL", "")
	cfg := loadPrefillConfig()
	if got := trieSnapshotInterval(cfg, 50_000); got != 8192 {
		t.Fatalf("long prompt trie interval = %d want 8192", got)
	}
	if got := prefillSnapshotInterval(cfg, 50_000); got != 0 {
		t.Fatalf("long prompt MTP interval = %d want 0", got)
	}
	if got := trieSnapshotInterval(cfg, 10_000); got != defaultPrefillSnapshotInt {
		t.Fatalf("short prompt trie = %d want %d", got, defaultPrefillSnapshotInt)
	}

	t.Setenv("OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL", "16384")
	cfg = loadPrefillConfig()
	if got := trieSnapshotInterval(cfg, 50_000); got != 16384 {
		t.Fatalf("explicit env = %d want 16384", got)
	}
}

func TestEffectivePrefillConfigLongPrompt(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "")
	t.Setenv("OLLAMA_MLX_PREFILL_CLEAR_CACHE_EVERY", "")
	t.Setenv("OLLAMA_MLX_PREFILL_MATERIALIZE_EVERY", "")

	cfg := loadPrefillConfig()
	short := effectivePrefillConfig(10_000, cfg)
	if short.chunkSize != defaultPrefillChunk {
		t.Fatalf("short chunk = %d want %d", short.chunkSize, defaultPrefillChunk)
	}
	if short.clearCacheEvery != 4 {
		t.Fatalf("short clear = %d want 4", short.clearCacheEvery)
	}

	long := effectivePrefillConfig(65_000, cfg)
	if long.chunkSize != longPrefillChunkCap {
		t.Fatalf("long chunk = %d want %d", long.chunkSize, longPrefillChunkCap)
	}
	if long.clearCacheEvery != 1 {
		t.Fatalf("long clear = %d want 1", long.clearCacheEvery)
	}
	if long.materializeEvery != 1 {
		t.Fatalf("long materialize = %d want 1", long.materializeEvery)
	}

	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "8192")
	cfg = loadPrefillConfig()
	long = effectivePrefillConfig(65_000, cfg)
	if long.chunkSize != 8192 {
		t.Fatalf("env chunk override = %d want 8192", long.chunkSize)
	}
}
