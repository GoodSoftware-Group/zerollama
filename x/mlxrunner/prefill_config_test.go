package mlxrunner

import (
	"slices"
	"testing"
)

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

func TestPrefillSnapshotIntervalLongPrompt(t *testing.T) {
	cfg := defaultPrefillConfig(50_000, 0)
	if got := prefillSnapshotInterval(cfg, 50_000); got != 0 {
		t.Fatalf("long prompt MTP interval = %d want 0", got)
	}
	if got := trieSnapshotInterval(cfg, 50_000, 0, ""); got != 8192 {
		t.Fatalf("long prompt trie interval = %d want 8192", got)
	}
	short := defaultPrefillConfig(10_000, 0)
	if got := prefillSnapshotInterval(short, 10_000); got != defaultPrefillSnapshotInt {
		t.Fatalf("short prompt = %d want %d", got, defaultPrefillSnapshotInt)
	}
	if got := trieSnapshotInterval(short, 10_000, 0, ""); got != defaultPrefillSnapshotInt {
		t.Fatalf("short prompt trie = %d want %d", got, defaultPrefillSnapshotInt)
	}
}

func TestTrieSnapshotIntervalAgentRotatingKV(t *testing.T) {
	cfg := defaultPrefillConfig(6587, 1024)
	if got := trieSnapshotInterval(cfg, 6587, 1024, ""); got != 1024 {
		t.Fatalf("rotating KV = %d want 1024", got)
	}
	if got := trieSnapshotInterval(cfg, 6587, 0, "hermes:session"); got != 8192 {
		t.Fatalf("agent key only = %d want 8192", got)
	}
	if got := trieSnapshotInterval(cfg, 6587, 1024, "hermes:session"); got != 1024 {
		t.Fatalf("agent + rotating = %d want 1024", got)
	}
}

func TestBuildTriePrefillSnapshotOffsets(t *testing.T) {
	t.Parallel()
	cfg := defaultPrefillConfig(100_000, 0)
	offsets := buildTriePrefillSnapshotOffsets(cfg, 100_000, "", 0)
	if len(offsets) < 2 {
		t.Fatalf("expected trie offsets, got %v", offsets)
	}
	withKey := buildTriePrefillSnapshotOffsets(cfg, 100_000, "conv:abc", 0)
	if !slices.Contains(withKey, 99_999) {
		t.Fatalf("prompt cache key end missing from %v", withKey)
	}
	agent := defaultPrefillConfig(6587, 1024)
	agentOffsets := buildTriePrefillSnapshotOffsets(agent, 6587, "hermes:abc", 1024)
	for _, want := range []int{1024, 2048, 3072, 4096, 5120, 6144, 6586} {
		if !slices.Contains(agentOffsets, want) {
			t.Fatalf("agent rotating offsets missing %d in %v", want, agentOffsets)
		}
	}
}

func TestEffectivePrefillConfigLongPrompt(t *testing.T) {
	short := defaultPrefillConfig(10_000, 0)
	if short.chunkSize != defaultPrefillChunk {
		t.Fatalf("short chunk = %d want %d", short.chunkSize, defaultPrefillChunk)
	}
	if short.clearCacheEvery != 4 {
		t.Fatalf("short clear = %d want 4", short.clearCacheEvery)
	}

	long := defaultPrefillConfig(65_000, 0)
	if long.chunkSize != longPrefillChunkCap {
		t.Fatalf("long chunk = %d want %d", long.chunkSize, longPrefillChunkCap)
	}
	if long.clearCacheEvery != 1 {
		t.Fatalf("long clear = %d want 1", long.clearCacheEvery)
	}
	if long.materializeEvery != 1 {
		t.Fatalf("long materialize = %d want 1", long.materializeEvery)
	}
}

func TestPrefillConfigForCachedTail(t *testing.T) {
	long := defaultPrefillConfig(65_000, 0)
	tail := tunePrefillConfig(long, prefillTuneInput{
		total: 65_000, cachedPrefix: 60_000, remaining: 2_000,
		promptCacheKey: "hermes:main",
	})
	if tail.materializeEvery < 8 || tail.clearCacheEvery < 8 {
		t.Fatalf("agent hot tail = mat %d clear %d want >=8", tail.materializeEvery, tail.clearCacheEvery)
	}
	highHit := tunePrefillConfig(long, prefillTuneInput{
		total: 50_000, cachedPrefix: 48_000, remaining: 500,
		promptCacheKey: "hermes:main",
	})
	if highHit.materializeEvery < 16 {
		t.Fatalf("high-hit tail materialize=%d want >=16", highHit.materializeEvery)
	}
	deprecated := prefillConfigForCachedTail(long, 60_000, 2_000)
	if deprecated.materializeEvery != 4 || deprecated.clearCacheEvery != 4 {
		t.Fatalf("deprecated cached tail = mat %d clear %d want 4/4", deprecated.materializeEvery, deprecated.clearCacheEvery)
	}
	unchanged := tunePrefillConfig(long, prefillTuneInput{
		total: 65_000, cachedPrefix: 10_000, remaining: 2_000,
	})
	if unchanged.materializeEvery != 1 {
		t.Fatalf("low cache ratio should keep cold hygiene: mat=%d", unchanged.materializeEvery)
	}
}

func TestTunePrefillConfigFastPath(t *testing.T) {
	long := defaultPrefillConfig(65_000, 1024)
	out := tunePrefillConfig(long, prefillTuneInput{
		total: 50_000, cachedPrefix: 49_000, remaining: 800,
		promptCacheKey: "hermes:main", fastPath: true,
	})
	if out.materializeEvery < 8 {
		t.Fatalf("fast_path materialize=%d want >=8", out.materializeEvery)
	}
}

func TestCapPrefillChunkForRotatingKV(t *testing.T) {
	optiq := defaultPrefillConfig(65_000, 1024)
	if optiq.chunkSize != 2048 {
		t.Fatalf("gemma4 optiq long prefill chunk = %d want 2048", optiq.chunkSize)
	}
	short := defaultPrefillConfig(10_000, 1024)
	if short.chunkSize != 2048 {
		t.Fatalf("gemma4 optiq short prefill chunk = %d want 2048", short.chunkSize)
	}
}

func TestPrefillChunkEnvOverridesRotatingCap(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "4096")
	got := defaultPrefillConfig(65_000, 1024)
	if got.chunkSize != 4096 {
		t.Fatalf("env chunk = %d want 4096", got.chunkSize)
	}
}

func TestCapPrefillChunkForWorkingSet(t *testing.T) {
	t.Parallel()
	base := prefillConfig{chunkSize: 8192, materializeEvery: 4, clearCacheEvery: 4}
	got := capPrefillChunkForWorkingSet(base, 7<<30, 8<<30)
	if got.chunkSize != 1024 || got.materializeEvery != 1 || got.clearCacheEvery != 1 {
		t.Fatalf("tight working set: %+v", got)
	}
	loose := capPrefillChunkForWorkingSet(base, 1<<30, 8<<30)
	if loose.chunkSize != 8192 {
		t.Fatalf("roomy working set should keep chunk, got %d", loose.chunkSize)
	}
	env := prefillConfig{chunkSize: 4096, chunkSizeFromEnv: true, materializeEvery: 4, clearCacheEvery: 4}
	kept := capPrefillChunkForWorkingSet(env, 7<<30, 8<<30)
	if kept.chunkSize != 4096 {
		t.Fatalf("env chunk must win, got %d", kept.chunkSize)
	}
}
