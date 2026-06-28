package ollamarunner

import (
	"reflect"
	"testing"
)

func testCachedData(data any) cachedMultimodal {
	return cachedMultimodal{
		parts: []cachedMultimodalPart{{data: data}},
	}
}

func TestLookupCached_precomputedGlobalPinsSession(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 4)}
	const session = "agent-thread-1"
	hash := uint64(0xdead0001)
	val := testCachedData("precomputed-rows")

	cache.mu.Lock()
	cache.addGlobalLocked(hash, val)
	cache.mu.Unlock()

	mm, ok := cache.lookupCached(nil, hash, session, true, "precomputed_embedding")
	if !ok || len(mm) != 1 || mm[0].Data != "precomputed-rows" {
		t.Fatalf("global lookup: ok=%v mm=%v", ok, mm)
	}

	cache.mu.Lock()
	got, ok := cache.findSessionEmbedLocked(session, hash)
	cache.mu.Unlock()
	if !ok || !reflect.DeepEqual(got, val) {
		t.Fatalf("session pin after global hit: ok=%v got=%v", ok, got)
	}
}

func TestLookupCached_precomputedSessionWithoutGlobal(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 2)}
	const session = "agent-thread-2"
	hash := uint64(0xdead0002)
	val := testCachedData("session-only")

	cache.mu.Lock()
	cache.storeSessionEmbedLocked(session, hash, val)
	cache.addGlobalLocked(uint64(0x1), testCachedEmbed(9))
	cache.addGlobalLocked(uint64(0x2), testCachedEmbed(8))
	cache.addGlobalLocked(uint64(0x3), testCachedEmbed(7))
	_, err := cache.findGlobalLocked(hash)
	cache.mu.Unlock()
	if err != errVisionEmbedNotFound {
		t.Fatalf("expected global miss after eviction pressure, err=%v", err)
	}

	mm, ok := cache.lookupCached(nil, hash, session, true, "precomputed_embedding")
	if !ok || len(mm) != 1 || mm[0].Data != "session-only" {
		t.Fatalf("session lookup: ok=%v mm=%v", ok, mm)
	}
}

func TestLookupCached_overlayOffSkipsSession(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 4)}
	const session = "agent-thread-3"
	hash := uint64(0xdead0003)
	val := testCachedData("overlay-off")

	cache.mu.Lock()
	cache.storeSessionEmbedLocked(session, hash, val)
	cache.mu.Unlock()

	if _, ok := cache.lookupCached(nil, hash, session, false, "precomputed_embedding"); ok {
		t.Fatal("overlay off should not read session embeds")
	}
}
