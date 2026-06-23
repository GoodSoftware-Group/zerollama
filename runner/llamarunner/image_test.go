package llamarunner

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/ollama/ollama/llama"
)

func TestImageCache(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, defaultImageCacheSize)}

	valA := []llama.MtmdChunk{{Embed: []float32{0.1, 0.2}}, {Embed: []float32{0.3}}}
	valB := []llama.MtmdChunk{{Embed: []float32{0.4}}, {Embed: []float32{0.5}}, {Embed: []float32{0.6}}}
	valC := []llama.MtmdChunk{{Embed: []float32{0.7}}}
	valD := []llama.MtmdChunk{{Embed: []float32{0.8}}}
	valE := []llama.MtmdChunk{{Embed: []float32{0.9}}}

	// Empty cache
	result, err := cache.findImage(0x5adb61d31933a946)
	if err != errImageNotFound {
		t.Errorf("found result in empty cache: result %v, err %v", result, err)
	}

	// Insert A
	cache.addImage(0x5adb61d31933a946, valA)

	result, err = cache.findImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Insert B
	cache.addImage(0x011551369a34a901, valB)

	result, err = cache.findImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valB) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Replace B with C
	cache.addImage(0x011551369a34a901, valC)

	result, err = cache.findImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valC) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Evict A
	cache.addImage(0x756b218a517e7353, valB)
	cache.addImage(0x75e5e8d35d7e3967, valD)
	cache.addImage(0xd96f7f268ca0646e, valE)

	result, err = cache.findImage(0x5adb61d31933a946)
	if reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x756b218a517e7353)
	if !reflect.DeepEqual(result, valB) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valC) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x75e5e8d35d7e3967)
	if !reflect.DeepEqual(result, valD) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0xd96f7f268ca0646e)
	if !reflect.DeepEqual(result, valE) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
}

// TestImageCache_enlargedForVideoAgents verifies that a larger cache (simulating
// OLLAMA_IMAGE_EMBED_CACHE_SIZE=8 for video agents) retains all inserted frames
// without eviction, unlike the 4-slot default.
func TestImageCache_enlargedForVideoAgents(t *testing.T) {
	const videoAgentCacheSize = 8
	cache := ImageContext{images: make([]imageCache, videoAgentCacheSize)}

	frames := make([][]llama.MtmdChunk, videoAgentCacheSize)
	keys := make([]uint64, videoAgentCacheSize)
	for i := range frames {
		frames[i] = []llama.MtmdChunk{{Embed: []float32{float32(i) + 0.1}}}
		keys[i] = uint64(0x1000000000000000 + i)
		cache.addImage(keys[i], frames[i])
	}

	// All 8 frames must still be present (no eviction on equal-size fill).
	for i, key := range keys {
		got, err := cache.findImage(key)
		if err != nil {
			t.Errorf("frame %d evicted prematurely: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(got, frames[i]) {
			t.Errorf("frame %d: got %v, want %v", i, got, frames[i])
		}
	}

	// Adding a 9th entry must evict exactly one of the existing entries.
	extra := []llama.MtmdChunk{{Embed: []float32{99.0}}}
	extraKey := uint64(0x9000000000000000)
	cache.addImage(extraKey, extra)

	got, err := cache.findImage(extraKey)
	if err != nil || !reflect.DeepEqual(got, extra) {
		t.Errorf("extra frame not found after insert: %v %v", got, err)
	}

	// Exactly one of the original frames should be gone.
	evicted := 0
	for _, key := range keys {
		if _, err := cache.findImage(key); err == errImageNotFound {
			evicted++
		}
	}
	if evicted != 1 {
		t.Errorf("expected exactly 1 eviction, got %d", evicted)
	}
}

func TestGrowCacheForDistinctFrames(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 4)}
	cache.growCacheForDistinctFrames(8)
	if len(cache.images) != 8 {
		t.Fatalf("len=%d want 8", len(cache.images))
	}
	cache.growCacheForDistinctFrames(4)
	if len(cache.images) != 8 {
		t.Fatalf("shrink not expected: len=%d", len(cache.images))
	}
}

func TestSessionEmbedOverlay_survivesGlobalEviction(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 2)}
	const session = "agent-thread-1"
	hash := uint64(0xabc123)
	val := []llama.MtmdChunk{{Embed: []float32{1.0, 2.0}}}

	cache.mu.Lock()
	cache.storeSessionEmbedLocked(session, hash, val)
	cache.mu.Unlock()

	// Fill global LRU and evict the only session frame from global slots.
	cache.addImage(uint64(0x1), []llama.MtmdChunk{{Embed: []float32{9}}})
	cache.addImage(uint64(0x2), []llama.MtmdChunk{{Embed: []float32{8}}})
	cache.addImage(uint64(0x3), []llama.MtmdChunk{{Embed: []float32{7}}})

	if _, err := cache.findImage(hash); err != errImageNotFound {
		t.Fatalf("expected global miss after eviction, got err=%v", err)
	}

	cache.mu.Lock()
	got, ok := cache.findSessionEmbedLocked(session, hash)
	cache.mu.Unlock()
	if !ok || !reflect.DeepEqual(got, val) {
		t.Fatalf("session overlay miss: ok=%v got=%v want=%v", ok, got, val)
	}
}

func TestSessionEmbedOverlay_noCrossSession(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 2)}
	hash := uint64(0xdef456)
	val := []llama.MtmdChunk{{Embed: []float32{3.0}}}

	cache.mu.Lock()
	cache.storeSessionEmbedLocked("session-a", hash, val)
	_, ok := cache.findSessionEmbedLocked("session-b", hash)
	cache.mu.Unlock()
	if ok {
		t.Fatal("session-b should not see session-a embeds")
	}
}

func TestSessionEmbedOverlay_globalHitPromotesToSession(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 4)}
	const session = "agent-thread-2"
	hash := uint64(0xfeed)
	val := []llama.MtmdChunk{{Embed: []float32{4.0}}}

	cache.addImage(hash, val)

	cache.mu.Lock()
	cache.storeSessionEmbedLocked(session, hash, val)
	_, ok := cache.findSessionEmbedLocked(session, hash)
	cache.mu.Unlock()
	if !ok {
		t.Fatal("expected session overlay after global hit promotion")
	}
}

func TestSessionEmbedOverlay_ttlEviction(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 2)}
	const session = "stale-session"
	hash := uint64(0xdead)
	val := []llama.MtmdChunk{{Embed: []float32{5.0}}}

	cache.mu.Lock()
	cache.sessionEmbeds = map[string]sessionEmbedState{
		session: {
			byHash:    map[uint64][]llama.MtmdChunk{hash: val},
			updatedAt: time.Now().Add(-sessionEmbedTTL - time.Minute),
		},
	}
	cache.sessionEmbedLRU = []string{session}
	_, ok := cache.findSessionEmbedLocked(session, hash)
	cache.mu.Unlock()
	if ok {
		t.Fatal("expected TTL expiry to evict session embeds")
	}
	if _, exists := cache.sessionEmbeds[session]; exists {
		t.Fatal("stale session should be removed from overlay map")
	}
}

func TestSessionEmbedOverlay_lruEvictionAtCap(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 2)}
	cache.sessionEmbeds = make(map[string]sessionEmbedState)

	for i := range sessionEmbedMaxSessions + 1 {
		key := fmt.Sprintf("session-%d", i)
		cache.mu.Lock()
		cache.storeSessionEmbedLocked(key, uint64(i), []llama.MtmdChunk{{Embed: []float32{float32(i)}}})
		cache.mu.Unlock()
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()
	if len(cache.sessionEmbeds) != sessionEmbedMaxSessions {
		t.Fatalf("sessions=%d want %d", len(cache.sessionEmbeds), sessionEmbedMaxSessions)
	}
	if _, ok := cache.sessionEmbeds["session-0"]; ok {
		t.Fatal("oldest session should be evicted at cap")
	}
	if _, ok := cache.sessionEmbeds[fmt.Sprintf("session-%d", sessionEmbedMaxSessions)]; !ok {
		t.Fatal("newest session should remain after cap eviction")
	}
}
