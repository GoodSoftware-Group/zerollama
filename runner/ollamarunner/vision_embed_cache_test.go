package ollamarunner

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

func testCachedEmbed(vals ...float32) cachedMultimodal {
	return cachedMultimodal{
		parts: []cachedMultimodalPart{{
			floats: append([]float32(nil), vals...),
		}},
	}
}

func TestVisionEmbedCache_lru(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, defaultImageCacheSize)}

	valA := testCachedEmbed(0.1, 0.2)
	valB := testCachedEmbed(0.4, 0.5, 0.6)

	cache.mu.Lock()
	_, err := cache.findGlobalLocked(0x5adb61d31933a946)
	cache.mu.Unlock()
	if err != errVisionEmbedNotFound {
		t.Fatalf("expected miss in empty cache: %v", err)
	}

	cache.mu.Lock()
	cache.addGlobalLocked(0x5adb61d31933a946, valA)
	got, err := cache.findGlobalLocked(0x5adb61d31933a946)
	cache.mu.Unlock()
	if err != nil || !reflect.DeepEqual(got, valA) {
		t.Fatalf("find A: got=%v err=%v", got, err)
	}

	cache.mu.Lock()
	cache.addGlobalLocked(0x011551369a34a901, valB)
	got, err = cache.findGlobalLocked(0x011551369a34a901)
	cache.mu.Unlock()
	if err != nil || !reflect.DeepEqual(got, valB) {
		t.Fatalf("find B: got=%v err=%v", got, err)
	}

	cache.mu.Lock()
	cache.addGlobalLocked(0x756b218a517e7353, valB)
	cache.addGlobalLocked(0x75e5e8d35d7e3967, testCachedEmbed(0.8))
	cache.addGlobalLocked(0xd96f7f268ca0646e, testCachedEmbed(0.9))
	_, err = cache.findGlobalLocked(0x5adb61d31933a946)
	cache.mu.Unlock()
	if err != errVisionEmbedNotFound {
		t.Fatalf("expected eviction of A, got err=%v", err)
	}
}

func TestVisionEmbedCache_growForVideoAgents(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 4)}
	cache.growCacheForDistinctFrames(8)
	if len(cache.entries) != 8 {
		t.Fatalf("len=%d want 8", len(cache.entries))
	}
}

func TestVisionEmbedCache_sessionOverlay_survivesGlobalEviction(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 2)}
	const session = "agent-thread-1"
	hash := uint64(0xabc123)
	val := testCachedEmbed(1.0, 2.0)

	cache.mu.Lock()
	cache.storeSessionEmbedLocked(session, hash, val)
	cache.addGlobalLocked(uint64(0x1), testCachedEmbed(9))
	cache.addGlobalLocked(uint64(0x2), testCachedEmbed(8))
	cache.addGlobalLocked(uint64(0x3), testCachedEmbed(7))
	_, err := cache.findGlobalLocked(hash)
	got, ok := cache.findSessionEmbedLocked(session, hash)
	cache.mu.Unlock()

	if err != errVisionEmbedNotFound {
		t.Fatalf("expected global miss, err=%v", err)
	}
	if !ok || !reflect.DeepEqual(got, val) {
		t.Fatalf("session overlay miss: ok=%v got=%v", ok, got)
	}
}

func TestVisionEmbedCache_sessionOverlay_noCrossSession(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 2)}
	hash := uint64(0xdef456)
	val := testCachedEmbed(3.0)

	cache.mu.Lock()
	cache.storeSessionEmbedLocked("session-a", hash, val)
	_, ok := cache.findSessionEmbedLocked("session-b", hash)
	cache.mu.Unlock()
	if ok {
		t.Fatal("session-b should not see session-a embeds")
	}
}

func TestVisionEmbedCache_sessionOverlay_ttlEviction(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 2)}
	const session = "stale-session"
	hash := uint64(0xdead)
	val := testCachedEmbed(5.0)

	cache.mu.Lock()
	cache.sessionEmbeds = map[string]sessionEmbedState{
		session: {
			byHash:    map[uint64]cachedMultimodal{hash: val},
			updatedAt: time.Now().Add(-sessionEmbedTTL - time.Minute),
		},
	}
	cache.sessionEmbedLRU = []string{session}
	_, ok := cache.findSessionEmbedLocked(session, hash)
	cache.mu.Unlock()
	if ok {
		t.Fatal("expected TTL expiry")
	}
	if _, exists := cache.sessionEmbeds[session]; exists {
		t.Fatal("stale session should be removed")
	}
}

func TestVisionEmbedCache_sessionOverlay_lruEvictionAtCap(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 2)}
	cache.sessionEmbeds = make(map[string]sessionEmbedState)

	for i := range sessionEmbedMaxSessions + 1 {
		key := fmt.Sprintf("session-%d", i)
		cache.mu.Lock()
		cache.storeSessionEmbedLocked(key, uint64(i), testCachedEmbed(float32(i)))
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
}
