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

func TestVisionEmbedCache_byteBudgetEviction(t *testing.T) {
	// 4 float32s = 16 bytes; budget holds one entry + forces eviction before second.
	cache := VisionEmbedCache{
		entries:    make([]imageEmbedCache, 4),
		byteBudget: 20,
	}
	a := testCachedEmbed(1, 2, 3, 4)
	b := testCachedEmbed(5, 6, 7, 8)

	cache.mu.Lock()
	cache.addGlobalLocked(0x1, a)
	if cache.totalBytes != 16 {
		t.Fatalf("totalBytes=%d want 16", cache.totalBytes)
	}
	cache.addGlobalLocked(0x2, b)
	_, errA := cache.findGlobalLocked(0x1)
	gotB, errB := cache.findGlobalLocked(0x2)
	cache.mu.Unlock()

	if errA != errVisionEmbedNotFound {
		t.Fatalf("expected A evicted under byte budget, err=%v", errA)
	}
	if errB != nil || !reflect.DeepEqual(gotB, b) {
		t.Fatalf("B should remain: err=%v got=%v", errB, gotB)
	}
	if cache.totalBytes != 16 {
		t.Fatalf("totalBytes=%d want 16 after eviction", cache.totalBytes)
	}
}

func TestVisionEmbedCache_sessionHashCap(t *testing.T) {
	cache := VisionEmbedCache{
		entries:          make([]imageEmbedCache, 2),
		sessionMaxHashes: 2,
		sessionEmbeds:    make(map[string]sessionEmbedState),
	}
	const session = "agent-1"

	cache.mu.Lock()
	cache.storeSessionEmbedLocked(session, 1, testCachedEmbed(1))
	cache.storeSessionEmbedLocked(session, 2, testCachedEmbed(2))
	cache.storeSessionEmbedLocked(session, 3, testCachedEmbed(3))
	_, ok1 := cache.findSessionEmbedLocked(session, 1)
	got2, ok2 := cache.findSessionEmbedLocked(session, 2)
	got3, ok3 := cache.findSessionEmbedLocked(session, 3)
	cache.mu.Unlock()

	if ok1 {
		t.Fatal("hash 1 should be evicted at session hash cap")
	}
	if !ok2 || !reflect.DeepEqual(got2, testCachedEmbed(2)) {
		t.Fatalf("hash 2 miss: ok=%v got=%v", ok2, got2)
	}
	if !ok3 || !reflect.DeepEqual(got3, testCachedEmbed(3)) {
		t.Fatalf("hash 3 miss: ok=%v got=%v", ok3, got3)
	}
}

func TestVisionEmbedCache_radixGrowsBeyondSlotsUnderBudget(t *testing.T) {
	// Slot-only LRU of size 2 would thrash; byte budget lets the pool grow.
	cache := VisionEmbedCache{
		entries:    make([]imageEmbedCache, 2),
		byteBudget: 1024,
	}
	cache.mu.Lock()
	for i := 0; i < 6; i++ {
		cache.addGlobalLocked(uint64(0x100+i), testCachedEmbed(float32(i), float32(i)+0.1))
	}
	n := len(cache.entries)
	_, err0 := cache.findGlobalLocked(0x100)
	_, err5 := cache.findGlobalLocked(0x105)
	cache.mu.Unlock()

	if n < 6 {
		t.Fatalf("slots=%d want >=6 under radix byte budget", n)
	}
	if err0 != nil || err5 != nil {
		t.Fatalf("expected all embeds retained: err0=%v err5=%v", err0, err5)
	}
}

func TestVisionEmbedCache_radixCrossSessionHit(t *testing.T) {
	// Success criterion: different prompt_cache_keys share embeds via content pool
	// even when the slot-only MAX would have evicted them.
	cache := VisionEmbedCache{
		entries:          make([]imageEmbedCache, 2),
		byteBudget:       4096,
		sessionMaxHashes: 4,
		sessionEmbeds:    make(map[string]sessionEmbedState),
	}
	shared := testCachedEmbed(1, 2, 3, 4)
	const hash = uint64(0xfeed)

	cache.mu.Lock()
	cache.addGlobalLocked(hash, shared)
	cache.storeSessionEmbedLocked("agent-a", hash, shared)
	// Fill more distinct hashes than initial slots — radix grows, keeps shared.
	for i := 0; i < 8; i++ {
		cache.addGlobalLocked(uint64(0x200+i), testCachedEmbed(float32(i)))
	}
	_, okA := cache.findSessionEmbedLocked("agent-a", hash)
	got, err := cache.findGlobalLocked(hash)
	_, okB := cache.findSessionEmbedLocked("agent-b", hash)
	cache.mu.Unlock()

	if !okA {
		t.Fatal("session-a overlay should still pin")
	}
	if err != nil {
		t.Fatalf("radix pool should keep shared hash across fills: %v", err)
	}
	if okB {
		t.Fatal("session-b must not see session-a overlay pins")
	}
	if !reflect.DeepEqual(got, shared) {
		t.Fatalf("cross-session radix value mismatch: %v", got)
	}

	// agent-b gets the same embed by content hash (global/radix), then may pin.
	cache.mu.Lock()
	cache.storeSessionEmbedLocked("agent-b", hash, got)
	_, okB2 := cache.findSessionEmbedLocked("agent-b", hash)
	cache.mu.Unlock()
	if !okB2 {
		t.Fatal("agent-b should pin after radix restore")
	}
}

func TestVisionEmbedCache_noRadixStillEvictsBySlots(t *testing.T) {
	cache := VisionEmbedCache{entries: make([]imageEmbedCache, 2)} // byteBudget 0
	cache.mu.Lock()
	cache.addGlobalLocked(1, testCachedEmbed(1))
	cache.addGlobalLocked(2, testCachedEmbed(2))
	cache.addGlobalLocked(3, testCachedEmbed(3))
	_, err1 := cache.findGlobalLocked(1)
	cache.mu.Unlock()
	if err1 != errVisionEmbedNotFound {
		t.Fatalf("slot-only mode should evict oldest, err=%v", err1)
	}
}
