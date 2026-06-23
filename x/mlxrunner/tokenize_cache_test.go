package mlxrunner

import (
	"testing"
	"time"
)

func TestTokenizeCacheHitMiss(t *testing.T) {
	var c tokenizeCache
	prompt := "repeat me " + string(make([]byte, 1000))

	tokens, ok := c.lookup(prompt)
	if ok || tokens != nil {
		t.Fatal("expected miss on empty cache")
	}

	c.remember(prompt, []int{1, 2, 3})
	got, ok := c.lookup(prompt)
	if !ok || len(got) != 3 || got[0] != 1 {
		t.Fatalf("lookup = %v ok=%v", got, ok)
	}

	// Mutating returned slice must not corrupt cache.
	got[0] = 99
	got2, ok := c.lookup(prompt)
	if !ok || got2[0] != 1 {
		t.Fatalf("cache corrupted after caller mutation: %v", got2)
	}
}

func TestTokenizeCacheEvictsByEntryCount(t *testing.T) {
	var c tokenizeCache
	for i := range tokenizeCacheMaxEntries + 2 {
		p := string(rune('a' + i%26))
		c.remember(p, []int{i})
		time.Sleep(time.Millisecond)
	}
	if len(c.entries) > tokenizeCacheMaxEntries {
		t.Fatalf("entries = %d want <= %d", len(c.entries), tokenizeCacheMaxEntries)
	}
}

func TestTokenizeCacheLengthGuard(t *testing.T) {
	var c tokenizeCache
	c.remember("abc", []int{1, 2})
	if _, ok := c.lookup("abcd"); ok {
		t.Fatal("expected miss for same hash bucket with different length")
	}
}
