package llm

import "testing"

func TestKVCacheBytesShrinksQuantized(t *testing.T) {
	f16 := kvCacheBytes(32, "f16")
	q8 := kvCacheBytes(32, "q8_0")
	q4 := kvCacheBytes(32, "q4_0")
	if f16 != 64 || q8 != 34 || q4 != 18 {
		t.Fatalf("f16=%d q8=%d q4=%d", f16, q8, q4)
	}
	if kvCacheBytes(32, "") != f16 || kvCacheBytes(32, "mystery") != f16 {
		t.Fatal("unknown types stay at f16")
	}
}
