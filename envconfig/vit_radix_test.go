package envconfig

import (
	"testing"
)

func TestEffectiveImageEmbedCacheBytes_radixDefault(t *testing.T) {
	t.Setenv("OLLAMA_IMAGE_EMBED_CACHE_BYTES", "")
	t.Setenv("OLLAMA_VIT_RADIX", "1")
	if got := EffectiveImageEmbedCacheBytes(); got != DefaultVitRadixBytes {
		t.Fatalf("got %d want %d", got, DefaultVitRadixBytes)
	}
}

func TestEffectiveImageEmbedCacheBytes_explicitWins(t *testing.T) {
	t.Setenv("OLLAMA_VIT_RADIX", "1")
	t.Setenv("OLLAMA_IMAGE_EMBED_CACHE_BYTES", "1048576")
	if got := EffectiveImageEmbedCacheBytes(); got != 1048576 {
		t.Fatalf("got %d want 1048576", got)
	}
}

func TestEffectiveImageEmbedCacheBytes_radixOff(t *testing.T) {
	t.Setenv("OLLAMA_VIT_RADIX", "0")
	t.Setenv("OLLAMA_IMAGE_EMBED_CACHE_BYTES", "")
	if got := EffectiveImageEmbedCacheBytes(); got != 0 {
		t.Fatalf("got %d want 0", got)
	}
}

func TestKVMMPadRadixEnabled_defaultOn(t *testing.T) {
	t.Setenv("OLLAMA_KV_MM_PAD_RADIX", "")
	if !KVMMPadRadixEnabled() {
		t.Fatal("expected default on")
	}
	t.Setenv("OLLAMA_KV_MM_PAD_RADIX", "0")
	if KVMMPadRadixEnabled() {
		t.Fatal("expected off when 0")
	}
}
