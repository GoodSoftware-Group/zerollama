package openai

import (
	"testing"
)

func TestVideoURLFetchCache_hit(t *testing.T) {
	resetVideoURLFetchCache()
	url := "https://cdn.example.com/clip.mp4"
	body := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}
	rememberVideoURLFetchCache(url, body)
	got, ok := lookupVideoURLFetchCache(url)
	if !ok || string(got) != string(body) {
		t.Fatalf("cache miss or body mismatch")
	}
}

func TestVideoURLFetchCache_missDifferentURL(t *testing.T) {
	resetVideoURLFetchCache()
	rememberVideoURLFetchCache("https://a.example/v.mp4", []byte{1})
	if _, ok := lookupVideoURLFetchCache("https://b.example/v.mp4"); ok {
		t.Fatal("expected miss")
	}
}
