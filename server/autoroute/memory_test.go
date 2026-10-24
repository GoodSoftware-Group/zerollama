package autoroute

import (
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
)

func TestMemoryTTLSlidesAndExpires(t *testing.T) {
	now := time.Date(2026, 10, 9, 0, 0, 0, 0, time.UTC)
	m := NewMemory(10*time.Minute, 8)
	m.now = func() time.Time { return now }

	m.Put("conv", "qwen")
	now = now.Add(9 * time.Minute)
	if got, ok := m.Get("conv"); !ok || got != "qwen" {
		t.Fatalf("within ttl: %q %v", got, ok)
	}
	now = now.Add(9 * time.Minute) // 18m after put, 9m after last use ⇒ still live (sliding)
	if _, ok := m.Get("conv"); !ok {
		t.Fatal("sliding expiry should keep a used key")
	}
	now = now.Add(11 * time.Minute)
	if _, ok := m.Get("conv"); ok {
		t.Fatal("expired key returned")
	}
	if m.Len() != 0 {
		t.Fatalf("expired key not swept on read: len %d", m.Len())
	}
}

func TestMemoryPeekDoesNotRefreshTTL(t *testing.T) {
	now := time.Date(2026, 10, 9, 0, 0, 0, 0, time.UTC)
	m := NewMemory(10*time.Minute, 8)
	m.now = func() time.Time { return now }

	m.Put("conv", "qwen")
	now = now.Add(9 * time.Minute)
	if got, ok := m.Peek("conv"); !ok || got != "qwen" {
		t.Fatalf("peek within ttl: %q %v", got, ok)
	}
	now = now.Add(2 * time.Minute) // 11m after put; Get would have slid, Peek must not
	if _, ok := m.Peek("conv"); ok {
		t.Fatal("peek must not refresh TTL")
	}
}

func TestMemoryLRUAndForget(t *testing.T) {
	m := NewMemory(time.Hour, 2)
	m.Put("a", "1")
	m.Put("b", "2")
	m.Get("a") // a is now most recent
	m.Put("c", "3")
	if _, ok := m.Get("b"); ok {
		t.Fatal("least recently used key should be evicted")
	}
	if _, ok := m.Get("a"); !ok {
		t.Fatal("recently used key evicted")
	}
	m.Forget("a")
	if _, ok := m.Get("a"); ok {
		t.Fatal("forgotten key returned")
	}
	m.Put("c", "4")
	if got, _ := m.Get("c"); got != "4" {
		t.Fatalf("overwrite: %q", got)
	}

	var nilMem *Memory
	nilMem.Put("x", "y")
	if _, ok := nilMem.Get("x"); ok {
		t.Fatal("nil memory must be a no-op")
	}
	off := NewMemory(0, 4)
	off.Put("x", "y")
	if _, ok := off.Get("x"); ok {
		t.Fatal("ttl 0 disables remembering")
	}
}

func TestValidateIsStrict(t *testing.T) {
	c := cand("small", 1, capChat)
	c.ContextLength = 4096

	if _, ok := Validate(c, Signals{Surface: SurfaceChat, PromptTokens: 1000}, allCold); !ok {
		t.Fatal("fitting candidate rejected")
	}
	if reason, ok := Validate(c, Signals{Surface: SurfaceChat, PromptTokens: 9000}, allCold); ok || !strings.Contains(reason, "context") {
		t.Fatalf("outgrown context must fail strictly: %q %v", reason, ok)
	}
	if reason, ok := Validate(c, Signals{Surface: SurfaceChat, HasImages: true}, allCold); ok || !strings.Contains(reason, string(model.CapabilityVision)) {
		t.Fatalf("new capability need: %q %v", reason, ok)
	}
	cooling := func(Candidate) Availability { return Availability{Detail: "load cooldown"} }
	if reason, ok := Validate(c, Signals{Surface: SurfaceChat}, cooling); ok || !strings.Contains(reason, "cooldown") {
		t.Fatalf("unavailable: %q %v", reason, ok)
	}
	piper := cand("piper", 1, []model.Capability{model.CapabilitySpeech})
	piper.Voices = []string{"amy"}
	if _, ok := Validate(piper, Signals{Surface: SurfaceSpeech, Voice: "ryan"}, allCold); ok {
		t.Fatal("voice mismatch must fail validation (strict, unlike Decide's soft filter)")
	}
}
