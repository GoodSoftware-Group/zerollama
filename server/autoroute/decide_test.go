package autoroute

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

var (
	capChat   = []model.Capability{model.CapabilityCompletion}
	capTools  = []model.Capability{model.CapabilityCompletion, model.CapabilityTools}
	capVision = []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityVideo}
)

func cand(name string, size int64, caps []model.Capability) Candidate {
	return Candidate{Name: name, Digest: "sha256-" + name, SizeBytes: size, Capabilities: caps}
}

func allCold(Candidate) Availability { return Availability{CanLoad: true} }

func warmOnly(names ...string) AvailabilityFunc {
	return func(c Candidate) Availability {
		for _, n := range names {
			if c.Name == n {
				return Availability{Warm: true, CanLoad: true}
			}
		}
		return Availability{CanLoad: true}
	}
}

func rejectedReason(d Decision, name string) string {
	for _, r := range d.Rejected {
		if r.Model == name {
			return r.Reason
		}
	}
	return ""
}

func TestDecideSurfaceExcludesNonChatModels(t *testing.T) {
	cands := []Candidate{
		cand("laya", 10, []model.Capability{model.CapabilityCompletion, model.CapabilityDecision}),
		cand("nomic", 10, []model.Capability{model.CapabilityEmbedding}),
		cand("zimage", 10, []model.Capability{model.CapabilityCompletion, model.CapabilityImage}),
		cand("qwen", 5, capChat),
	}
	d, err := Decide(cands, Signals{Surface: SurfaceChat, Text: "hi"}, allCold)
	if err != nil {
		t.Fatal(err)
	}
	if d.Model != "qwen" || d.Reason != "only-candidate" {
		t.Fatalf("got %q (%s), want qwen only-candidate", d.Model, d.Reason)
	}
	if !strings.Contains(rejectedReason(d, "laya"), "decision") {
		t.Fatalf("laya reason = %q", rejectedReason(d, "laya"))
	}
}

func TestDecideImageSurfaceNeverPicksChatModel(t *testing.T) {
	cands := []Candidate{cand("qwen", 100, capChat), cand("zimage", 5, []model.Capability{model.CapabilityImage})}
	d, err := Decide(cands, Signals{Surface: SurfaceImage, Text: "a cat"}, warmOnly("qwen"))
	if err != nil {
		t.Fatal(err)
	}
	if d.Model != "zimage" {
		t.Fatalf("image surface picked %q", d.Model)
	}
}

func TestDecideRequestCapabilities(t *testing.T) {
	cands := []Candidate{cand("small", 1, capChat), cand("tooly", 2, capTools), cand("vlm", 3, capVision)}
	cases := []struct {
		sig  Signals
		want string
	}{
		{Signals{Surface: SurfaceChat, HasTools: true}, "tooly"},
		{Signals{Surface: SurfaceChat, HasImages: true}, "vlm"},
		{Signals{Surface: SurfaceChat, HasVideo: true}, "vlm"},
	}
	for _, tc := range cases {
		d, err := Decide(cands, tc.sig, allCold)
		if err != nil {
			t.Fatalf("%+v: %v", tc.sig, err)
		}
		if d.Model != tc.want {
			t.Fatalf("%+v: got %q want %q", tc.sig, d.Model, tc.want)
		}
	}

	d, err := Decide(cands, Signals{Surface: SurfaceChat, HasAudio: true}, allCold)
	if !errors.Is(err, ErrNoCandidate) {
		t.Fatalf("audio: err = %v, want ErrNoCandidate", err)
	}
	if !strings.Contains(rejectedReason(d, "vlm"), "audio") {
		t.Fatalf("vlm reason = %q", rejectedReason(d, "vlm"))
	}
}

func TestDecideThinkingRequired(t *testing.T) {
	thinker := cand("thinker", 1, []model.Capability{model.CapabilityCompletion, model.CapabilityThinking})
	plain := cand("plain", 9, capChat)
	d, err := Decide([]Candidate{plain, thinker}, Signals{Surface: SurfaceChat, WantsThink: true}, allCold)
	if err != nil || d.Model != "thinker" {
		t.Fatalf("got %q err %v, want thinker", d.Model, err)
	}
}

func TestDecideContextFit(t *testing.T) {
	short := cand("short", 9, capChat)
	short.ContextLength = 4096
	long := cand("long", 1, capChat)
	long.ContextLength = 131072
	unknown := cand("unknown", 5, capChat) // ContextLength 0 never filters

	d, err := Decide([]Candidate{short, long, unknown}, Signals{Surface: SurfaceChat, PromptTokens: 8000}, allCold)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rejectedReason(d, "short"), "context 4096") {
		t.Fatalf("short reason = %q", rejectedReason(d, "short"))
	}
	if d.Model != "unknown" { // larger of the two survivors
		t.Fatalf("got %q, want unknown", d.Model)
	}

	// Soft filter: nothing fits → keep everyone rather than 503.
	d, err = Decide([]Candidate{short}, Signals{Surface: SurfaceChat, PromptTokens: 500000}, allCold)
	if err != nil || d.Model != "short" {
		t.Fatalf("soft context: got %q err %v", d.Model, err)
	}
}

func TestDecideAvailability(t *testing.T) {
	big := cand("big", 100, capChat)
	small := cand("small", 10, capChat)
	runtime := cand("rt", 50, capChat)

	avail := func(c Candidate) Availability {
		switch c.Name {
		case "big":
			return Availability{CanLoad: false, Detail: "load cooldown"}
		case "rt":
			return Availability{Excluded: "runtime-routed"}
		}
		return Availability{CanLoad: true}
	}
	d, err := Decide([]Candidate{big, small, runtime}, Signals{Surface: SurfaceChat}, avail)
	if err != nil {
		t.Fatal(err)
	}
	if d.Model != "small" {
		t.Fatalf("got %q, want small", d.Model)
	}
	if rejectedReason(d, "big") != "cannot load now: load cooldown" || rejectedReason(d, "rt") != "runtime-routed" {
		t.Fatalf("rejections = %+v", d.Rejected)
	}
}

func TestDecideScoringPrefersWarmThenAvoidsEviction(t *testing.T) {
	big := cand("big", 100, capChat)
	small := cand("small", 10, capChat)

	d, _ := Decide([]Candidate{big, small}, Signals{Surface: SurfaceChat}, warmOnly("small"))
	if d.Model != "small" || d.Reason != "warm" {
		t.Fatalf("warm: got %q (%s)", d.Model, d.Reason)
	}

	d, _ = Decide([]Candidate{big, small}, Signals{Surface: SurfaceChat}, allCold)
	if d.Model != "big" || d.Reason != "best-fit" {
		t.Fatalf("cold: got %q (%s), want bigger model", d.Model, d.Reason)
	}

	evictBig := func(c Candidate) Availability {
		return Availability{CanLoad: true, NeedsEviction: c.Name == "big", Detail: "max_loaded_models"}
	}
	d, _ = Decide([]Candidate{big, small}, Signals{Surface: SurfaceChat}, evictBig)
	if d.Model != "small" {
		t.Fatalf("eviction: got %q, want small", d.Model)
	}
}

func TestDecideBenchSpeedBreaksTies(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	path := filepath.Join(home, ".ollama", "bench.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"sha256-slow":{"kind":"image","gen_sec":60},"sha256-fast":{"kind":"image","gen_sec":6}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	imgCaps := []model.Capability{model.CapabilityImage}
	// Image surface ignores size, so bench speed decides.
	d, err := Decide([]Candidate{cand("slow", 100, imgCaps), cand("fast", 1, imgCaps)}, Signals{Surface: SurfaceImage}, allCold)
	if err != nil || d.Model != "fast" {
		t.Fatalf("got %q err %v, want fast", d.Model, err)
	}
}

func TestDecideNoCandidatesExplains(t *testing.T) {
	d, err := Decide(nil, Signals{Surface: SurfaceChat}, allCold)
	if !errors.Is(err, ErrNoCandidate) || !strings.Contains(err.Error(), "no local models") {
		t.Fatalf("err = %v", err)
	}
	if d.Model != "" {
		t.Fatalf("model = %q", d.Model)
	}
}
