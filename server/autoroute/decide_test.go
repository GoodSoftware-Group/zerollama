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

func TestDecideSpeechAndMusicBackends(t *testing.T) {
	speech := []model.Capability{model.CapabilitySpeech}
	piper := cand("piper", 1, speech)
	piper.Backends = map[string]string{model.ModalitySpeech: model.BackendPiper}
	piper.Voices = []string{"amy", "ryan"}
	kokoro := cand("kokoro", 2, speech)
	kokoro.Backends = map[string]string{model.ModalitySpeech: model.BackendRemoteTTS} // no voice list ⇒ any voice
	music := cand("music", 3, speech)
	music.Backends = map[string]string{model.ModalitySpeech: model.BackendMusic3}
	all := []Candidate{piper, kokoro, music}

	d, err := Decide(all, Signals{Surface: SurfaceMusic}, allCold)
	if err != nil || d.Model != "music" {
		t.Fatalf("music: got %q err %v", d.Model, err)
	}

	d, err = Decide(all, Signals{Surface: SurfaceSpeech, Voice: "ryan"}, warmOnly("kokoro"))
	if err != nil || d.Model != "kokoro" { // warm wins; both accept ryan
		t.Fatalf("speech warm: got %q err %v", d.Model, err)
	}
	if !strings.Contains(rejectedReason(d, "music"), "music3") {
		t.Fatalf("music reason = %q", rejectedReason(d, "music"))
	}

	d, _ = Decide([]Candidate{piper, kokoro}, Signals{Surface: SurfaceSpeech, Voice: "nova"}, warmOnly("piper"))
	if d.Model != "kokoro" || !strings.Contains(rejectedReason(d, "piper"), `voice "nova"`) {
		t.Fatalf("voice filter: got %q, rejected %+v", d.Model, d.Rejected)
	}

	// Soft: nobody lists the voice ⇒ keep everyone.
	d, err = Decide([]Candidate{piper}, Signals{Surface: SurfaceSpeech, Voice: "nova"}, allCold)
	if err != nil || d.Model != "piper" {
		t.Fatalf("soft voice: got %q err %v", d.Model, err)
	}
}

func TestDecideTranscribe(t *testing.T) {
	whisper := cand("whisper", 1, []model.Capability{model.CapabilityAudio})
	whisper.Backends = map[string]string{model.ModalityTranscribe: model.BackendWhisper}
	audioChat := cand("omni", 9, []model.Capability{model.CapabilityCompletion, model.CapabilityAudio})
	plain := cand("plain", 5, capChat)

	d, err := Decide([]Candidate{whisper, audioChat, plain}, Signals{Surface: SurfaceTranscribe, HasAudio: true}, warmOnly("whisper"))
	if err != nil || d.Model != "whisper" {
		t.Fatalf("got %q err %v", d.Model, err)
	}
	if !strings.Contains(rejectedReason(d, "plain"), "whisper") {
		t.Fatalf("plain reason = %q", rejectedReason(d, "plain"))
	}
	d, _ = Decide([]Candidate{audioChat, plain}, Signals{Surface: SurfaceTranscribe, HasAudio: true}, allCold)
	if d.Model != "omni" {
		t.Fatalf("audio chat fallback: got %q", d.Model)
	}
}

func TestDecidePinnedEmbedRefusesAmbiguity(t *testing.T) {
	emb := []model.Capability{model.CapabilityEmbedding}
	laya := cand("laya", 1, []model.Capability{model.CapabilityEmbedding, model.CapabilityDecision})

	d, err := Decide([]Candidate{cand("nomic", 1, emb), laya, cand("qwen", 9, capChat)}, Signals{Surface: SurfaceEmbed}, allCold)
	if err != nil || d.Model != "nomic" {
		t.Fatalf("single embedder: got %q err %v", d.Model, err)
	}
	if !strings.Contains(rejectedReason(d, "laya"), "decision") {
		t.Fatalf("laya reason = %q", rejectedReason(d, "laya"))
	}

	_, err = Decide([]Candidate{cand("nomic", 1, emb), cand("bge", 1, emb)}, Signals{Surface: SurfaceEmbed}, warmOnly("bge"))
	if !errors.Is(err, ErrAmbiguousPinned) || !strings.Contains(err.Error(), "bge, nomic") {
		t.Fatalf("err = %v, want ErrAmbiguousPinned listing both", err)
	}

	// Cooldown must not hide a second eligible embedder — that would silently switch
	// embedding spaces when the other model becomes available again.
	oneCooling := func(c Candidate) Availability {
		if c.Name == "bge" {
			return Availability{CanLoad: false, Detail: "load cooldown"}
		}
		return Availability{CanLoad: true}
	}
	_, err = Decide([]Candidate{cand("nomic", 1, emb), cand("bge", 1, emb)}, Signals{Surface: SurfaceEmbed}, oneCooling)
	if !errors.Is(err, ErrAmbiguousPinned) {
		t.Fatalf("ambiguous before availability: err = %v", err)
	}
}

func TestDecideRerankRequiresRankPooling(t *testing.T) {
	emb := cand("nomic", 1, []model.Capability{model.CapabilityEmbedding})
	rerank := cand("bge-reranker", 1, []model.Capability{model.CapabilityEmbedding})
	rerank.CanRerank = true
	_, err := Decide([]Candidate{emb}, Signals{Surface: SurfaceRerank}, allCold)
	if !errors.Is(err, ErrNoCandidate) {
		t.Fatalf("ordinary embedder: err = %v", err)
	}
	d, err := Decide([]Candidate{emb, rerank}, Signals{Surface: SurfaceRerank}, allCold)
	if err != nil || d.Model != "bge-reranker" {
		t.Fatalf("got %q err %v", d.Model, err)
	}
}

func TestDecideCheckAndBonusHooks(t *testing.T) {
	dec := []model.Capability{model.CapabilityDecision}
	laya, gd, clef := cand("laya", 1, dec), cand("gliner-decide", 1, dec), cand("clef", 1, dec)

	sig := Signals{
		Surface: SurfaceDecisions,
		Bonus: func(c Candidate) (float64, string) {
			if c.Name == "laya" {
				return 0.5, "calibrated"
			}
			return 0, ""
		},
	}
	d, err := Decide([]Candidate{clef, gd, laya}, sig, allCold)
	if err != nil || d.Model != "laya" {
		t.Fatalf("bonus: got %q err %v", d.Model, err)
	}
	if !slicesContains(d.Ranked[0].Reasons, "calibrated") {
		t.Fatalf("reasons = %v", d.Ranked[0].Reasons)
	}

	sig.Check = func(c Candidate) string {
		if c.Name != "clef" {
			return "images need a multimodal (clef) decision model"
		}
		return ""
	}
	d, err = Decide([]Candidate{clef, gd, laya}, sig, allCold)
	if err != nil || d.Model != "clef" || rejectedReason(d, "laya") == "" {
		t.Fatalf("check: got %q err %v rejected %+v", d.Model, err, d.Rejected)
	}
}

func slicesContains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
