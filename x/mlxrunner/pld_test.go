package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/sample"
)

func TestPLDFindMatchLatestSite(t *testing.T) {
	committed := []int32{0, 1, 2, 3, 1, 2, 4, 5, 6}
	got := pldFindMatch(committed, []int32{1, 2}, 3)
	want := []int32{4, 5, 6}
	if !equalTokens(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestPLDFindMatchMissing(t *testing.T) {
	if pldFindMatch([]int32{0, 1, 2, 3, 4, 5, 6, 7}, []int32{99, 100}, 3) != nil {
		t.Fatal("expected nil")
	}
}

func TestPLDFindMatchClips(t *testing.T) {
	committed := []int32{9, 8, 1, 2, 7, 1, 2}
	got := pldFindMatch(committed, []int32{1, 2}, 5)
	want := []int32{7, 1, 2}
	if !equalTokens(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestPLDFindMatchPrefersLatest(t *testing.T) {
	committed := []int32{1, 2, 100, 200, 1, 2, 50, 60, 1, 2}
	got := pldFindMatch(committed, []int32{1, 2}, 2)
	want := []int32{50, 60}
	if !equalTokens(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestPLDFindMatchRejectsEmpty(t *testing.T) {
	if pldFindMatch([]int32{1, 2, 3}, nil, 3) != nil {
		t.Fatal("empty key")
	}
	if pldFindMatch([]int32{1, 2, 3, 1, 2, 4}, []int32{1, 2}, 0) != nil {
		t.Fatal("zero max")
	}
}

func TestNgramRepeatScoreRepetitive(t *testing.T) {
	tokens := []int32{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}
	if s := ngramRepeatScore(tokens, 3); s != 1 {
		t.Fatalf("score=%v want 1", s)
	}
}

func TestNgramRepeatScoreNovel(t *testing.T) {
	tokens := make([]int32, 30)
	for i := range tokens {
		tokens[i] = int32(i + 1000)
	}
	if s := ngramRepeatScore(tokens, 3); s >= 0.05 {
		t.Fatalf("score=%v want <0.05", s)
	}
}

func TestNgramRepeatScoreMixed(t *testing.T) {
	tokens := make([]int32, 20)
	for i := 0; i < 10; i++ {
		tokens[i] = int32((i % 3) + 1)
	}
	for i := 10; i < 20; i++ {
		tokens[i] = int32(i + 4990)
	}
	s := ngramRepeatScore(tokens, 3)
	if s <= 0.10 || s >= 0.55 {
		t.Fatalf("score=%v want (0.10, 0.55)", s)
	}
}

func TestNgramRepeatScoreShort(t *testing.T) {
	if ngramRepeatScore([]int32{1, 2}, 3) != 0 {
		t.Fatal("short")
	}
	if ngramRepeatScore([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 9) != 0 {
		t.Fatal("ngram too large")
	}
}

func TestTailMatchFractionEcho(t *testing.T) {
	committed := make([]int32, 64)
	for i := 0; i < 32; i++ {
		committed[i] = int32(i + 100)
	}
	for i := 32; i < 48; i++ {
		committed[i] = int32(i + 8968)
	}
	for i := 48; i < 64; i++ {
		committed[i] = int32(i - 48 + 100)
	}
	if f := tailMatchFraction(committed, 16, 3); f <= 0.7 {
		t.Fatalf("frac=%v want >0.7", f)
	}
}

func TestTailMatchFractionNovel(t *testing.T) {
	committed := make([]int32, 64)
	for i := range committed {
		committed[i] = int32(i*13 + 7)
	}
	if f := tailMatchFraction(committed, 16, 3); f != 0 {
		t.Fatalf("frac=%v want 0", f)
	}
}

func TestTailMatchFractionDegenerate(t *testing.T) {
	if tailMatchFraction([]int32{1, 2}, 16, 3) != 0 {
		t.Fatal("short")
	}
	if tailMatchFraction([]int32{1, 2, 3, 4, 5}, 0, 3) != 0 {
		t.Fatal("zero window")
	}
}

func TestPLDPromptGate(t *testing.T) {
	novel := make([]int32, 40)
	for i := range novel {
		novel[i] = int32(i + 50)
	}
	if ngramRepeatScore(novel, 3) >= pldSpecGateThresh {
		t.Fatal("novel prompt should sit under mlx-serve 0.01 gate")
	}
	echo := make([]int32, 0, 60)
	block := []int32{10, 11, 12, 13, 14}
	for range 12 {
		echo = append(echo, block...)
	}
	if ngramRepeatScore(echo, 3) < pldSpecGateThresh {
		t.Fatal("echo prompt should pass the gate")
	}
}

func TestNewSpeculationPLDOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "off")
	if newSpeculation(&Runner{}, nil) != nil {
		t.Fatal("PLD off and no draft head must leave speculation nil")
	}
}

func TestNewSpeculationPLDDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	s := newSpeculation(&Runner{}, nil)
	if s == nil || s.draft != nil || s.drafter != nil {
		t.Fatalf("want PLD-only speculation, got %+v", s)
	}
	if s.depth.scheduled != pldDraftLen {
		t.Fatalf("scheduled=%d want %d", s.depth.scheduled, pldDraftLen)
	}
}

type parkSpecModel struct{ fakeMTPModel }

func (parkSpecModel) ParkSpeculation() string { return "test: park PLD" }

func TestNewSpeculationParkedByModel(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	r := &Runner{Model: &parkSpecModel{}}
	if newSpeculation(r, nil) != nil {
		t.Fatal("ParkSpeculation must disable default PLD")
	}
	if newSpeculation(r, &fakeMTPDraft{}) != nil {
		t.Fatal("ParkSpeculation must disable MTP draft too")
	}
}

func TestPLDOpenGatesNovelPrompt(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	s := newSpeculation(&Runner{}, nil)
	tokens := make([]int32, 40)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	spec := s.open(Request{Tokens: tokens}, nil)
	if spec == nil {
		t.Fatal("novel prompt still opens a PLD session (parked, may re-enable)")
	}
	p, ok := spec.drafter.(*pldDraftSession)
	if !ok || !p.skip {
		t.Fatalf("want parked PLD, got %T skip=%v", spec.drafter, ok && p.skip)
	}
}

func TestPLDOpenRespectsEnablePLDFalse(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	s := newSpeculation(&Runner{}, nil)
	off := false
	spec := s.open(Request{Tokens: []int32{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, CompletionRequest: CompletionRequest{EnablePLD: &off}}, nil)
	if spec != nil {
		t.Fatalf("enable_pld=false must skip PLD, got %+v", spec)
	}
}

func TestPLDOpenEchoPrompt(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	s := newSpeculation(&Runner{}, nil)
	echo := make([]int32, 0, 60)
	block := []int32{10, 11, 12, 13, 14}
	for range 12 {
		echo = append(echo, block...)
	}
	spec := s.open(Request{Tokens: echo}, nil)
	if spec == nil || !spec.pld || !spec.enabled {
		t.Fatalf("repetitive prompt should auto-enable PLD, got %+v", spec)
	}
}

func TestRuntimeGatePLDSticky(t *testing.T) {
	s := &speculationSession{
		pld:     true,
		enabled: true,
		spec:    &speculation{depth: newDepthController()},
	}
	for range pldRuntimeMinRounds {
		s.endRound(5, 0, 5)
	}
	if s.enabled {
		t.Fatal("zero acceptance after 5 PLD rounds must stick off")
	}
	if s.limit != 0 {
		t.Fatalf("limit=%d want 0", s.limit)
	}
}

type stubDrafter struct {
	hit bool
	n   int
}

func (s *stubDrafter) propose(*mlx.Array, int) *draftCandidates {
	s.n++
	if !s.hit {
		return nil
	}
	return &draftCandidates{}
}
func (s *stubDrafter) committed(_, _ *mlx.Array, _ int) {}
func (s *stubDrafter) settle(*mlx.Array)                {}
func (s *stubDrafter) close()                           {}

func TestStackedPrefersPLD(t *testing.T) {
	pld, mtp := &stubDrafter{hit: true}, &stubDrafter{hit: true}
	st := &stackedDrafter{pld: pld, mtp: mtp}
	if st.propose(nil, 5) == nil {
		t.Fatal("want PLD hit")
	}
	if pld.n != 1 || mtp.n != 0 {
		t.Fatalf("pld=%d mtp=%d", pld.n, mtp.n)
	}
}

func TestStackedFallsBackMTP(t *testing.T) {
	pld, mtp := &stubDrafter{hit: false}, &stubDrafter{hit: true}
	st := &stackedDrafter{pld: pld, mtp: mtp}
	if st.propose(nil, 5) == nil {
		t.Fatal("want MTP fallback")
	}
	if pld.n != 1 || mtp.n != 1 {
		t.Fatalf("pld=%d mtp=%d", pld.n, mtp.n)
	}
}

func TestStackedPLDRuntimeGateFallsBack(t *testing.T) {
	pld, mtp := &stubDrafter{hit: true}, &stubDrafter{hit: true}
	st := &stackedDrafter{pld: pld, mtp: mtp}
	sess := &speculationSession{
		drafter: st,
		enabled: true,
		spec:    &speculation{depth: newDepthController()},
	}
	for range pldRuntimeMinRounds {
		if st.propose(nil, 5) == nil {
			t.Fatal("PLD should hit during warmup")
		}
		sess.endRound(5, 0, 5)
	}
	if !st.skipPLD || !sess.enabled {
		t.Fatalf("want skipPLD with MTP still enabled, skip=%v enabled=%v", st.skipPLD, sess.enabled)
	}
	if st.propose(nil, 5) == nil {
		t.Fatal("want MTP after PLD skip")
	}
	if pld.n != pldRuntimeMinRounds || mtp.n != 1 {
		t.Fatalf("pld=%d mtp=%d", pld.n, mtp.n)
	}
}

func TestOpenEchoWithMTPStacks(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	s := newSpeculation(&Runner{}, &fakeMTPDraft{})
	echo := make([]int32, 0, 60)
	block := []int32{10, 11, 12, 13, 14}
	for range 12 {
		echo = append(echo, block...)
	}
	spec := s.open(Request{Tokens: echo}, nil)
	if spec == nil {
		t.Fatal("want speculation session")
	}
	if _, ok := spec.drafter.(*stackedDrafter); !ok {
		t.Fatalf("want stacked PLD+MTP, got %T", spec.drafter)
	}
}

func TestOpenNovelWithMTPKeepsHead(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	s := newSpeculation(&Runner{}, &fakeMTPDraft{})
	tokens := make([]int32, 40)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	spec := s.open(Request{Tokens: tokens}, nil)
	if spec == nil {
		t.Fatal("want session on novel prompt")
	}
	st, ok := spec.drafter.(*stackedDrafter)
	if !ok {
		t.Fatalf("want stacked (parked PLD + MTP), got %T", spec.drafter)
	}
	p, ok := st.pld.(*pldDraftSession)
	if !ok || !p.skip {
		t.Fatal("PLD should start skipped on a novel prompt")
	}
}

func TestPLDReenableOnEchoTail(t *testing.T) {
	d := newPLDSession(nil)
	d.skip = true
	committed := make([]int32, 64)
	for i := 0; i < 32; i++ {
		committed[i] = int32(i + 100)
	}
	for i := 32; i < 48; i++ {
		committed[i] = int32(i + 8968)
	}
	for i := 48; i < 64; i++ {
		committed[i] = int32(i - 48 + 100)
	}
	d.ids = committed
	d.maybeReenable()
	if d.skip {
		t.Fatal("echo tail should re-enable prompt-gated PLD")
	}
}

func TestConfigSparseMoE(t *testing.T) {
	if configSparseMoE([]byte(`{"architectures":["Qwen3ForCausalLM"]}`)) {
		t.Fatal("dense")
	}
	if !configSparseMoE([]byte(`{"num_experts":16,"architectures":["Qwen3MoeForCausalLM"]}`)) {
		t.Fatal("num_experts")
	}
	if !configSparseMoE([]byte(`{"model_type":"qwen3_5_moe"}`)) {
		t.Fatal("model_type")
	}
	if !configSparseMoE([]byte(`{"text_config":{"num_experts":8}}`)) {
		t.Fatal("nested")
	}
	if configSparseMoE([]byte(`{"num_experts":1}`)) {
		t.Fatal("single expert is dense")
	}
}

func TestMTPRequestedMoEDefault(t *testing.T) {
	on := true
	off := false
	if !mtpRequested(nil, false) {
		t.Fatal("dense default on")
	}
	if mtpRequested(nil, true) {
		t.Fatal("MoE default off")
	}
	if !mtpRequested(&on, true) {
		t.Fatal("explicit on")
	}
	if mtpRequested(&off, false) {
		t.Fatal("explicit off")
	}
}

func TestSpecOpenEnabled(t *testing.T) {
	if !specOpenEnabled(Request{}) {
		t.Fatal("plain request")
	}
	if specOpenEnabled(Request{SamplerOpts: sample.Options{Logprobs: true}}) {
		t.Fatal("logprobs")
	}
	if specOpenEnabled(Request{SamplerOpts: sample.Options{TopLogprobs: 5}}) {
		t.Fatal("top_logprobs")
	}
	if specOpenEnabled(Request{CompletionRequest: CompletionRequest{Format: []byte(`{"type":"json"}`)}}) {
		t.Fatal("format")
	}
	if specOpenEnabled(Request{CompletionRequest: CompletionRequest{Grammar: "root ::= \"a\""}}) {
		t.Fatal("grammar")
	}
}
