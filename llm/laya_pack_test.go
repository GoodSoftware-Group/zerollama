package llm

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestRenderOptionsChoice(t *testing.T) {
	q := DecisionQuestion{
		Type:         "choice",
		Instructions: "pick one",
		Criteria:     json.RawMessage(`{"alpha":"first","beta":"","gamma":null,"delta":0}`),
	}
	got, err := RenderOptions(q)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"alpha: first", "beta", "gamma", "delta: 0"}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%q want %q", i, got[i], want[i])
		}
	}
}

func TestRenderOptionsScore(t *testing.T) {
	q := DecisionQuestion{
		Type:         "score",
		Instructions: "rate",
		Criteria:     json.RawMessage(`["bad","ok","great"]`),
	}
	got, err := RenderOptions(q)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"level 0: bad", "level 1: ok", "level 2: great"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%q want %q", i, got[i], want[i])
		}
	}
}

func TestRenderOptionsNoul(t *testing.T) {
	q := DecisionQuestion{
		Type:         "noul",
		Instructions: "holds?",
		Criteria:     json.RawMessage(`{}`),
	}
	got, err := RenderOptions(q)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"false: no, the statement does not hold",
		"true: yes, the statement holds",
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%q want %q", i, got[i], want[i])
		}
	}

	q.Labels = map[string]string{"false": "nope", "true": "yep"}
	q.Criteria = json.RawMessage(`{"false":"denied","true":{"reason":"ok"}}`)
	got, err = RenderOptions(q)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got[0], "nope: denied") {
		t.Fatalf("false opt %q", got[0])
	}
	if !strings.HasPrefix(got[1], "yep: ") || !strings.Contains(got[1], "reason") {
		t.Fatalf("true opt %q", got[1])
	}
}

func TestRenderOptionsLabelsOnlyNoul(t *testing.T) {
	_, err := RenderOptions(DecisionQuestion{
		Type:     "choice",
		Criteria: json.RawMessage(`{"a":"x"}`),
		Labels:   map[string]string{"false": "f", "true": "t"},
	})
	if err == nil {
		t.Fatal("expected labels error on choice")
	}
}

func TestConfidenceFromProbs(t *testing.T) {
	if c := ConfidenceFromProbs([]float64{1.0}, 1); c != 1.0 {
		t.Fatalf("k<2 got %v", c)
	}
	// Uniform over 2 → entropy = log(2) → confidence 0.
	if c := ConfidenceFromProbs([]float64{0.5, 0.5}, 2); c > 1e-9 {
		t.Fatalf("uniform confidence %v want ~0", c)
	}
	// Peak → high confidence.
	if c := ConfidenceFromProbs([]float64{0.99, 0.01}, 2); c < 0.9 {
		t.Fatalf("peaked confidence %v", c)
	}
}

func TestTempBucketAndClamp(t *testing.T) {
	if got := TempBucket(0, 4); got != "choice:3-5" {
		t.Fatalf("bucket %q", got)
	}
	if got := TempBucket(2, 2); got != "noul:2" {
		t.Fatalf("bucket %q", got)
	}
	if ClampTemperature(0.1) != 0.5 {
		t.Fatal("clamp lo")
	}
	if ClampTemperature(9) != 5.0 {
		t.Fatal("clamp hi")
	}
	if ClampTemperature(math.NaN()) != 1.0 {
		t.Fatal("nan")
	}
}

type fakeLayaTok struct {
	mask string
	ids  map[string][]int
}

func (f fakeLayaTok) Encode(text string) ([]int, error) {
	if ids, ok := f.ids[text]; ok {
		return append([]int(nil), ids...), nil
	}
	// Deterministic fallback: one id per rune starting at 100.
	out := make([]int, 0, len([]rune(text)))
	for i := range []rune(text) {
		out = append(out, 100+i)
	}
	return out, nil
}
func (f fakeLayaTok) MaskToken() string { return f.mask }
func (f fakeLayaTok) MaskTokenID() int  { return 1 }
func (f fakeLayaTok) ClsTokenID() int   { return 2 }
func (f fakeLayaTok) SepTokenID() int   { return 3 }

func TestBuildSequenceMarkers(t *testing.T) {
	tok := fakeLayaTok{
		mask: "[MASK]",
		ids: map[string][]int{
			"choice question: pick": {10, 11},
			" a: x":                 {20},
			" b: y":                 {21},
			"hello":                 {30, 31},
		},
	}
	q := DecisionQuestion{
		Type:         "choice",
		Instructions: "pick",
		Criteria:     json.RawMessage(`{"a":"x","b":"y"}`),
	}
	ids, markers, err := BuildSequence(tok, "hello", q, 512, 192, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(markers) != 2 {
		t.Fatalf("markers=%v", markers)
	}
	if ids[0] != 2 {
		t.Fatalf("missing CLS: %v", ids)
	}
	if ids[markers[0]] != 1 || ids[markers[1]] != 1 {
		t.Fatalf("markers not on MASK: ids=%v markers=%v", ids, markers)
	}
	if ids[len(ids)-1] != 3 {
		t.Fatalf("missing final SEP: %v", ids)
	}
}

func TestDecodeAnswerChoice(t *testing.T) {
	q := DecisionQuestion{
		Type:         "choice",
		Instructions: "pick",
		Criteria:     json.RawMessage(`{"a":"x","b":"y"}`),
	}
	raw, err := DecodeAnswer(q, []float64{2.0, 0.0}, []float64{0.75, 0.25}, []float64{1, 1, 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var ans ChoiceAnswer
	if err := json.Unmarshal(raw, &ans); err != nil {
		t.Fatal(err)
	}
	if ans.Type != "choice" || ans.Choice != "a" {
		t.Fatalf("%+v", ans)
	}
	if ans.Action.ActProbability != 0.75 {
		t.Fatalf("act %v", ans.Action.ActProbability)
	}
}

func TestSoftmaxTemp(t *testing.T) {
	p := SoftmaxTemp([]float64{0, 0}, 2, 1.0)
	if len(p) != 2 || math.Abs(p[0]-0.5) > 1e-9 {
		t.Fatalf("%v", p)
	}
}
