package llm

import (
	"encoding/json"
	"math"
	"testing"
)

// Frozen fixture shapes for Laya packing / calibration parity (no HF download).
// When a converted GGUF + Python Agent goldens are available, extend with token-id
// and logit argmax checks under testdata/laya/.

func TestLayaParityFixture_ChoiceSoftmaxAndConfidence(t *testing.T) {
	q := DecisionQuestion{
		Type:         "choice",
		Instructions: "Which department?",
		Criteria:     json.RawMessage(`{"billing":"invoices","tech":"bugs","sales":"pricing"}`),
	}
	logits := []float64{2.0, 0.1, 0.05}
	act := []float64{0.9, 0.1}
	raw, err := DecodeAnswer(q, logits, act, []float64{1, 1, 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var ans ChoiceAnswer
	if err := json.Unmarshal(raw, &ans); err != nil {
		t.Fatal(err)
	}
	if ans.Choice != "billing" {
		t.Fatalf("choice=%q want billing", ans.Choice)
	}
	if ans.Confidence <= 0 || ans.Confidence > 1 {
		t.Fatalf("confidence=%v", ans.Confidence)
	}
	sum := 0.0
	for _, p := range ans.Probabilities {
		sum += p
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("probs sum=%v", sum)
	}
}

func TestLayaParityFixture_PackQuestionOrderStable(t *testing.T) {
	tok := fakeLayaTok{
		mask: "[MASK]",
		ids: map[string][]int{
			"choice question: Which?": {10, 11},
			" billing: invoices":      {20, 21},
			" tech: bugs":             {30},
			"hello":                   {40, 41, 42},
		},
	}
	qs := map[string]DecisionQuestion{
		"z_last": {
			Type:         "choice",
			Instructions: "Which?",
			Criteria:     json.RawMessage(`{"billing":"invoices","tech":"bugs"}`),
		},
		"a_first": {
			Type:         "choice",
			Instructions: "Which?",
			Criteria:     json.RawMessage(`{"billing":"invoices","tech":"bugs"}`),
		},
	}
	packed, err := PackQuestions(tok, "hello", qs, 64, 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) != 2 {
		t.Fatalf("len=%d", len(packed))
	}
	if packed[0].Question != "a_first" || packed[1].Question != "z_last" {
		t.Fatalf("order=%q,%q", packed[0].Question, packed[1].Question)
	}
	for _, p := range packed {
		if p.QType != 0 {
			t.Fatalf("qtype=%d", p.QType)
		}
		if len(p.MarkerPos) != 2 {
			t.Fatalf("markers=%v", p.MarkerPos)
		}
		if p.Tokens[0] != 2 { // fake CLS
			t.Fatalf("missing CLS: %v", p.Tokens)
		}
	}
}
