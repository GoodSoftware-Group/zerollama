package sample

import (
	"math"
	"testing"
)

func TestApplyPenalties_Presence(t *testing.T) {
	logits := []float32{1, 2, 3, 4}
	out := applyPenalties(logits, []int32{1, 1, 2}, Penalties{LastN: 64, Presence: 10})
	if out[1] >= logits[1] || out[2] >= logits[2] {
		t.Fatalf("presence should lower seen tokens: %v vs %v", out, logits)
	}
	if out[0] != logits[0] || out[3] != logits[3] {
		t.Fatalf("unseen tokens should be unchanged: %v", out)
	}
}

func TestApplyPenalties_Repeat(t *testing.T) {
	logits := []float32{4, -4}
	out := applyPenalties(logits, []int32{0, 1}, Penalties{LastN: -1, Repeat: 2})
	if out[0] != 2 { // 4/2
		t.Fatalf("positive logit /= repeat: got %v", out[0])
	}
	if out[1] != -8 { // -4*2
		t.Fatalf("negative logit *= repeat: got %v", out[1])
	}
}

func TestSamplerWithPenaltiesChangesArgmax(t *testing.T) {
	// Token 1 has slightly higher logit but appears in history with huge presence.
	logits := []float32{1.0, 1.1, 0.5}
	s := NewSampler(0, 0, 0, 0, 0, nil).WithPenalties(64, 1, 5, 0)
	got, err := s.Sample(logits, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("expected argmax to flip to 0 after presence penalty, got %d", got)
	}
	// Without history, greedy still picks 1.
	got, err = s.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("no history: want 1 got %d", got)
	}
	if math.IsNaN(float64(logits[0])) {
		t.Fatal("original logits mutated")
	}
}
