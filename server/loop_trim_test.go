package server

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestTrimTripleRepeatRunes(t *testing.T) {
	cycle := "abcdefgh"
	s := "hi" + cycle + cycle + cycle
	got, ok := trimTripleRepeatRunes(s)
	if !ok || got != "hi" {
		t.Fatalf("got %q ok=%v", got, ok)
	}
}

func TestApplyLoopTrim(t *testing.T) {
	cycle := "abcdefgh"
	s := "hi" + cycle + cycle + cycle
	if got := applyLoopTrim(s, nil); got != s {
		t.Fatal("no details must not trim")
	}
	if got := applyLoopTrim(s, &api.FinishDetails{Type: "length"}); got != s {
		t.Fatal("wrong type must not trim")
	}
	if got := applyLoopTrim(s, &api.FinishDetails{Type: "repetition_loop"}); got != "hi" {
		t.Fatalf("trimmed=%q", got)
	}
}

func TestApplyLoopTrimDisabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_LOOP_TRIM", "0")
	cycle := "abcdefgh"
	s := "hi" + cycle + cycle + cycle
	if got := applyLoopTrim(s, &api.FinishDetails{Type: "repetition_loop"}); got != s {
		t.Fatalf("disabled trim should keep content, got %q", got)
	}
}
