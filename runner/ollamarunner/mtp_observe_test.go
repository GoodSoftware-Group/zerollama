package ollamarunner

import "testing"

func TestMaybeAcceptMTPRequiresTwoLogits(t *testing.T) {
	s := &Server{}
	seq := &Sequence{mtpVerify: true, mtpIBatch0: 0}
	if _, _, ok := s.maybeAcceptMTP(seq, nil, 4, 0); ok {
		t.Fatal("one logit slot must not accept")
	}
	if !seq.mtpVerify {
		t.Fatal("mtpVerify should stay set when verify cannot run")
	}
}
