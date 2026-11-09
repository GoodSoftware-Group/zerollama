package voicepipe

import (
	"errors"
	"testing"
)

func TestParseTurnRecordsThresholdSet(t *testing.T) {
	on, turn, err := ParseTurn(map[string]any{"type": "server_vad"})
	if err != nil || !on || turn.ThresholdSet || turn.Threshold != DefaultAmplitude || turn.AmplitudeScale != DefaultAmplitude {
		t.Fatalf("omitted threshold: on=%v set=%v thr=%v amp=%v err=%v", on, turn.ThresholdSet, turn.Threshold, turn.AmplitudeScale, err)
	}
	on, turn, err = ParseTurn(map[string]any{"type": "server_vad", "threshold": 0.5})
	if err != nil || !on || !turn.ThresholdSet || turn.Threshold != 0.5 || turn.AmplitudeScale != 0.5 {
		t.Fatalf("set threshold: on=%v set=%v thr=%v amp=%v err=%v", on, turn.ThresholdSet, turn.Threshold, turn.AmplitudeScale, err)
	}
	turn.ApplyNeuralThreshold()
	if turn.Threshold != 0.5 {
		t.Fatalf("explicit neural threshold changed to %v", turn.Threshold)
	}
	if turn.AmplitudeScale != DefaultAmplitude {
		t.Fatalf("neural amp scale %v", turn.AmplitudeScale)
	}
	omit := DefaultTurn()
	omit.ApplyNeuralThreshold()
	if omit.Threshold != NeuralDefaultThreshold {
		t.Fatalf("neural default %v", omit.Threshold)
	}
	if omit.AmplitudeScale != DefaultAmplitude {
		t.Fatalf("neural amp %v", omit.AmplitudeScale)
	}
}

func TestGateUsesScorerLastSpeech(t *testing.T) {
	// 4 samples: quiet, quiet, loud-prob, quiet → last speech index 2 → tail 1.
	pcm := make([]byte, 8)
	var calls int
	g := Gate{
		Turn: Turn{Threshold: 0.5, AmplitudeScale: DefaultAmplitude, SilenceMS: 1000, PrefixMS: 0},
		Rate: 4, // 1 sample = 250ms; need 4 quiet samples to commit
		Score: func(p []byte, rate int, thr float64) (int, error) {
			calls++
			if rate != 4 || thr != 0.5 || len(p) != 8 {
				t.Fatalf("score args rate=%d thr=%v len=%d", rate, thr, len(p))
			}
			return 2, nil
		},
	}
	step := g.Push(pcm, false)
	if !step.Started || calls != 1 {
		t.Fatalf("started=%v calls=%d", step.Started, calls)
	}
	if g.quiet != 1 {
		t.Fatalf("quiet=%d want 1", g.quiet)
	}
}

func TestGateFallsBackUsesAmplitudeScale(t *testing.T) {
	// Neural Threshold 0.5 would miss this sample as amplitude; AmplitudeScale 0.02 must win.
	pcm := make([]byte, 4)
	pcm[0], pcm[1] = 0xa0, 0x0f // 4000 little-endian (~0.12 of full scale)
	g := Gate{
		Turn: Turn{Threshold: 0.5, AmplitudeScale: DefaultAmplitude, SilenceMS: 500, PrefixMS: 0},
		Rate: 16000,
		Score: func([]byte, int, float64) (int, error) {
			return 0, errors.New("timeout")
		},
	}
	step := g.Push(pcm, false)
	if !step.Started {
		t.Fatal("expected amplitude fallback start")
	}
}

func TestGateOnResetOnCommit(t *testing.T) {
	var resets int
	pcm := make([]byte, 4)
	pcm[0], pcm[1] = 0xa0, 0x0f
	g := Gate{
		Turn:    Turn{Threshold: DefaultAmplitude, AmplitudeScale: DefaultAmplitude, SilenceMS: 0, PrefixMS: 0},
		Rate:    16000,
		OnReset: func() { resets++ },
	}
	step := g.Push(pcm, false)
	if step.Commit == nil {
		quiet := make([]byte, 4)
		step = g.Push(quiet, false)
	}
	if step.Commit == nil {
		t.Fatal("expected commit")
	}
	if resets != 1 {
		t.Fatalf("resets=%d", resets)
	}
}
