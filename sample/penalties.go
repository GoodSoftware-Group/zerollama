package sample

import "slices"

// Penalties are repetition / presence / frequency adjustments applied to logits
// before sampling. Zero-value Repeat is treated as 1 (disabled). Matches the
// llama.cpp sampling-penalty shape used by llamarunner.
//
// Minefield U02: ollamarunner historically accepted these API fields and
// discarded them because NewSampler ignored them.
type Penalties struct {
	LastN     int     // 0 = disabled; -1 = full history
	Repeat    float32 // 1 = disabled
	Presence  float32
	Frequency float32
}

func (p Penalties) active() bool {
	repeat := p.Repeat
	if repeat == 0 {
		repeat = 1
	}
	return p.LastN != 0 && (repeat != 1 || p.Presence != 0 || p.Frequency != 0)
}

// applyPenalties returns a copy of logits with penalties applied over recent tokens.
func applyPenalties(logits []float32, recent []int32, p Penalties) []float32 {
	if !p.active() || len(recent) == 0 || len(logits) == 0 {
		return logits
	}
	lastN := p.LastN
	if lastN < 0 || lastN > len(recent) {
		lastN = len(recent)
	}
	window := recent[len(recent)-lastN:]
	counts := make(map[int32]int, len(window))
	for _, id := range window {
		counts[id]++
	}
	repeat := p.Repeat
	if repeat == 0 {
		repeat = 1
	}
	out := slices.Clone(logits)
	for id, count := range counts {
		if id < 0 || int(id) >= len(out) {
			continue
		}
		v := out[id]
		if repeat != 1 {
			if v > 0 {
				v /= repeat
			} else {
				v *= repeat
			}
		}
		if p.Presence != 0 {
			v -= p.Presence
		}
		if p.Frequency != 0 {
			v -= p.Frequency * float32(count)
		}
		out[id] = v
	}
	return out
}
