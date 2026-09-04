package mlxrunner

import (
	"log/slog"
)

// mlx-serve loop-stop: halt decode when a generated cycle repeats three
// times. Prompt tokens are not in session.outputs. Finish as length (same
// as NumPredict), not EOS. finish_details.type = repetition_loop.
const (
	// Short cycles (": 2,5") sit under mlx-serve's 8-token floor and never
	// hit EOS on DSv4 Flash; six copies is enough to halt without treating
	// "ha ha ha" as a loop.
	loopShortMinPeriod = 2
	loopShortMaxPeriod = 7
	loopShortRepeats   = 6

	loopMinPeriod = 8
	loopMaxPeriod = 48
	loopRepeats   = 3

	// mlx-serve isDegenerateTailLoopRange: longer cycles need more copies
	// so a 50-token refrain is not a false stop at 3×.
	loopLongMinPeriod = 9
	loopLongMaxPeriod = 64
	loopLongRepeats   = 10

	nearRepeatWindow   = 1024
	nearRepeatMinLen   = 64
	nearRepeatDistinct = 0.12
	nearRepeat4gram    = 0.35
	nearRepeatNovelty  = 0.10
)

const finishDetailsRepetitionLoop = "repetition_loop"

func generationShouldStop(toks []int32) bool {
	return generationIsLoop(toks) || generationIsNearRepeatLoop(toks)
}

func generationIsLoop(toks []int32) bool {
	_, ok := loopSpanStart(toks)
	return ok
}

// loopSpanStart is the index of the first copy of a triple-repeated cycle.
// mlx-serve trims non-streaming output to this span start.
func loopSpanStart(toks []int32) (int, bool) {
	if start, ok := exactRepeatSpan(toks, loopShortMinPeriod, loopShortMaxPeriod, loopShortRepeats); ok {
		return start, true
	}
	if start, ok := exactRepeatSpan(toks, loopMinPeriod, loopMaxPeriod, loopRepeats); ok {
		return start, true
	}
	return exactRepeatSpan(toks, loopLongMinPeriod, loopLongMaxPeriod, loopLongRepeats)
}

func exactRepeatSpan(toks []int32, minP, maxP, repeats int) (int, bool) {
	n := len(toks)
	if maxP > n/repeats {
		maxP = n / repeats
	}
	for p := minP; p <= maxP; p++ {
		base := n - repeats*p
		first := toks[base : base+p]
		ok := true
		for r := 1; r < repeats; r++ {
			if !equalTokens(first, toks[base+r*p:base+(r+1)*p]) {
				ok = false
				break
			}
		}
		if ok {
			return base, true
		}
	}
	return 0, false
}

// generationIsNearRepeatLoop is mlx-serve's soft loop: a long tail that is
// not an exact cycle but has collapsed vocabulary, 4-grams, and novelty.
func generationIsNearRepeatLoop(toks []int32) bool {
	w := toks
	if len(w) > nearRepeatWindow {
		w = w[len(w)-nearRepeatWindow:]
	}
	n := len(w)
	if n < nearRepeatMinLen {
		return false
	}
	seen := make(map[int32]struct{}, n)
	for _, t := range w {
		seen[t] = struct{}{}
	}
	if float64(len(seen))/float64(n) > nearRepeatDistinct {
		return false
	}
	grams := make(map[[4]int32]struct{}, n)
	for i := 0; i+4 <= n; i++ {
		grams[[4]int32{w[i], w[i+1], w[i+2], w[i+3]}] = struct{}{}
	}
	if n > 3 && float64(len(grams))/float64(n-3) > nearRepeat4gram {
		return false
	}
	half := n / 2
	prefix := make(map[int32]struct{}, half)
	for _, t := range w[:half] {
		prefix[t] = struct{}{}
	}
	novel := 0
	for _, t := range w[half:] {
		if _, ok := prefix[t]; !ok {
			novel++
		}
	}
	if float64(novel)/float64(n-half) > nearRepeatNovelty {
		return false
	}
	return true
}

func logLoopStop(toks []int32) {
	slog.Info("mlx loop-stop", "generated", len(toks), "reason", finishDetailsRepetitionLoop)
}
