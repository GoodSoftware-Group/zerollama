package freetokenlab

import "math"

// StickyDecode synthesizes decode routing with temporal locality: each layer
// keeps a working set of size work and only rarely jumps (pJump).
func StickyDecode(layers, steps, nExperts, work int, pJumpNum, pJumpDen, seed int) []TraceStep {
	if work < 1 {
		work = 1
	}
	if work > nExperts {
		work = nExperts
	}
	rng := seed
	next := func() int {
		rng = rng*1103515245 + 12345
		if rng < 0 {
			rng = -rng
		}
		return rng
	}
	out := make([]TraceStep, 0, layers*steps)
	ws := make([][]int, layers)
	for L := 0; L < layers; L++ {
		base := (L * 7) % nExperts
		w := make([]int, work)
		for i := 0; i < work; i++ {
			w[i] = (base + i) % nExperts
		}
		ws[L] = w
	}
	for t := 0; t < steps; t++ {
		for L := 0; L < layers; L++ {
			if pJumpDen > 0 && next()%pJumpDen < pJumpNum {
				shift := next() % nExperts
				for i := range ws[L] {
					ws[L][i] = (ws[L][i] + shift) % nExperts
				}
			}
			ex := append([]int(nil), ws[L]...)
			out = append(out, TraceStep{Layer: L, Experts: ex})
		}
	}
	return out
}

// DensePrefill activates most experts per layer (prefill working set ≈ dense).
func DensePrefill(layers, nExperts, tokens, k int) []TraceStep {
	if k < 1 {
		k = 1
	}
	out := make([]TraceStep, 0, layers*tokens)
	for t := 0; t < tokens; t++ {
		for L := 0; L < layers; L++ {
			ex := make([]int, k)
			for i := 0; i < k; i++ {
				ex[i] = (t*k + i + L) % nExperts
			}
			out = append(out, TraceStep{Layer: L, Experts: ex})
		}
	}
	return out
}

// ZipfDecode samples k unique experts per layer/step from a Zipf over the pool.
func ZipfDecode(layers, steps, nExperts, k int, s float64, seed int) []TraceStep {
	if k < 1 {
		k = 1
	}
	if k > nExperts {
		k = nExperts
	}
	if s < 0.1 {
		s = 0.1
	}
	w := make([]float64, nExperts)
	var sum float64
	for i := 0; i < nExperts; i++ {
		w[i] = 1.0 / math.Pow(float64(i+1), s)
		sum += w[i]
	}
	rng := seed
	next := func() int {
		rng = rng*1103515245 + 12345
		if rng < 0 {
			rng = -rng
		}
		return rng
	}
	pick := func() int {
		x := float64(next()%1000000) / 1000000.0 * sum
		acc := 0.0
		for i, wi := range w {
			acc += wi
			if x <= acc {
				return i
			}
		}
		return nExperts - 1
	}
	out := make([]TraceStep, 0, layers*steps)
	for t := 0; t < steps; t++ {
		for L := 0; L < layers; L++ {
			seen := map[int]bool{}
			ex := make([]int, 0, k)
			guard := 0
			for len(ex) < k && guard < nExperts*8 {
				guard++
				e := (pick() + L*3) % nExperts
				if seen[e] {
					continue
				}
				seen[e] = true
				ex = append(ex, e)
			}
			out = append(out, TraceStep{Layer: L, Experts: ex})
		}
	}
	return out
}

// ZipfStickyDecode is Zipf with inter-token overlap: with probability
// pReuseNum/pReuseDen the layer keeps last step's experts.
func ZipfStickyDecode(layers, steps, nExperts, k int, s float64, pReuseNum, pReuseDen, seed int) []TraceStep {
	base := ZipfDecode(layers, steps, nExperts, k, s, seed)
	if pReuseDen <= 0 || len(base) == 0 {
		return base
	}
	rng := seed + 99
	next := func() int {
		rng = rng*1103515245 + 12345
		if rng < 0 {
			rng = -rng
		}
		return rng
	}
	last := map[int][]int{}
	out := make([]TraceStep, 0, len(base))
	for _, st := range base {
		if prev, ok := last[st.Layer]; ok && next()%pReuseDen < pReuseNum {
			st.Experts = append([]int(nil), prev...)
		}
		last[st.Layer] = st.Experts
		out = append(out, st)
	}
	return out
}
