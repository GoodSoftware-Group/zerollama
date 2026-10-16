package llm

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// clmScalarProject is the pre-batching reference: one text, one core, one running sum.
func clmScalarProject(m clmMLP, x []float32, act string, residual bool) []float32 {
	lin := func(l clmLinear, x []float32) []float32 {
		y := make([]float32, l.out)
		for o := range l.out {
			s := l.b[o]
			for i := range l.in {
				s += x[i] * l.w[o*l.in+i]
			}
			y[o] = s
		}
		return y
	}
	h := lin(m.inp, x)
	clmActivate(h, act)
	for i, l := range m.hidden {
		z := lin(l, h)
		if len(m.norms) > i {
			clmLayerNorm(z, m.norms[i], m.normB[i])
		}
		clmActivate(z, act)
		if residual {
			for j := range h {
				h[j] += z[j]
			}
		} else {
			h = z
		}
	}
	y := lin(m.out, h)
	clmL2Normalize(y)
	return y
}

func randCLMMLP(rng *rand.Rand, hidden, width, proj, depth int, ln bool) clmMLP {
	mk := func(in, out int) clmLinear {
		l := clmLinear{in: in, out: out, w: make([]float32, in*out), b: make([]float32, out)}
		scale := float32(1 / math.Sqrt(float64(in)))
		for i := range l.w {
			l.w[i] = (rng.Float32()*2 - 1) * scale
		}
		for i := range l.b {
			l.b[i] = rng.Float32()*0.2 - 0.1
		}
		return l
	}
	m := clmMLP{inp: mk(hidden, width), out: mk(width, proj)}
	for range depth - 2 {
		m.hidden = append(m.hidden, mk(width, width))
		if ln {
			w, b := make([]float32, width), make([]float32, width)
			for i := range w {
				w[i], b[i] = 1+rng.Float32()*0.1, rng.Float32()*0.1
			}
			m.norms, m.normB = append(m.norms, w), append(m.normB, b)
		}
	}
	return m
}

// TestCLMProjectBatchMatchesScalar pins Finding 25: the batched, threaded heads give the
// same projections as the old one-text-at-a-time code, up to float32 rounding, for every
// layer option and for batches above and below the threading threshold.
func TestCLMProjectBatchMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, tc := range []struct {
		name                 string
		hidden, width, depth int
		act                  string
		ln, resid            bool
		batch                int
	}{
		{"tiny-inline", 37, 19, 2, "gelu", false, false, 3},
		{"clm-v0.1-shape", 4096, 1536, 3, "gelu", false, false, 6},
		{"layernorm-residual", 512, 256, 4, "silu", true, true, 5},
		{"relu-single", 1024, 512, 3, "relu", true, false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := randCLMMLP(rng, tc.hidden, tc.width, 64, tc.depth, tc.ln)
			h := &CLMHeads{state: m, action: m, Act: tc.act, Resid: tc.resid}
			xs := make([][]float32, tc.batch)
			for r := range xs {
				xs[r] = make([]float32, tc.hidden)
				for i := range xs[r] {
					xs[r][i] = rng.Float32()*2 - 1
				}
			}
			got := h.ProjectStates(xs)
			for r, x := range xs {
				want := clmScalarProject(m, x, tc.act, tc.resid)
				for i := range want {
					if d := math.Abs(float64(got[r][i] - want[i])); d > 1e-5 {
						t.Fatalf("row %d dim %d: batched %v scalar %v (|Δ| %g)", r, i, got[r][i], want[i], d)
					}
				}
			}
		})
	}
}

// TestCLMProjectBatchSpeed logs the CLM v0.1 shape cost (1 state + 5 actions) so a
// regression back to scalar projection is visible in -v output.
func TestCLMProjectBatchSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("timing")
	}
	rng := rand.New(rand.NewSource(2))
	m := randCLMMLP(rng, 4096, 1536, 512, 3, false)
	h := &CLMHeads{state: m, action: m, Act: "gelu"}
	xs := make([][]float32, 6)
	for r := range xs {
		xs[r] = make([]float32, 4096)
	}
	h.ProjectActions(xs[1:])
	start := time.Now()
	for range 10 {
		h.ProjectStates(xs[:1])
		h.ProjectActions(xs[1:])
	}
	t.Logf("CLM v0.1 shape: %v per decision (1 state + 5 actions)", time.Since(start)/10)
}
