package llm

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/ollama/ollama/fs/ggml"
)

// CLMHeads is a loaded Contrastive-LM projection-head pair from a heads-only GGUF.
// Encoder embeddings come from a separate /v1/embeddings server (llama-server).
type CLMHeads struct {
	Path   string
	Hidden int
	Width  int
	Depth  int
	Proj   int
	Act    string
	LN     bool
	Resid  bool
	Scale  float32 // exp(logit_scale), clamped

	state  clmMLP
	action clmMLP
}

type clmLinear struct {
	// W is row-major (out, in) matching PyTorch nn.Linear.weight
	out, in int
	w       []float32
	b       []float32
}

type clmMLP struct {
	inp    clmLinear
	hidden []clmLinear
	norms  [][]float32 // weight, bias interleaved per layer; empty if no LN
	normB  [][]float32
	out    clmLinear
}

var (
	clmHeadsCache   sync.Map // path -> *CLMHeads
	clmHeadsLoadMu  sync.Mutex
)

// LoadCLMHeads loads (and caches) a CLM heads GGUF produced by scripts/convert_clm_heads_to_gguf.py.
func LoadCLMHeads(path string) (*CLMHeads, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, fmt.Errorf("empty CLM heads path")
	}
	if v, ok := clmHeadsCache.Load(path); ok {
		return v.(*CLMHeads), nil
	}
	clmHeadsLoadMu.Lock()
	defer clmHeadsLoadMu.Unlock()
	if v, ok := clmHeadsCache.Load(path); ok {
		return v.(*CLMHeads), nil
	}
	h, err := loadCLMHeadsUncached(path)
	if err != nil {
		return nil, err
	}
	clmHeadsCache.Store(path, h)
	return h, nil
}

func loadCLMHeadsUncached(path string) (*CLMHeads, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	meta, err := ggml.Decode(f, -1)
	if err != nil {
		return nil, fmt.Errorf("decode CLM GGUF: %w", err)
	}
	kv := meta.KV()
	arch := kv.Architecture()
	if arch != "" && arch != "clm" {
		return nil, fmt.Errorf("expected general.architecture=clm, got %q", arch)
	}
	h := &CLMHeads{
		Path:   path,
		Hidden: int(kv.Uint("hidden_size", 4096)),
		Width:  int(kv.Uint("width", 0)),
		Depth:  int(kv.Uint("depth", 2)),
		Proj:   int(kv.Uint("projection_dim", 512)),
		Act:    strings.ToLower(kv.String("activation", "gelu")),
		LN:     kv.Bool("layernorm", false),
		Resid:  kv.Bool("residual", false),
	}
	if h.Width == 0 {
		return nil, fmt.Errorf("clm.width missing in %s", path)
	}
	ls := float32(kv.Float("logit_scale", 0))
	h.Scale = float32(math.Exp(float64(ls)))
	if h.Scale > 100 {
		h.Scale = 100
	}
	if h.state, err = loadCLMMLP(path, "state", h); err != nil {
		return nil, err
	}
	if h.action, err = loadCLMMLP(path, "action", h); err != nil {
		return nil, err
	}
	return h, nil
}

func loadCLMMLP(path, which string, h *CLMHeads) (clmMLP, error) {
	prefix := "clm." + which + "."
	inp, err := loadCLMLinear(path, prefix+"inp", h.Hidden, h.Width)
	if err != nil {
		return clmMLP{}, err
	}
	nHidden := h.Depth - 2
	if nHidden < 0 {
		nHidden = 0
	}
	var hidden []clmLinear
	var normsW, normsB [][]float32
	for i := 0; i < nHidden; i++ {
		lin, err := loadCLMLinear(path, fmt.Sprintf("%shidden.%d", prefix, i), h.Width, h.Width)
		if err != nil {
			return clmMLP{}, err
		}
		hidden = append(hidden, lin)
		if h.LN {
			w, err := loadCLMVec(path, fmt.Sprintf("%snorms.%d.weight", prefix, i), h.Width)
			if err != nil {
				return clmMLP{}, err
			}
			b, err := loadCLMVec(path, fmt.Sprintf("%snorms.%d.bias", prefix, i), h.Width)
			if err != nil {
				return clmMLP{}, err
			}
			normsW = append(normsW, w)
			normsB = append(normsB, b)
		}
	}
	out, err := loadCLMLinear(path, prefix+"out", h.Width, h.Proj)
	if err != nil {
		return clmMLP{}, err
	}
	return clmMLP{inp: inp, hidden: hidden, norms: normsW, normB: normsB, out: out}, nil
}

func loadCLMLinear(path, base string, in, out int) (clmLinear, error) {
	w, shape, err := readCLMF32(path, base+".weight")
	if err != nil {
		return clmLinear{}, err
	}
	// GGUF shape is typically [ne0, ne1] = [cols, rows] in ggml — check element count.
	if len(w) != in*out {
		// Try transposed interpretation: PyTorch dumps (out, in); gguf-py may reverse.
		if len(shape) == 2 && int(shape[0])*int(shape[1]) == in*out {
			// ok size-wise
		} else {
			return clmLinear{}, fmt.Errorf("%s.weight: got %d floats, want %d*%d", base, len(w), out, in)
		}
	}
	// Prefer (out, in) layout as written by convert script (ne = out, in in numpy C order).
	// If shape[0]==in && shape[1]==out, data is already (out,in) row-major from numpy (out, in).
	b, err := loadCLMVec(path, base+".bias", out)
	if err != nil {
		return clmLinear{}, err
	}
	return clmLinear{out: out, in: in, w: w, b: b}, nil
}

func loadCLMVec(path, name string, n int) ([]float32, error) {
	v, _, err := readCLMF32(path, name)
	if err != nil {
		return nil, err
	}
	if len(v) != n {
		return nil, fmt.Errorf("%s: got %d, want %d", name, len(v), n)
	}
	return v, nil
}

func readCLMF32(path, name string) ([]float32, []uint64, error) {
	raw, tensor, err := ggml.ReadTensorBytes(path, name)
	if err != nil {
		return nil, nil, err
	}
	if ggml.TensorType(tensor.Kind) != ggml.TensorTypeF32 {
		return nil, nil, fmt.Errorf("%s: want F32, kind=%d", name, tensor.Kind)
	}
	if len(raw)%4 != 0 {
		return nil, nil, fmt.Errorf("%s: bad byte length %d", name, len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, tensor.Shape, nil
}

// projectBatch runs every row of xs through the MLP one layer at a time.
func (m clmMLP) projectBatch(xs [][]float32, act string, residual bool) [][]float32 {
	hs := m.inp.forwardBatch(xs)
	for _, h := range hs {
		clmActivate(h, act)
	}
	for i, lin := range m.hidden {
		zs := lin.forwardBatch(hs)
		for r, z := range zs {
			if len(m.norms) > i {
				clmLayerNorm(z, m.norms[i], m.normB[i])
			}
			clmActivate(z, act)
			if residual {
				for j := range hs[r] {
					hs[r][j] += z[j]
				}
			} else {
				hs[r] = z
			}
		}
	}
	return m.out.forwardBatch(hs)
}

// clmParallelMinMACs is the work below which threading costs more than it saves.
const clmParallelMinMACs = 1 << 18

// forwardBatch computes y[r][o] = bias[o] + sum_i xs[r][i]*W[o,i] for every row r.
//
// WHY batched + threaded: the heads used to project one text at a time with a scalar
// loop on one core — 10–50 ms per decision for 1 state + 5 actions (width 512–2048),
// growing with every action (docs/auto-router-findings.md Finding 25). Output rows
// are split across goroutines and each weight row is reused for every input while it
// is still in cache, so W streams from memory once per call instead of once per text.
func (l clmLinear) forwardBatch(xs [][]float32) [][]float32 {
	ys := make([][]float32, len(xs))
	for r, x := range xs {
		if len(x) != l.in {
			panic(fmt.Sprintf("clm linear: in=%d got %d", l.in, len(x)))
		}
		ys[r] = make([]float32, l.out)
	}
	rows := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			w := l.w[o*l.in : (o+1)*l.in]
			for r, x := range xs {
				ys[r][o] = l.b[o] + clmDot(x, w)
			}
		}
	}
	workers := min(runtime.GOMAXPROCS(0), l.out/16)
	if workers <= 1 || l.out*l.in*len(xs) < clmParallelMinMACs {
		rows(0, l.out)
		return ys
	}
	var wg sync.WaitGroup
	chunk := (l.out + workers - 1) / workers
	for lo := 0; lo < l.out; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			rows(lo, hi)
		}(lo, min(lo+chunk, l.out))
	}
	wg.Wait()
	return ys
}

func clmActivate(x []float32, act string) {
	switch act {
	case "relu":
		for i, v := range x {
			if v < 0 {
				x[i] = 0
			}
		}
	case "silu":
		for i, v := range x {
			x[i] = v / (1 + float32(math.Exp(float64(-v))))
		}
	default: // gelu (tanh approx)
		for i, v := range x {
			x[i] = 0.5 * v * (1 + float32(math.Tanh(float64(0.7978845608*(v+0.044715*v*v*v)))))
		}
	}
}

func clmLayerNorm(x, w, b []float32) {
	const eps = 1e-5
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var var_ float32
	for _, v := range x {
		d := v - mean
		var_ += d * d
	}
	var_ /= float32(len(x))
	inv := float32(1 / math.Sqrt(float64(var_+eps)))
	for i := range x {
		x[i] = (x[i]-mean)*inv*w[i] + b[i]
	}
}

func clmL2Normalize(x []float32) {
	var n float64
	for _, v := range x {
		n += float64(v) * float64(v)
	}
	n = math.Sqrt(n)
	if n < 1e-12 {
		return
	}
	inv := float32(1 / n)
	for i := range x {
		x[i] *= inv
	}
}

// ProjectStates runs the state head and L2-normalizes rows.
func (h *CLMHeads) ProjectStates(embs [][]float32) [][]float32 {
	out := h.state.projectBatch(embs, h.Act, h.Resid)
	for _, p := range out {
		clmL2Normalize(p)
	}
	return out
}

// ProjectActions runs the action head and L2-normalizes rows.
func (h *CLMHeads) ProjectActions(embs [][]float32) [][]float32 {
	out := h.action.projectBatch(embs, h.Act, h.Resid)
	for _, p := range out {
		clmL2Normalize(p)
	}
	return out
}

// clmDot returns sum_i a[i]*b[i]. Four partial sums break the add dependency chain
// (~4x on one core); the result differs from a single running sum only by rounding.
func clmDot(a, b []float32) float32 {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}
