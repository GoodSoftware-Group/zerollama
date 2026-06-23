package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// ANEPrefillBenchResult is JSON from ane-prefill-bench (dynamic matmul prefill proxy).
type ANEPrefillBenchResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	IC           int     `json:"ic"`
	OC           int     `json:"oc"`
	Seq          int     `json:"seq"`
	EvalMS       float64 `json:"eval_ms"`
	GFLOP        float64 `json:"gflop"`
	TFLOPS       float64 `json:"tflops"`
	CompileCount int     `json:"compile_count"`
	Source       string  `json:"source"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// PrefillMatmulGFLOP returns 2×IC×OC×SEQ / 1e9 for matmul FLOP accounting.
func PrefillMatmulGFLOP(ic, oc, seq int) float64 {
	return 2.0 * float64(ic) * float64(oc) * float64(seq) / 1e9
}

// DefaultPrefillICCap is the max IC/OC for --model prefill probes unless --full-embed is set.
const DefaultPrefillICCap = 2048

// PrefillProxyFromEmbed picks IC/OC/SEQ for a single-layer prefill proxy from embedding width.
func PrefillProxyFromEmbed(embedding, numTokens int) (ic, oc, seq int) {
	return PrefillProxyFromEmbedCap(embedding, numTokens, DefaultPrefillICCap)
}

// PrefillProxyFromEmbedCap picks proxy dims; icCap<=0 uses full embedding width.
func PrefillProxyFromEmbedCap(embedding, numTokens, icCap int) (ic, oc, seq int) {
	ic = embedding
	oc = embedding
	seq = numTokens
	if icCap > 0 && ic > icCap {
		ic = icCap
		oc = icCap
	}
	if ic < 64 {
		ic = 64
		oc = 64
	}
	if seq <= 0 {
		seq = 512
	}
	if seq > 4096 {
		seq = 4096
	}
	return ic, oc, seq
}

// FindANEPrefillBenchBin locates the prefill matmul bench binary.
func FindANEPrefillBenchBin() string {
	return aneToolBin("ane-prefill-bench")
}

func anePrefillBenchArgs(ic, oc, seq int, quick bool) []string {
	args := []string{}
	if ic > 0 {
		args = append(args, "--ic", strconv.Itoa(ic))
	}
	if oc > 0 {
		args = append(args, "--oc", strconv.Itoa(oc))
	}
	if seq > 0 {
		args = append(args, "--seq", strconv.Itoa(seq))
	}
	if quick {
		args = append(args, "--quick") // fewer iterations only
	}
	return args
}

// RunANEPrefillBench runs prefill-shaped dynamic matmul on ANE.
func RunANEPrefillBench(ctx context.Context, w io.Writer, ic, oc, seq int, quick bool) error {
	bin := FindANEPrefillBenchBin()
	out, err := runANETool(ctx, bin, anePrefillBenchArgs(ic, oc, seq, quick))
	if len(out) > 0 {
		_, _ = w.Write(out)
	}
	return err
}

// RunANEPrefillBenchForModel uses GGUF embedding width and optional token count.
func RunANEPrefillBenchForModel(ctx context.Context, w io.Writer, preferred string, numTokens int, quick, fullEmbed bool) error {
	ic, oc, seq, err := prefillDimsForModel(preferred, numTokens, quick, fullEmbed)
	if err != nil {
		return err
	}
	return RunANEPrefillBench(ctx, w, ic, oc, seq, quick)
}

// ProbeANEPrefillBench parses prefill bench JSON.
func ProbeANEPrefillBench(ctx context.Context, ic, oc, seq int, quick bool) (ANEPrefillBenchResult, error) {
	bin := FindANEPrefillBenchBin()
	if bin == "" {
		return ANEPrefillBenchResult{}, fmt.Errorf("ane-prefill-bench not found — run ./scripts/ane_probe_build.sh")
	}
	out, err := runANETool(ctx, bin, anePrefillBenchArgs(ic, oc, seq, quick))
	if err != nil && len(out) == 0 {
		return ANEPrefillBenchResult{}, err
	}
	var res ANEPrefillBenchResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEPrefillBenchResult{}, fmt.Errorf("ane-prefill-bench json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "prefill bench returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

// MetalPrefillBenchResult is JSON from metal-prefill-bench.
type MetalPrefillBenchResult = ANEPrefillBenchResult

// FindMetalPrefillBenchBin locates the Metal prefill matmul bench binary.
func FindMetalPrefillBenchBin() string {
	return aneToolBin("metal-prefill-bench")
}

// ProbeMetalPrefillBench runs Metal matmul at IC×OC×SEQ.
func ProbeMetalPrefillBench(ctx context.Context, ic, oc, seq int, quick bool) (MetalPrefillBenchResult, error) {
	bin := FindMetalPrefillBenchBin()
	if bin == "" {
		return MetalPrefillBenchResult{}, fmt.Errorf("metal-prefill-bench not found — run ./scripts/ane_probe_build.sh")
	}
	out, err := runANETool(ctx, bin, anePrefillBenchArgs(ic, oc, seq, quick))
	if err != nil && len(out) == 0 {
		return MetalPrefillBenchResult{}, err
	}
	var res MetalPrefillBenchResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return MetalPrefillBenchResult{}, fmt.Errorf("metal-prefill-bench json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "metal prefill bench returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

// ANEPrefillCompareResult is ANE vs Metal at the same matmul geometry.
type ANEPrefillCompareResult struct {
	OK                       bool                     `json:"ok"`
	IC                       int                      `json:"ic"`
	OC                       int                      `json:"oc"`
	Seq                      int                      `json:"seq"`
	GFLOP                    float64                  `json:"gflop"`
	ANE                      ANEPrefillBenchResult    `json:"ane"`
	Metal                    MetalPrefillBenchResult  `json:"metal"`
	MetalMPS                 *MetalPrefillBenchResult `json:"metal_mps,omitempty"`
	Faster                   string                   `json:"faster"`
	FasterBy                 float64                  `json:"faster_by"`
	LatencyRatioANEOverMetal float64                  `json:"latency_ratio_ane_over_metal"`
	Note                     string                   `json:"note,omitempty"`
	Error                    string                   `json:"error,omitempty"`
}

func prefillDimsForModel(preferred string, numTokens int, quick, fullEmbed bool) (ic, oc, seq int, err error) {
	entries, err := ListANEModelInventory()
	if err != nil {
		return 0, 0, 0, err
	}
	entry, ok := SelectANEModel(entries, preferred)
	if !ok {
		return 0, 0, 0, fmt.Errorf("no local GGUF model matching %q — run ane-model-resolve", preferred)
	}
	embed := entry.EmbeddingLength
	if embed <= 0 {
		if info, gerr := ProbeANEDraftGGUF(entry.GGUFPath, ""); gerr == nil {
			embed = info.EmbeddingLength
		}
	}
	if embed <= 0 {
		return 0, 0, 0, fmt.Errorf("model %q has unknown embedding_length", entry.Tag)
	}
	icCap := DefaultPrefillICCap
	if fullEmbed {
		icCap = 0
	}
	ic, oc, seq = PrefillProxyFromEmbedCap(embed, numTokens, icCap)
	if quick && seq > 128 {
		seq = 128
	}
	return ic, oc, seq, nil
}

// FindMetalMPSPrefillBenchBin locates the MPS matmul prefill bench binary.
func FindMetalMPSPrefillBenchBin() string {
	return aneToolBin("metal-prefill-mps-bench")
}

// ProbeMetalMPSPrefillBench runs MPS matmul at IC×OC×SEQ.
func ProbeMetalMPSPrefillBench(ctx context.Context, ic, oc, seq int, quick bool) (MetalPrefillBenchResult, error) {
	bin := FindMetalMPSPrefillBenchBin()
	if bin == "" {
		return MetalPrefillBenchResult{}, fmt.Errorf("metal-prefill-mps-bench not found — run ./scripts/ane_probe_build.sh")
	}
	out, err := runANETool(ctx, bin, anePrefillBenchArgs(ic, oc, seq, quick))
	if err != nil && len(out) == 0 {
		return MetalPrefillBenchResult{}, err
	}
	var res MetalPrefillBenchResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return MetalPrefillBenchResult{}, fmt.Errorf("metal-prefill-mps-bench json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "metal mps prefill bench returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

type prefillBackendTiming struct {
	name string
	ms   float64
}

func pickPrefillWinner(candidates ...prefillBackendTiming) (name string, fasterBy float64) {
	if len(candidates) == 0 {
		return "", 0
	}
	best := candidates[0]
	slowest := candidates[0].ms
	for _, c := range candidates {
		if c.ms < best.ms {
			best = c
		}
		if c.ms > slowest {
			slowest = c.ms
		}
	}
	if best.ms <= 0 {
		return best.name, 0
	}
	return best.name, slowest / best.ms
}

// ProbeANEPrefillCompare runs ANE and Metal prefill proxies at the same dims.
func ProbeANEPrefillCompare(ctx context.Context, ic, oc, seq int, quick bool) (ANEPrefillCompareResult, error) {
	return ProbeANEPrefillCompareFull(ctx, ic, oc, seq, quick, false)
}

// ProbeANEPrefillCompareFull optionally includes MPS Metal baseline.
func ProbeANEPrefillCompareFull(ctx context.Context, ic, oc, seq int, quick, withMPS bool) (ANEPrefillCompareResult, error) {
	aneRes, err := ProbeANEPrefillBench(ctx, ic, oc, seq, quick)
	if err != nil {
		return ANEPrefillCompareResult{OK: false, IC: ic, OC: oc, Seq: seq, ANE: aneRes, Error: err.Error()}, err
	}
	metalRes, merr := ProbeMetalPrefillBench(ctx, ic, oc, seq, quick)
	out := ANEPrefillCompareResult{
		OK:    merr == nil,
		IC:    ic,
		OC:    oc,
		Seq:   seq,
		GFLOP: PrefillMatmulGFLOP(ic, oc, seq),
		ANE:   aneRes,
		Metal: metalRes,
		Note:  "single-layer matmul proxy; metal=naive shader; metal_mps=MPS GEMM when present",
	}
	if merr != nil {
		out.Error = merr.Error()
		return out, merr
	}

	candidates := []prefillBackendTiming{
		{name: "ane", ms: aneRes.EvalMS},
		{name: "metal", ms: metalRes.EvalMS},
	}

	if withMPS && FindMetalMPSPrefillBenchBin() != "" {
		if mpsRes, mpsErr := ProbeMetalMPSPrefillBench(ctx, ic, oc, seq, quick); mpsErr == nil {
			out.MetalMPS = &mpsRes
			candidates = append(candidates, prefillBackendTiming{name: "metal_mps", ms: mpsRes.EvalMS})
		}
	}

	if aneRes.EvalMS > 0 && metalRes.EvalMS > 0 {
		out.LatencyRatioANEOverMetal = aneRes.EvalMS / metalRes.EvalMS
	}
	out.Faster, out.FasterBy = pickPrefillWinner(candidates...)
	return out, nil
}

// RunANEPrefillCompare writes compare JSON to w.
func RunANEPrefillCompare(ctx context.Context, w io.Writer, ic, oc, seq int, quick, withMPS bool) error {
	res, err := ProbeANEPrefillCompareFull(ctx, ic, oc, seq, quick, withMPS)
	enc := json.NewEncoder(w)
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}

// RunANEPrefillCompareForModel derives dims from a local model tag.
func RunANEPrefillCompareForModel(ctx context.Context, w io.Writer, preferred string, numTokens int, quick, withMPS, fullEmbed bool) error {
	ic, oc, seq, err := prefillDimsForModel(preferred, numTokens, quick, fullEmbed)
	if err != nil {
		return err
	}
	return RunANEPrefillCompare(ctx, w, ic, oc, seq, quick, withMPS)
}

// DefaultPrefillSweepSeqs is the lab SEQ grid for prefill proxy sweeps.
func DefaultPrefillSweepSeqs(quick bool) []int {
	if quick {
		return []int{128, 512, 2048}
	}
	return []int{128, 256, 512, 1024, 2048, 4096}
}

// ParsePrefillSweepSeqs parses comma-separated SEQ lengths (e.g. "128,512,2048").
func ParsePrefillSweepSeqs(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, fmt.Errorf("empty seq list")
	}
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("invalid seq %q", p)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no seq values parsed")
	}
	return out, nil
}

// ANEPrefillSweepResult is ANE vs Metal across multiple SEQ values at fixed IC×OC.
type ANEPrefillSweepResult struct {
	OK           bool                      `json:"ok"`
	IC           int                       `json:"ic"`
	OC           int                       `json:"oc"`
	Points       []ANEPrefillCompareResult `json:"points"`
	ANEWins      int                       `json:"ane_wins"`
	MetalWins    int                       `json:"metal_wins"`
	CrossoverSeq int                       `json:"crossover_seq,omitempty"`
	Note         string                    `json:"note,omitempty"`
	Error        string                    `json:"error,omitempty"`
}

func summarizePrefillSweep(ic, oc int, points []ANEPrefillCompareResult) ANEPrefillSweepResult {
	out := ANEPrefillSweepResult{
		OK:     len(points) > 0,
		IC:     ic,
		OC:     oc,
		Points: points,
		Note:   "ANE vs naive Metal vs MPS (when built); not ggml — sweep for crossover only",
	}
	prevANE := true
	for i, p := range points {
		if !p.OK {
			continue
		}
		switch p.Faster {
		case "ane":
			out.ANEWins++
		case "metal", "metal_mps":
			out.MetalWins++
			if out.CrossoverSeq == 0 && i > 0 && prevANE {
				out.CrossoverSeq = p.Seq
			}
		}
		prevANE = p.Faster == "ane"
	}
	return out
}

// ProbeANEPrefillSweep runs compare at each SEQ with fixed IC×OC.
func ProbeANEPrefillSweep(ctx context.Context, ic, oc int, seqs []int, quick, aneOnly bool) (ANEPrefillSweepResult, error) {
	if ic <= 0 || oc <= 0 {
		return ANEPrefillSweepResult{}, fmt.Errorf("ic and oc must be positive")
	}
	if len(seqs) == 0 {
		seqs = DefaultPrefillSweepSeqs(quick)
	}
	withMPS := !aneOnly && FindMetalMPSPrefillBenchBin() != ""
	var points []ANEPrefillCompareResult
	var firstErr error
	for _, seq := range seqs {
		var pt ANEPrefillCompareResult
		var err error
		if aneOnly {
			aneRes, aerr := ProbeANEPrefillBench(ctx, ic, oc, seq, quick)
			pt = ANEPrefillCompareResult{
				OK:    aerr == nil,
				IC:    ic,
				OC:    oc,
				Seq:   seq,
				GFLOP: PrefillMatmulGFLOP(ic, oc, seq),
				ANE:   aneRes,
				Faster: "ane",
				Note:  "ane_only — no Metal/MPS legs (GPU busy safe)",
			}
			err = aerr
		} else {
			pt, err = ProbeANEPrefillCompareFull(ctx, ic, oc, seq, quick, withMPS)
		}
		points = append(points, pt)
		if err != nil && firstErr == nil {
			firstErr = fmt.Errorf("seq=%d: %w", seq, err)
		}
	}
	out := summarizePrefillSweep(ic, oc, points)
	if aneOnly {
		out.Note = "ane_only sweep — safe while GPU is busy; no crossover vs MPS"
	}
	if firstErr != nil {
		out.OK = false
		out.Error = firstErr.Error()
		return out, firstErr
	}
	return out, nil
}

// RunANEPrefillSweep writes sweep JSON to w.
func RunANEPrefillSweep(ctx context.Context, w io.Writer, ic, oc int, seqs []int, quick, aneOnly bool) error {
	res, err := ProbeANEPrefillSweep(ctx, ic, oc, seqs, quick, aneOnly)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}

// RunANEPrefillSweepForModel sweeps SEQ at GGUF embedding width.
func RunANEPrefillSweepForModel(ctx context.Context, w io.Writer, preferred string, seqs []int, quick, fullEmbed, aneOnly bool) error {
	ic, oc, _, err := prefillDimsForModel(preferred, 512, quick, fullEmbed)
	if err != nil {
		return err
	}
	return RunANEPrefillSweep(ctx, w, ic, oc, seqs, quick, aneOnly)
}

// ANEPrefillHandoffResult is JSON from ane-prefill-handoff-smoke.
type ANEPrefillHandoffResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	IC           int     `json:"ic"`
	OC           int     `json:"oc"`
	Seq          int     `json:"seq"`
	SurfaceID    uint32  `json:"surface_id"`
	SurfaceBytes int     `json:"surface_bytes"`
	MetalFillMS  float64 `json:"metal_fill_ms"`
	EvalMS       float64 `json:"eval_ms"`
	TotalMS      float64 `json:"total_ms"`
	GFLOP        float64 `json:"gflop"`
	Source       string  `json:"source"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// FindANEPrefillHandoffSmokeBin locates the prefill handoff smoke binary.
func FindANEPrefillHandoffSmokeBin() string {
	return aneToolBin("ane-prefill-handoff-smoke")
}

// RunANEPrefillHandoffSmoke executes Metal→IOSurface→ANE prefill handoff timing.
func RunANEPrefillHandoffSmoke(ctx context.Context, w io.Writer, ic, oc, seq int, quick bool) error {
	bin := FindANEPrefillHandoffSmokeBin()
	out, err := runANETool(ctx, bin, anePrefillBenchArgs(ic, oc, seq, quick))
	if len(out) > 0 {
		_, _ = w.Write(out)
	}
	return err
}

// ProbeANEPrefillHandoffSmoke parses prefill handoff JSON.
func ProbeANEPrefillHandoffSmoke(ctx context.Context, ic, oc, seq int, quick bool) (ANEPrefillHandoffResult, error) {
	bin := FindANEPrefillHandoffSmokeBin()
	if bin == "" {
		return ANEPrefillHandoffResult{}, fmt.Errorf("ane-prefill-handoff-smoke not found — run ./scripts/ane_probe_build.sh")
	}
	out, err := runANETool(ctx, bin, anePrefillBenchArgs(ic, oc, seq, quick))
	if err != nil && len(out) == 0 {
		return ANEPrefillHandoffResult{}, err
	}
	var res ANEPrefillHandoffResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEPrefillHandoffResult{}, fmt.Errorf("ane-prefill-handoff-smoke json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "prefill handoff smoke returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

// RunANEPrefillHandoffForModel runs handoff at GGUF embedding width.
func RunANEPrefillHandoffForModel(ctx context.Context, w io.Writer, preferred string, numTokens int, quick, fullEmbed bool) error {
	ic, oc, seq, err := prefillDimsForModel(preferred, numTokens, quick, fullEmbed)
	if err != nil {
		return err
	}
	return RunANEPrefillHandoffSmoke(ctx, w, ic, oc, seq, quick)
}
