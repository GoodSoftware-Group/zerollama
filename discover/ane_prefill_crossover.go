package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// ANEPrefillCrossoverResult is ANE vs MPS across IC widths at fixed SEQ.
type ANEPrefillCrossoverResult struct {
	OK             bool                      `json:"ok"`
	Seq            int                       `json:"seq"`
	Widths         []int                     `json:"widths"`
	Points         []ANEPrefillCompareResult `json:"points"`
	ANEWins        int                       `json:"ane_wins"`
	MetalWins      int                       `json:"metal_wins"`
	WidthCrossover int                       `json:"width_crossover,omitempty"`
	ModelTag       string                    `json:"model_tag,omitempty"`
	ModelEmbed     int                       `json:"model_embed,omitempty"`
	Note           string                    `json:"note,omitempty"`
	Error          string                    `json:"error,omitempty"`
}

// DefaultCrossoverWidths is the lab IC grid for width crossover scans.
func DefaultCrossoverWidths(quick bool) []int {
	if quick {
		return []int{512, 640, 704, 736, 768, 896, 1024, 2048}
	}
	var out []int
	for ic := 512; ic <= 2048; ic += 128 {
		out = append(out, ic)
	}
	return out
}

// ParseCrossoverWidths parses comma-separated IC values or "from:to:step".
func ParseCrossoverWidths(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	if strings.Count(raw, ":") == 2 {
		parts := strings.Split(raw, ":")
		from, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
		to, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
		step, err3 := strconv.Atoi(strings.TrimSpace(parts[2]))
		if err1 != nil || err2 != nil || err3 != nil || from <= 0 || to <= 0 || step <= 0 || from > to {
			return nil, fmt.Errorf("invalid width range %q (want from:to:step)", raw)
		}
		var out []int
		for ic := from; ic <= to; ic += step {
			out = append(out, ic)
		}
		return out, nil
	}
	return ParsePrefillSweepSeqs(raw)
}

func summarizeWidthCrossover(seq int, widths []int, points []ANEPrefillCompareResult) ANEPrefillCrossoverResult {
	out := ANEPrefillCrossoverResult{
		OK:     len(points) > 0,
		Seq:    seq,
		Widths: widths,
		Points: points,
		Note:   "width crossover: ANE vs MPS at fixed SEQ; metal_mps counts as metal",
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
			if out.WidthCrossover == 0 && i > 0 && prevANE {
				out.WidthCrossover = p.IC
			}
		}
		prevANE = p.Faster == "ane"
	}
	return out
}

// ProbeANEPrefillWidthCrossover scans IC=OC widths at fixed SEQ.
func ProbeANEPrefillWidthCrossover(ctx context.Context, seq int, widths []int, quick, aneOnly bool, variant string) (ANEPrefillCrossoverResult, error) {
	if seq <= 0 {
		seq = 512
	}
	if len(widths) == 0 {
		widths = DefaultCrossoverWidths(quick)
	}
	if !aneOnly && FindMetalMPSPrefillBenchBin() == "" {
		return ANEPrefillCrossoverResult{}, fmt.Errorf("metal-prefill-mps-bench not found — width crossover requires MPS leg (or use --ane-only)")
	}

	var points []ANEPrefillCompareResult
	var firstErr error
	for _, ic := range widths {
		var pt ANEPrefillCompareResult
		var err error
		if aneOnly {
			aneRes, aerr := ProbeANEPrefillBenchVariant(ctx, ic, ic, seq, quick, variant)
			pt = ANEPrefillCompareResult{
				OK:      aerr == nil,
				IC:      ic,
				OC:      ic,
				Seq:     seq,
				Variant: variant,
				GFLOP:   PrefillMatmulGFLOP(ic, ic, seq),
				ANE:     aneRes,
				Faster:  "ane",
				Note:    "ane_only — no Metal/MPS legs (GPU busy safe)",
			}
			err = aerr
		} else {
			pt, err = ProbeANEPrefillCompareFull(ctx, ic, ic, seq, quick, true, variant)
		}
		points = append(points, pt)
		if err != nil && firstErr == nil {
			firstErr = fmt.Errorf("ic=%d: %w", ic, err)
		}
	}
	out := summarizeWidthCrossover(seq, widths, points)
	if aneOnly {
		out.Note = "ane_only width scan — ANE throughput vs IC; rerun without --ane-only when GPU is idle for MPS crossover"
		out.WidthCrossover = 0
		out.MetalWins = 0
		out.ANEWins = len(points)
	}
	if variant != "" {
		out.Note = strings.TrimSpace(out.Note + "; ane variant=" + variant)
	}
	if firstErr != nil {
		out.OK = false
		out.Error = firstErr.Error()
		return out, firstErr
	}
	return out, nil
}

// ProbeANEPrefillWidthCrossoverForModel scans around a model embedding width.
func ProbeANEPrefillWidthCrossoverForModel(ctx context.Context, preferred string, seq int, widths []int, quick, fullEmbed, aneOnly bool, variant string) (ANEPrefillCrossoverResult, error) {
	entries, err := ListANEModelInventory()
	if err != nil {
		return ANEPrefillCrossoverResult{}, err
	}
	entry, ok := SelectANEModel(entries, preferred)
	if !ok {
		return ANEPrefillCrossoverResult{}, fmt.Errorf("no local GGUF model matching %q", preferred)
	}
	embed := entry.EmbeddingLength
	if embed <= 0 {
		if info, gerr := ProbeANEDraftGGUF(entry.GGUFPath, ""); gerr == nil {
			embed = info.EmbeddingLength
		}
	}
	if len(widths) == 0 {
		widths = crossoverWidthsAround(embed, quick)
	}
	out, err := ProbeANEPrefillWidthCrossover(ctx, seq, widths, quick, aneOnly, variant)
	if err != nil {
		return out, err
	}
	out.ModelTag = entry.Tag
	out.ModelEmbed = embed
	return out, nil
}

func crossoverWidthsAround(embed int, quick bool) []int {
	all := DefaultCrossoverWidths(quick)
	if embed <= 0 {
		return all
	}
	var out []int
	for _, w := range all {
		if w <= embed {
			out = append(out, w)
		}
	}
	if len(out) == 0 || out[len(out)-1] < embed {
		out = append(out, embed)
	}
	return out
}

// RunANEPrefillCrossoverJSON writes width crossover JSON to w.
func RunANEPrefillCrossoverJSON(ctx context.Context, w io.Writer, preferred string, seq int, widths []int, quick, fullEmbed, aneOnly bool, variant string) error {
	var (
		res ANEPrefillCrossoverResult
		err error
	)
	if preferred != "" {
		res, err = ProbeANEPrefillWidthCrossoverForModel(ctx, preferred, seq, widths, quick, fullEmbed, aneOnly, variant)
	} else {
		res, err = ProbeANEPrefillWidthCrossover(ctx, seq, widths, quick, aneOnly, variant)
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
