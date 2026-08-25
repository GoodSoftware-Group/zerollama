package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
)

// ANEPrefillSwiGLUResult is JSON from ane-prefill-swiglu-smoke.
type ANEPrefillSwiGLUResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	Variant      string  `json:"variant,omitempty"`
	Dim          int     `json:"dim"`
	Hidden       int     `json:"hidden"`
	Seq          int     `json:"seq"`
	SurfaceID    uint32  `json:"surface_id"`
	SurfaceBytes int     `json:"surface_bytes"`
	CompileMS    float64 `json:"compile_ms"`
	EvalMS       float64 `json:"eval_ms"`
	GFLOP        float64 `json:"gflop"`
	TFLOPS       float64 `json:"tflops"`
	EvalCount    int     `json:"eval_count"`
	KernelReused bool    `json:"kernel_reused"`
	GoldenCosine float64 `json:"golden_cosine,omitempty"`
	MaxAbsErr    float64 `json:"max_abs_err,omitempty"`
	Parity       bool    `json:"parity,omitempty"`
	ANEMaxAbs    float64 `json:"ane_max_abs,omitempty"`
	Source       string  `json:"source"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// FindANEPrefillSwiGLUSmokeBin locates the fused SwiGLU session smoke.
func FindANEPrefillSwiGLUSmokeBin() string {
	return aneToolBin("ane-prefill-swiglu-smoke")
}

// RunANEPrefillSwiGLUSmoke runs fused gate+up+silu*+down parity smoke.
func RunANEPrefillSwiGLUSmoke(ctx context.Context, w io.Writer, dim, hidden, seq int, quick bool) error {
	bin := FindANEPrefillSwiGLUSmokeBin()
	args := []string{}
	if dim > 0 {
		args = append(args, "--dim", strconv.Itoa(dim))
	}
	if hidden > 0 {
		args = append(args, "--hidden", strconv.Itoa(hidden))
	}
	if seq > 0 {
		args = append(args, "--seq", strconv.Itoa(seq))
	}
	if quick {
		args = append(args, "--quick")
	}
	out, err := runANETool(ctx, bin, args)
	if len(out) > 0 {
		_, _ = w.Write(out)
	}
	return err
}

// ProbeANEPrefillSwiGLUSmoke parses SwiGLU smoke JSON.
func ProbeANEPrefillSwiGLUSmoke(ctx context.Context, dim, hidden, seq int, quick bool) (ANEPrefillSwiGLUResult, error) {
	bin := FindANEPrefillSwiGLUSmokeBin()
	if bin == "" {
		return ANEPrefillSwiGLUResult{}, fmt.Errorf("ane-prefill-swiglu-smoke not found — run ./scripts/ane/ane_probe_build.sh")
	}
	args := []string{"--dim", strconv.Itoa(dim), "--hidden", strconv.Itoa(hidden), "--seq", strconv.Itoa(seq)}
	if quick {
		args = append(args, "--quick")
	}
	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEPrefillSwiGLUResult{}, err
	}
	var res ANEPrefillSwiGLUResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEPrefillSwiGLUResult{}, fmt.Errorf("ane-prefill-swiglu-smoke json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "swiglu smoke returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}
