package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
)

// ANEPrefillFFNSliceResult is JSON from ane-prefill-ffn-slice-smoke.
type ANEPrefillFFNSliceResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	Variant      string  `json:"variant,omitempty"`
	IC           int     `json:"ic"`
	OC           int     `json:"oc"`
	Hidden       int     `json:"hidden,omitempty"`
	OutCh        int     `json:"out_ch,omitempty"`
	Seq          int     `json:"seq"`
	SurfaceID    uint32  `json:"surface_id"`
	SurfaceBytes int     `json:"surface_bytes"`
	CompileMS    float64 `json:"compile_ms"`
	MapMS        float64 `json:"map_ms"`
	EvalMS       float64 `json:"eval_ms"`
	TotalMS      float64 `json:"total_ms"`
	GFLOP        float64 `json:"gflop"`
	TFLOPS       float64 `json:"tflops"`
	EvalCount    int     `json:"eval_count"`
	KernelReused bool    `json:"kernel_reused"`
	Int8Scale    float64 `json:"int8_scale,omitempty"`
	ANEMaxAbs    float64 `json:"ane_max_abs,omitempty"`
	GoldenCosine float64 `json:"golden_cosine,omitempty"`
	MaxAbsErr    float64 `json:"max_abs_err,omitempty"`
	Parity       bool    `json:"parity,omitempty"`
	Source       string  `json:"source"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// FindANEPrefillFFNSliceSmokeBin locates the in-process FFN-slice session smoke.
func FindANEPrefillFFNSliceSmokeBin() string {
	return aneToolBin("ane-prefill-ffn-slice-smoke")
}

// FindANEPrefillFFNPolicySmokeBin locates the FFN intercept policy unit smoke.
func FindANEPrefillFFNPolicySmokeBin() string {
	return aneToolBin("ane-prefill-ffn-policy-smoke")
}

// RunANEPrefillFFNSliceSmoke runs the in-process session + map + steady eval smoke.
func RunANEPrefillFFNSliceSmoke(ctx context.Context, w io.Writer, ic, oc, seq int, quick, swiglu, int8, fuseGU, w8a8, w8a8x, int8In bool, tile string) error {
	bin := FindANEPrefillFFNSliceSmokeBin()
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
		args = append(args, "--quick")
	}
	if swiglu {
		args = append(args, "--swiglu")
	}
	if int8 {
		args = append(args, "--int8")
	}
	if fuseGU {
		args = append(args, "--fuse-gu")
	}
	if int8In {
		args = append(args, "--int8-in")
	} else if w8a8x {
		args = append(args, "--w8a8-x")
	} else if w8a8 {
		args = append(args, "--w8a8")
	}
	if tile != "" {
		args = append(args, "--tile", tile)
	}
	out, err := runANETool(ctx, bin, args)
	if len(out) > 0 {
		_, _ = w.Write(out)
	}
	return err
}

// ProbeANEPrefillFFNSliceSmoke parses FFN-slice smoke JSON.
func ProbeANEPrefillFFNSliceSmoke(ctx context.Context, ic, oc, seq int, quick, swiglu, int8, fuseGU, w8a8, w8a8x, int8In bool, tile string) (ANEPrefillFFNSliceResult, error) {
	bin := FindANEPrefillFFNSliceSmokeBin()
	if bin == "" {
		return ANEPrefillFFNSliceResult{}, fmt.Errorf("ane-prefill-ffn-slice-smoke not found — run ./scripts/ane/ane_probe_build.sh")
	}
	args := []string{"--ic", strconv.Itoa(ic), "--oc", strconv.Itoa(oc), "--seq", strconv.Itoa(seq)}
	if quick {
		args = append(args, "--quick")
	}
	if swiglu {
		args = append(args, "--swiglu")
	}
	if int8 {
		args = append(args, "--int8")
	}
	if fuseGU {
		args = append(args, "--fuse-gu")
	}
	if int8In {
		args = append(args, "--int8-in")
	} else if w8a8x {
		args = append(args, "--w8a8-x")
	} else if w8a8 {
		args = append(args, "--w8a8")
	}
	if tile != "" {
		args = append(args, "--tile", tile)
	}
	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEPrefillFFNSliceResult{}, err
	}
	var res ANEPrefillFFNSliceResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEPrefillFFNSliceResult{}, fmt.Errorf("ane-prefill-ffn-slice-smoke json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "ffn-slice smoke returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}
