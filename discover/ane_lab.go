package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
)

// ANELabBinStatus reports which lab binaries are installed.
type ANELabBinStatus struct {
	ANEProbe               bool   `json:"ane_probe"`
	ANEMatmulBench           bool   `json:"ane_matmul_bench"`
	ANEDraftBench            bool   `json:"ane_draft_bench"`
	ANEIOSurfaceSmoke        bool   `json:"ane_iosurface_smoke"`
	ANEMetalHandoff          bool   `json:"ane_metal_handoff_smoke"`
	ANEDraftDaemon           bool   `json:"ane_draft_daemon"`
	ANEGGMLMapSmoke          bool   `json:"ane_ggml_map_smoke"`
	ANEPrefillBench          bool   `json:"ane_prefill_bench"`
	MetalPrefillBench        bool   `json:"metal_prefill_bench"`
	MetalMPSPrefillBench     bool   `json:"metal_mps_prefill_bench"`
	ANEPrefillHandoffSmoke   bool   `json:"ane_prefill_handoff_smoke"`
	BinDir                   string `json:"bin_dir,omitempty"`
}

// ANELabStatus is a consolidated ANE lab readiness snapshot.
type ANELabStatus struct {
	Platform     string                    `json:"platform"`
	ANERepo      string                    `json:"ane_repo"`
	ANEDraftEnv  bool                      `json:"ane_draft_env"`
	Bins         ANELabBinStatus           `json:"bins"`
	Models       ANEModelInventorySummary  `json:"models,omitempty"`
	PrefillSweep *ANEPrefillSweepResult    `json:"prefill_sweep,omitempty"`
	ModelTag     string                    `json:"model_tag,omitempty"`
	ModelIC      int                       `json:"model_ic,omitempty"`
	ModelOC      int                       `json:"model_oc,omitempty"`
	Note         string                    `json:"note,omitempty"`
}

// ANELabStatusOpts configures optional lab status probes.
type ANELabStatusOpts struct {
	WithPrefillSweep bool
	Model            string
	FullEmbed        bool
	AneOnly          bool
}

// ProbeANELabBins returns installed lab binary flags.
func ProbeANELabBins() ANELabBinStatus {
	return ANELabBinStatus{
		ANEProbe:               FindANEProbeBin() != "",
		ANEMatmulBench:           FindANEMatmulBenchBin() != "",
		ANEDraftBench:            FindANEDraftBenchBin() != "",
		ANEIOSurfaceSmoke:        FindANEIOSurfaceSmokeBin() != "",
		ANEMetalHandoff:          FindANEMetalHandoffSmokeBin() != "",
		ANEDraftDaemon:           FindANEDraftDaemonBin() != "",
		ANEGGMLMapSmoke:          FindANEGGMLMapSmokeBin() != "",
		ANEPrefillBench:          FindANEPrefillBenchBin() != "",
		MetalPrefillBench:        FindMetalPrefillBenchBin() != "",
		MetalMPSPrefillBench:     FindMetalMPSPrefillBenchBin() != "",
		ANEPrefillHandoffSmoke:   FindANEPrefillHandoffSmokeBin() != "",
		BinDir:                   HandoffBinDir(),
	}
}

// ProbeANELabStatus builds lab status; optional sweep at 256² or --model dims.
func ProbeANELabStatus(ctx context.Context, opts ANELabStatusOpts) (ANELabStatus, error) {
	out := ANELabStatus{
		Platform:    runtime.GOOS,
		ANERepo:     ANERepoPath(),
		ANEDraftEnv: ANEDraftLabEnabled(),
		Bins:        ProbeANELabBins(),
		Note:        "lab subprocesses only — production serve unchanged",
	}
	if summary, err := ProbeANEModelInventorySummary(); err == nil {
		out.Models = summary
	}
	if runtime.GOOS != "darwin" {
		return out, nil
	}
	if !opts.WithPrefillSweep {
		return out, nil
	}
	if !out.Bins.ANEPrefillBench || !out.Bins.MetalPrefillBench {
		return out, nil
	}

	ic, oc := 256, 256
	if opts.Model != "" {
		modelIC, modelOC, _, err := prefillDimsForModel(opts.Model, 512, true, opts.FullEmbed)
		if err != nil {
			return out, err
		}
		ic, oc = modelIC, modelOC
		out.ModelTag = opts.Model
		out.ModelIC = ic
		out.ModelOC = oc
	}

	sweep, err := ProbeANEPrefillSweep(ctx, ic, oc, DefaultPrefillSweepSeqs(true), true, opts.AneOnly)
	if err != nil {
		return out, err
	}
	out.PrefillSweep = &sweep
	return out, nil
}

// RunANELabStatusJSON writes lab status as JSON.
func RunANELabStatusJSON(ctx context.Context, w io.Writer, opts ANELabStatusOpts) error {
	st, err := ProbeANELabStatus(ctx, opts)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(st)
		return err
	}
	return enc.Encode(st)
}

// PrefillLabReady reports whether ANE+Metal prefill compare binaries exist.
func PrefillLabReady() bool {
	b := ProbeANELabBins()
	return b.ANEPrefillBench && b.MetalPrefillBench
}

// DoctorPrefillDetail is a one-line doctor summary for prefill compare at 256×256×128.
func DoctorPrefillDetail(ctx context.Context) string {
	if !PrefillLabReady() {
		return "prefill bench not built"
	}
	c, err := ProbeANEPrefillCompareFull(ctx, 256, 256, 128, true, FindMetalMPSPrefillBenchBin() != "")
	if err != nil {
		return "prefill compare: " + err.Error()
	}
	line := "prefill 256²×128 " + c.Faster + " " + formatFloat(c.FasterBy) + "×"
	if c.MetalMPS != nil {
		line += " (incl MPS)"
	}
	return line
}

func formatFloat(f float64) string {
	return fmt.Sprintf("%.1f", f)
}
