package discover

// ANE draft parity smoke — shadow-drive ANE vs Metal token comparison at minimal cost.
// Why not the 8-conv chain: conv proxies validate IOSurface plumbing only; they add
// latency without improving draft acceptance. Parity mode uses one real ffn_gate matmul
// (or conv-depth=1 fallback) and reports shadow token match rate as the success metric.
import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"strings"
)

// ANEDraftParityOpts configures matmul parity smoke on lab port 11435.
type ANEDraftParityOpts struct {
	Quick        bool
	UseMatmul    bool
	Telemetry    bool
	DriveMode    string // shadow (default) or force
	DriveMetrics string // hidden (default), tokens, both — matmul shadow only
}

// ANEDraftParityResult is the focused lab report for ANE vs Metal draft token parity.
type ANEDraftParityResult struct {
	OK                 bool                    `json:"ok"`
	Mode               string                  `json:"mode"`
	Tag                string                  `json:"tag,omitempty"`
	Kernel             string                  `json:"kernel,omitempty"`
	DriveMode          string                  `json:"drive_mode,omitempty"`
	DriveMetrics       string                  `json:"drive_metrics,omitempty"`
	ProxyChannels      int                     `json:"proxy_channels"`
	ProxySpatial       int                     `json:"proxy_spatial"`
	MatmulOC           int                     `json:"matmul_oc,omitempty"`
	MatmulIC           int                     `json:"matmul_ic,omitempty"`
	MatmulFullEmbd     int                     `json:"matmul_full_embd,omitempty"`
	MatmulSeq          int                     `json:"matmul_seq,omitempty"`
	MatmulChain        int                     `json:"matmul_chain,omitempty"`
	ShadowSteps        int                     `json:"shadow_steps"`
	ShadowMatches      int                     `json:"shadow_matches"`
	ShadowMatchPct     float64                 `json:"shadow_match_pct"`
	ShadowHiddenCos    float64                 `json:"shadow_hidden_cos,omitempty"`
	ShadowHiddenSteps  int                     `json:"shadow_hidden_steps,omitempty"`
	GoldenCosine       float64                 `json:"golden_cosine,omitempty"`
	HandoffStride      int                     `json:"handoff_stride,omitempty"`
	HookOverheadPct    float64                 `json:"hook_overhead_pct,omitempty"`
	ShadowOverheadPct  float64                 `json:"shadow_overhead_pct,omitempty"`
	MetalTokensPerSec  float64                 `json:"metal_tokens_per_sec,omitempty"`
	ANETokensPerSec    float64                 `json:"ane_tokens_per_sec,omitempty"`
	HookOnlyANETPS     float64                 `json:"hook_only_ane_tokens_per_sec,omitempty"`
	MetalAcceptance    float64                 `json:"metal_draft_acceptance,omitempty"`
	ANEAceptance       float64                 `json:"ane_draft_acceptance,omitempty"`
	WeightBundle       *ANEDraftWeightManifest `json:"weight_bundle,omitempty"`
	Comparison         *ANEDraftABComparison   `json:"comparison,omitempty"`
	Note               string                  `json:"note,omitempty"`
	Error              string                  `json:"error,omitempty"`
}

// ProbeANEDraftParity runs e2e dflash with shadow or force drive and matmul (or conv-depth=1) kernel.
func ProbeANEDraftParity(ctx context.Context, preferred string, opts ANEDraftParityOpts) (ANEDraftParityResult, error) {
	out := ANEDraftParityResult{
		Mode: "draft_parity_smoke",
		Note: "shadow B7: log ANE vs Metal draft token; raise shadow_match_pct before force drive",
	}
	if runtime.GOOS != "darwin" {
		return out, fmt.Errorf("ane draft parity: darwin only")
	}

	kernel := "conv"
	convDepth := 1
	if opts.UseMatmul {
		kernel = "matmul"
		convDepth = 0
	}
	out.Kernel = kernel

	driveMode := strings.TrimSpace(strings.ToLower(opts.DriveMode))
	if driveMode == "" {
		driveMode = "shadow"
	}
	out.DriveMode = driveMode

	driveMetrics := strings.TrimSpace(strings.ToLower(opts.DriveMetrics))
	if driveMetrics == "" && kernel == "matmul" && driveMode == "shadow" {
		driveMetrics = "hidden"
	}
	out.DriveMetrics = driveMetrics

	ab, err := ProbeANEDraftAB(ctx, preferred, 0, opts.Quick, true, opts.Telemetry, driveMode, driveMetrics, convDepth, kernel)
	if err != nil {
		out.Error = err.Error()
		out.Tag = ab.Tag
		return out, err
	}

	out.OK = ab.OK
	out.Tag = ab.Tag
	out.ProxyChannels = ab.ProxyChannels
	out.ProxySpatial = ab.ProxySpatial
	out.WeightBundle = ab.WeightBundle
	out.Comparison = &ab.Comparison
	out.MatmulOC = ab.ProxyChannels
	var labEntry ANEDraftEntry
	var haveEntry bool
	if entries, err := ListANEDraftInventory(); err == nil {
		if entry, ok := SelectANEDraftModel(entries, preferred); ok {
			labEntry, haveEntry = entry, true
			ic, oc, seq := DraftANEMatmulDims(entry)
			out.MatmulIC = ic
			out.MatmulOC = oc
			out.MatmulSeq = seq
			out.MatmulFullEmbd = entry.EmbeddingLength
		}
	}
	if out.MatmulSeq <= 0 {
		out.MatmulSeq = 16
	}
	out.MatmulChain = 1
	if ab.E2E != nil && ab.E2E.ANEHook.MatmulChain > 0 {
		out.MatmulChain = ab.E2E.ANEHook.MatmulChain
	} else if kernel == "matmul" && haveEntry && out.MatmulIC > 0 && out.MatmulOC > 0 {
		icUp, ocUp := DraftANEMatmulChain3UpDims(out.MatmulIC, out.MatmulOC)
		icDown, ocDown := DraftANEMatmulChain3DownDims(out.MatmulOC, out.MatmulIC)
		if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.0.ffn_down.weight", icDown, ocDown); err == nil {
			if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.0.ffn_up.weight", icUp, ocUp); err == nil {
				out.MatmulChain = 3
				ic4, oc4 := DraftANEMatmulChain4AttnGateDims(out.MatmulIC, out.MatmulOC)
				if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.0.attn_gate.weight", ic4, oc4); err == nil {
					out.MatmulChain = 4
					ic5, oc5 := DraftANEMatmulChain5SSMOutDims(out.MatmulIC, out.MatmulOC)
					if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.0.ssm_out.weight", ic5, oc5); err == nil {
						out.MatmulChain = 5
					}
				}
			}
		} else {
			ic2, oc2 := DraftANEMatmulChain2Dims(out.MatmulIC, out.MatmulOC)
			if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.0.ffn_up.weight", ic2, oc2); err == nil {
				out.MatmulChain = 2
			}
		}
	}

	if ab.E2E != nil {
		out.MetalTokensPerSec = ab.E2E.MetalOnly.TokensPerSec
		out.ANETokensPerSec = ab.E2E.ANEHook.TokensPerSec
		out.ShadowOverheadPct = ab.E2E.HookOverheadPct
		out.GoldenCosine = ab.E2E.ANEHook.GoldenCosine
		out.HandoffStride = ab.E2E.ANEHook.HandoffStride
		out.ShadowSteps = ab.E2E.ANEHook.DriveShadowSteps
		out.ShadowMatches = ab.E2E.ANEHook.DriveShadowMatches
		out.ShadowHiddenCos = ab.E2E.ANEHook.DriveShadowHiddenCos
		out.ShadowHiddenSteps = ab.E2E.ANEHook.DriveShadowHiddenSteps
		out.MetalAcceptance = ab.E2E.MetalOnly.DraftAcceptance
		out.ANEAceptance = ab.E2E.ANEHook.DraftAcceptance
		if out.ShadowSteps > 0 {
			out.ShadowMatchPct = float64(out.ShadowMatches) / float64(out.ShadowSteps) * 100
		}
		if haveEntry && driveMode != "force" {
			if hookPct, aneTPS, hookRun, herr := probeANEDraftHookOverhead(ctx, labEntry, opts.Quick, convDepth, kernel, ab.E2E.MetalOnly); herr == nil {
				out.HookOverheadPct = hookPct
				out.HookOnlyANETPS = aneTPS
				if out.HandoffStride == 0 {
					out.HandoffStride = hookRun.HandoffStride
				}
			}
		}
	}

	if kernel == "matmul" {
		switch driveMode {
		case "force":
			out.Note = "matmul P3 force: ANE blk.0 FFN proxy drives draft token via tied-embed argmax"
		case "shadow":
			switch driveMetrics {
			case "both":
				out.Note = "matmul P3 shadow: hidden_cos + token match (tied-embed argmax on ffn_down out)"
			case "tokens":
				out.Note = "matmul P3 shadow: token match only (tied-embed on ffn_down out)"
			default:
				out.Note = "matmul P3: gate+up SwiGLU+ffn_down; shadow uses DRIVE_METRICS=hidden"
			}
		}
	}

	return out, nil
}

// RunANEDraftParityJSON writes parity smoke JSON to w.
func RunANEDraftParityJSON(ctx context.Context, w io.Writer, preferred string, opts ANEDraftParityOpts) error {
	res, err := ProbeANEDraftParity(ctx, preferred, opts)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
