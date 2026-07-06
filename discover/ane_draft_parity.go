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
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
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
	MetalGoldenLegs    map[string]float64      `json:"metal_golden_legs,omitempty"`
	MetalGoldenInputCos  float64               `json:"metal_golden_input_cos,omitempty"`
	MetalGoldenHostFcCos float64               `json:"metal_golden_host_fc_cos,omitempty"`
	MetalGoldenPreOutputCos float64            `json:"metal_golden_pre_output_cos,omitempty"`
	MetalGoldenOutputCos float64               `json:"metal_golden_output_cos,omitempty"`
	GoldenCosine       float64                 `json:"golden_cosine,omitempty"`
	HandoffStride      int                     `json:"handoff_stride,omitempty"`
	HookOverheadPct    float64                 `json:"hook_overhead_pct,omitempty"`
	ShadowOverheadPct  float64                 `json:"shadow_overhead_pct,omitempty"`
	SidecarArchitecture string                 `json:"sidecar_architecture,omitempty"`
	HasDflashFcTensor  bool                    `json:"has_dflash_fc_tensor,omitempty"`
	DflashNTargetFeat  int                     `json:"dflash_n_target_features,omitempty"`
	DflashTargetLayers []uint32                `json:"dflash_target_layer_ids,omitempty"`
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

	// Lab e2e loads multi-GB GGUF cold; parent CLI ctx is often ~3min — use a dedicated budget.
	e2eBudget := 12 * time.Minute
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 17 {
			e2eBudget = 90 * time.Minute
		} else if err == nil && n >= 13 {
			e2eBudget = 35 * time.Minute
		}
	}
	e2eCtx, cancel := context.WithTimeout(context.Background(), e2eBudget)
	defer cancel()
	ab, err := ProbeANEDraftAB(e2eCtx, preferred, 0, opts.Quick, true, opts.Telemetry, driveMode, driveMetrics, convDepth, kernel, true)
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
			if draftPath, present := resolveDraftGGUFPath(entry); present {
				if arch, err := ProbeSidecarArchitecture(draftPath); err == nil {
					out.SidecarArchitecture = arch
				}
			}
			out.HasDflashFcTensor = DraftSidecarHasTensor(entry, "dflash_fc.weight")
			if nFeat, layers, ok := DraftDflashTargetMeta(entry); ok {
				out.DflashNTargetFeat = nFeat
				out.DflashTargetLayers = layers
			}
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
						ic6, oc6 := DraftANEMatmulChain6QKVDims(out.MatmulIC, out.MatmulOC)
						if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.0.attn_qkv.weight", ic6, oc6); err == nil {
							out.MatmulChain = 6
							ic7, oc7 := DraftANEMatmulChain7Blk1GateDims(out.MatmulIC, out.MatmulOC)
							if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.1.ffn_gate.weight", ic7, oc7); err == nil {
								out.MatmulChain = 7
								ic9, oc9 := DraftANEMatmulChain9Blk1UpDims(out.MatmulIC, out.MatmulOC)
								if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.1.ffn_up.weight", ic9, oc9); err == nil {
									out.MatmulChain = 9
									ic10, oc10 := DraftANEMatmulChain10Blk1DownDims(out.MatmulOC, out.MatmulIC)
									if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.1.ffn_down.weight", ic10, oc10); err == nil {
										out.MatmulChain = 10
									}
								}
							}
						}
					}
				}
			}
		} else {
			ic2, oc2 := DraftANEMatmulChain2Dims(out.MatmulIC, out.MatmulOC)
			if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "blk.0.ffn_up.weight", ic2, oc2); err == nil {
				out.MatmulChain = 2
			}
		}
		if icFc, ocFc, ok := DraftANEMatmulChain7DflashFcDims(labEntry); ok {
			if _, _, err := MaterializeANEDraftMatmulWeightFile(labEntry, "dflash_fc.weight", icFc, ocFc); err == nil &&
				IsNativeDflashDraftSidecar(labEntry) {
				out.MatmulChain = 8
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
		out.MetalGoldenLegs = ab.E2E.ANEHook.MetalGoldenLegs
		out.MetalGoldenInputCos = ab.E2E.ANEHook.MetalGoldenInputCos
		out.MetalGoldenHostFcCos = ab.E2E.ANEHook.MetalGoldenHostFcCos
		out.MetalGoldenPreOutputCos = ab.E2E.ANEHook.MetalGoldenPreOutputCos
		out.MetalGoldenOutputCos = ab.E2E.ANEHook.MetalGoldenOutputCos
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
		chainLabel := "P3"
		if out.MatmulChain >= 11 {
			chainLabel = "P10"
		} else if out.MatmulChain >= 10 {
			chainLabel = "P9"
		} else if out.MatmulChain >= 9 {
			chainLabel = "P8"
		} else if out.MatmulChain >= 8 {
			chainLabel = "P7b"
		} else if out.MatmulChain >= 7 {
			chainLabel = "P7"
		} else if out.MatmulChain >= 6 {
			chainLabel = "P6"
		} else if out.MatmulChain >= 3 {
			chainLabel = "P3"
		}
		switch driveMode {
		case "force":
			out.Note = fmt.Sprintf("matmul %s force: ANE blk.0 proxy drives draft token via tied-embed argmax", chainLabel)
		case "shadow":
			switch driveMetrics {
			case "both":
				if out.MatmulChain >= 11 {
					out.Note = fmt.Sprintf("matmul %s shadow: dflash_fc+hidden_norm+attn_q from ctx_tgt stub + token match", chainLabel)
				} else if out.MatmulChain >= 10 {
					out.Note = fmt.Sprintf("matmul %s shadow: hidden_cos on blk.1 ffn_down + token match", chainLabel)
				} else if out.MatmulChain >= 9 {
					out.Note = fmt.Sprintf("matmul %s shadow: hidden_cos on blk.1 SwiGLU + token match", chainLabel)
				} else if out.MatmulChain >= 8 {
					out.Note = fmt.Sprintf("matmul %s shadow: dflash_fc from ctx_tgt target_hidden stub + token match", chainLabel)
				} else if out.MatmulChain >= 7 {
					out.Note = fmt.Sprintf("matmul %s shadow: hidden_cos on blk.1 gate + token match (tied-embed argmax on ffn_down out)", chainLabel)
				} else {
					out.Note = fmt.Sprintf("matmul %s shadow: hidden_cos + token match (tied-embed argmax on ffn_down out)", chainLabel)
				}
			case "tokens":
				out.Note = fmt.Sprintf("matmul %s shadow: token match only (tied-embed on ffn_down out)", chainLabel)
			default:
				out.Note = fmt.Sprintf("matmul %s: gate+up SwiGLU+ffn_down; shadow uses DRIVE_METRICS=hidden", chainLabel)
			}
		}
		if out.SidecarArchitecture != "" && out.SidecarArchitecture != "dflash-draft" {
			out.Note += fmt.Sprintf("; sidecar=%s (blk.0 proxy — not native dflash_fc graph)", out.SidecarArchitecture)
		}
		if out.HasDflashFcTensor {
			out.Note += "; dflash_fc.weight present — P7b uses ctx_tgt handoff (lab stub: first ic target features)"
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
