package discover

// ANE dflash A/B bench — micro in-process ANE step vs Metal-only dflash e2e on lab port.
// Why lab port 11435: avoid colliding with production ./zerollama serve on :11434.
import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ollama/ollama/llm"
)

// ANEDraftABResult compares ANE in-process draft-step latency vs Metal dflash e2e.
type ANEDraftABResult struct {
	OK            bool                   `json:"ok"`
	Mode          string                 `json:"mode"`
	Tag           string                 `json:"tag,omitempty"`
	SpecType      string                 `json:"spec_type,omitempty"`
	ProxyChannels int                    `json:"proxy_channels"`
	ProxySpatial  int                    `json:"proxy_spatial"`
	ConvDepth     int                    `json:"conv_depth,omitempty"`
	Micro         ANEDraftABMicro        `json:"micro"`
	E2E           *ANEDraftABE2E         `json:"e2e,omitempty"`
	Comparison    ANEDraftABComparison   `json:"comparison"`
	WeightBundle  *ANEDraftWeightManifest `json:"weight_bundle,omitempty"`
	Note          string                 `json:"note,omitempty"`
	Error         string                 `json:"error,omitempty"`
}

// ANEDraftABMicro is isolated ANE map+eval (no full Metal draft graph).
type ANEDraftABMicro struct {
	InprocessAvgEvalMS     float64 `json:"inprocess_avg_eval_ms"`
	InprocessAvgMapFillMS  float64 `json:"inprocess_avg_map_fill_ms"`
	InprocessAvgStepMS     float64 `json:"inprocess_avg_step_ms"`
	InprocessSteps         int     `json:"inprocess_steps"`
	ConvOnlyEvalMS         float64 `json:"conv_only_eval_ms"`
	KernelReused           bool    `json:"kernel_reused"`
	HasSidecarGamma        bool    `json:"has_sidecar_gamma"`
}

// ANEDraftABE2E is llama-server dflash completion + speculative stats from logs.
type ANEDraftABE2E struct {
	Port              int               `json:"port"`
	MaxTokens         int               `json:"max_tokens"`
	DriveMode         string            `json:"drive_mode,omitempty"`
	MetalOnly         ANEDraftServerRun `json:"metal_only"`
	ANEHook           ANEDraftServerRun `json:"ane_hook"`
	AcceptanceParity  bool              `json:"acceptance_parity"`
	HookOverheadPct   float64           `json:"hook_overhead_pct,omitempty"`
}

// ANEDraftServerRun is one dflash generate leg on lab llama-server.
type ANEDraftServerRun struct {
	OK                 bool    `json:"ok"`
	TokensPerSec       float64 `json:"tokens_per_sec,omitempty"`
	EvalDurationMS     float64 `json:"eval_duration_ms,omitempty"`
	EvalCount          int     `json:"eval_count,omitempty"`
	GenDrafts          uint64  `json:"gen_drafts,omitempty"`
	AccDrafts          uint64  `json:"acc_drafts,omitempty"`
	GenTokens          uint64  `json:"gen_tokens,omitempty"`
	AccTokens          uint64  `json:"acc_tokens,omitempty"`
	DraftAcceptance    float64 `json:"draft_acceptance,omitempty"`
	HandoffSteps       int     `json:"handoff_steps,omitempty"`
	HandoffStride      int     `json:"handoff_stride,omitempty"`
	Conv2Chained       bool    `json:"conv2_chained,omitempty"`
	ConvDepthCap       int     `json:"conv_depth_cap,omitempty"`
	ActiveConvDepth    int     `json:"active_conv_depth,omitempty"`
	GoldenCosine       float64 `json:"golden_cosine,omitempty"`
	GoldenSteps        int     `json:"golden_steps,omitempty"`
	DriveShadowSteps       int     `json:"drive_shadow_steps,omitempty"`
	DriveShadowMatches     int     `json:"drive_shadow_matches,omitempty"`
	DriveShadowHiddenCos   float64 `json:"drive_shadow_hidden_cos,omitempty"`
	DriveShadowHiddenSteps int     `json:"drive_shadow_hidden_steps,omitempty"`
	MatmulChain            int     `json:"matmul_chain,omitempty"`
	Error              string  `json:"error,omitempty"`
}

// ANEDraftABComparison summarizes A/B deltas.
type ANEDraftABComparison struct {
	ANEMicroStepMS       float64 `json:"ane_micro_step_ms"`
	MetalE2ETokensPerSec float64 `json:"metal_e2e_tokens_per_sec,omitempty"`
	ANEE2ETokensPerSec   float64 `json:"ane_e2e_tokens_per_sec,omitempty"`
	MetalAcceptance      float64 `json:"metal_draft_acceptance,omitempty"`
	ANEAceptance         float64 `json:"ane_draft_acceptance,omitempty"`
	AcceptanceParity     bool    `json:"acceptance_parity,omitempty"`
	HookOverheadPct      float64 `json:"hook_overhead_pct,omitempty"`
}

var (
	dflashStatsRE       = regexp.MustCompile(`statistics dflash:.*?#gen drafts = (\d+), #acc drafts = (\d+), #gen tokens = (\d+), #acc tokens = (\d+)`)
	draftAcceptRateRE   = regexp.MustCompile(`draft acceptance rate = ([0-9.]+)`)
	aneHandoffStepRE    = regexp.MustCompile(`common_ane_draft_handoff_after_decode: step=\d+ (?:ggml iosurface handoff ok|iosurface handoff ok)`)
	aneConv2ChainedRE   = regexp.MustCompile(`B6 dual conv1 chain active|B8 triple conv1 chain active|B9 quad conv1 chain active|B10 pent conv1 chain active|B11 hex conv1 chain active|B12 hept conv1 chain active|B13 oct conv1 chain active`)
	aneConvDepthRE      = regexp.MustCompile(`conv depth cap=(\d+) active_convs=(\d+)`)
	aneGoldenCosineRE   = regexp.MustCompile(`B6 golden step=\d+ mode=\w+ mse_ref_vs_ane=[0-9.eE+-]+ cosine=([0-9.-]+)`)
	aneB7ShadowRE       = regexp.MustCompile(`B7 shadow step=\d+ seq=\d+(?: handoff_tok=\d+)? ane_tok=\d+ metal_tok=\d+ match=(\d+)(?: hidden_cos=([0-9.eE+-]+))?`)
	aneMatmulChain4RE   = regexp.MustCompile(`chain4=swiglu\+down\+attn_gate|mode=matmul_chain4`)
	aneMatmulChain5RE   = regexp.MustCompile(`chain5=swiglu\+down\+attn_gate\+ssm_out|mode=matmul_chain5`)
	aneMatmulChain10RE  = regexp.MustCompile(`chain10=|P9 matmul chain blk\.1 ffn_down|mode=matmul_chain10_blk1_down`)
	aneMatmulChain9RE   = regexp.MustCompile(`chain9=|P8 matmul chain blk\.1|mode=matmul_chain9_blk1_swiglu`)
	aneMatmulChain8RE   = regexp.MustCompile(`chain8=|P7b dflash_fc|mode=matmul_chain8_dflash_fc`)
	aneMatmulChain11RE  = regexp.MustCompile(`chain11=|P10 dflash chain11 active|mode=matmul_chain11_dflash_attn_q`)
	aneMatmulChain12RE  = regexp.MustCompile(`chain12=|P11 dflash chain12 active|mode=matmul_chain12_dflash_attn_v`)
	aneMatmulChain13RE  = regexp.MustCompile(`chain13=|P12 dflash chain13 active|mode=matmul_chain13_dflash_attn_out`)
	aneMatmulChain14RE  = regexp.MustCompile(`chain14=|P13 dflash chain14 active|mode=matmul_chain14_dflash_attn_wo`)
	aneMatmulChain15RE  = regexp.MustCompile(`chain15=|P14 dflash chain15 active|mode=matmul_chain15_dflash_ffn_gate`)
	aneMatmulChain16RE  = regexp.MustCompile(`chain16=|P15 dflash chain16 active|mode=matmul_chain16_dflash_ffn_down`)
	aneMatmulChain17RE  = regexp.MustCompile(`chain17=|P16 dflash chain17 active|mode=matmul_chain17_dflash_lm_head`)
	aneMatmulChain7RE   = regexp.MustCompile(`chain7=|P7 matmul chain blk\.1|mode=matmul_chain7_blk1_gate`)
	aneMatmulChain6RE   = regexp.MustCompile(`chain6=qkv\+|mode=matmul_chain6_qkv`)
	aneMatmulChain3RE   = regexp.MustCompile(`chain3=swiglu\+down|mode=matmul_chain3`)
	aneMatmulChain2RE   = regexp.MustCompile(`chain2=gate\+silu\+up|mode=matmul_chain2`)
)

func parseB7ShadowFromLog(logText string) (steps, matches int, hiddenCosSum float64, hiddenCosN int) {
	for _, m := range aneB7ShadowRE.FindAllStringSubmatch(logText, -1) {
		if len(m) >= 2 {
			steps++
			if m[1] == "1" {
				matches++
			}
			if len(m) >= 3 && m[2] != "" {
				if v, err := strconv.ParseFloat(m[2], 64); err == nil {
					hiddenCosSum += v
					hiddenCosN++
				}
			}
		}
	}
	return steps, matches, hiddenCosSum, hiddenCosN
}

func parseGoldenCosineFromLog(logText string) (last float64, count int) {
	for _, m := range aneGoldenCosineRE.FindAllStringSubmatch(logText, -1) {
		if len(m) == 2 {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				last = v
				count++
			}
		}
	}
	return last, count
}

// parseMatmulChainFromLog reads runtime chain depth from P1 init / B6 golden lines.
func parseMatmulChainFromLog(logText string) int {
	if aneMatmulChain17RE.MatchString(logText) {
		return 17
	}
	if aneMatmulChain16RE.MatchString(logText) {
		return 16
	}
	if aneMatmulChain15RE.MatchString(logText) {
		return 15
	}
	if aneMatmulChain14RE.MatchString(logText) {
		return 14
	}
	if aneMatmulChain13RE.MatchString(logText) {
		return 13
	}
	if aneMatmulChain12RE.MatchString(logText) {
		return 12
	}
	if aneMatmulChain11RE.MatchString(logText) {
		return 11
	}
	if aneMatmulChain10RE.MatchString(logText) {
		return 10
	}
	if aneMatmulChain9RE.MatchString(logText) {
		return 9
	}
	if aneMatmulChain8RE.MatchString(logText) {
		return 8
	}
	if aneMatmulChain7RE.MatchString(logText) {
		return 7
	}
	if aneMatmulChain6RE.MatchString(logText) {
		return 6
	}
	if aneMatmulChain5RE.MatchString(logText) {
		return 5
	}
	if aneMatmulChain4RE.MatchString(logText) {
		return 4
	}
	if aneMatmulChain3RE.MatchString(logText) {
		return 3
	}
	if aneMatmulChain2RE.MatchString(logText) {
		return 2
	}
	if strings.Contains(logText, "P1 matmul kernel active") {
		return 1
	}
	return 0
}

func inferActiveConvDepthFromLog(logText string) int {
	switch {
	case strings.Contains(logText, "B13 oct conv1 chain active"):
		return 8
	case strings.Contains(logText, "B12 hept conv1 chain active"):
		return 7
	case strings.Contains(logText, "B11 hex conv1 chain active"):
		return 6
	case strings.Contains(logText, "B10 pent conv1 chain active"):
		return 5
	case strings.Contains(logText, "B9 quad conv1 chain active"):
		return 4
	case strings.Contains(logText, "B8 triple conv1 chain active"):
		return 3
	case strings.Contains(logText, "B6 dual conv1 chain active"):
		return 2
	default:
		return 0
	}
}

func countANEHandoffsFromLog(logText string) int {
	return len(aneHandoffStepRE.FindAllString(logText, -1))
}

func parseDflashStatistics(logText string) (genDrafts, accDrafts, genTokens, accTokens uint64, ok bool) {
	matches := dflashStatsRE.FindAllStringSubmatch(logText, -1)
	if len(matches) == 0 {
		return 0, 0, 0, 0, false
	}
	m := matches[len(matches)-1]
	if len(m) != 5 {
		return 0, 0, 0, 0, false
	}
	genDrafts, _ = strconv.ParseUint(m[1], 10, 64)
	accDrafts, _ = strconv.ParseUint(m[2], 10, 64)
	genTokens, _ = strconv.ParseUint(m[3], 10, 64)
	accTokens, _ = strconv.ParseUint(m[4], 10, 64)
	return genDrafts, accDrafts, genTokens, accTokens, true
}

func draftAcceptance(genTokens, accTokens uint64) float64 {
	if genTokens == 0 {
		return 0
	}
	return float64(accTokens) / float64(genTokens)
}

func aneLabPort() int {
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_LAB_PORT")); v != "" {
		if p, err := strconv.Atoi(v); err == nil && p > 0 {
			return p
		}
	}
	return 11435
}

type chatCompletionResp struct {
	EvalCount       int `json:"eval_count"`
	EvalDuration    int `json:"eval_duration"` // nanoseconds (Ollama-style)
	PromptEvalCount int `json:"prompt_eval_count"`
	Usage           struct {
		CompletionTokens int `json:"completion_tokens"`
		PromptTokens     int `json:"prompt_tokens"`
	} `json:"usage"`
	Timings struct {
		PredictedN         int     `json:"predicted_n"`
		PredictedMS        float64 `json:"predicted_ms"`
		PredictedPerSecond float64 `json:"predicted_per_second"`
		DraftN             int     `json:"draft_n"`
		DraftNAccepted     int     `json:"draft_n_accepted"`
	} `json:"timings"`
}

func fillServerRunFromCompletion(run *ANEDraftServerRun, cc chatCompletionResp, wallMS float64) {
	evalCount := cc.EvalCount
	if evalCount == 0 {
		evalCount = cc.Usage.CompletionTokens
	}
	if evalCount == 0 {
		evalCount = cc.Timings.PredictedN
	}
	run.EvalCount = evalCount

	if cc.EvalDuration > 0 {
		run.EvalDurationMS = float64(cc.EvalDuration) / 1e6
	} else if cc.Timings.PredictedMS > 0 {
		run.EvalDurationMS = cc.Timings.PredictedMS
	} else if wallMS > 0 {
		run.EvalDurationMS = wallMS
	}

	if cc.Timings.PredictedPerSecond > 0 {
		run.TokensPerSec = cc.Timings.PredictedPerSecond
	} else if run.EvalDurationMS > 0 && evalCount > 0 {
		run.TokensPerSec = float64(evalCount) / (run.EvalDurationMS / 1000)
	}

	if cc.Timings.DraftN > 0 {
		run.GenTokens = uint64(cc.Timings.DraftN)
		run.AccTokens = uint64(cc.Timings.DraftNAccepted)
		run.DraftAcceptance = draftAcceptance(run.GenTokens, run.AccTokens)
	}
}

// ProbeANEDraftAB runs micro ANE bench and optional llama-server Metal vs ANE-hook e2e.
// driveMode: "" (off), "shadow", or "force" when runE2E && ANE hook leg uses B7 drive.
// convDepth: cap active conv kernels (0 = use full manifest chain).
// kernel: "conv" (default) or "matmul" for blk.0 ffn_gate matmul proxy.
// skipMicro: when true, skip ane-inprocess-smoke (parity e2e only — faster lab runs).
func ProbeANEDraftAB(ctx context.Context, preferred string, steps int, quick, runE2E, e2eTelemetry bool, driveMode, driveMetrics string, convDepth int, kernel string, skipMicro bool) (ANEDraftABResult, error) {
	out := ANEDraftABResult{
		Mode:      "draft_ab_smoke",
		ConvDepth: convDepth,
		Note:      "B4: micro ANE in-process step vs Metal dflash e2e; hook is telemetry-only until ANE drives draft tokens",
	}
	if runtime.GOOS != "darwin" {
		return out, fmt.Errorf("ane draft ab: darwin only")
	}

	entries, err := ListANEDraftInventory()
	if err != nil {
		return out, err
	}
	entry, ok := SelectANEDraftModel(entries, preferred)
	if !ok {
		return out, fmt.Errorf("no ANE draft target in local inventory")
	}

	out.Tag = entry.Tag
	out.SpecType = entry.SpecType
	out.ProxyChannels = entry.ProxyChannels
	out.ProxySpatial = entry.ProxySpatial

	if steps <= 0 {
		if quick {
			steps = 5
		} else {
			steps = 10
		}
	}

	bundle, _, berr := MaterializeANEDraftWeightBundleWithDrive(entry, ANEDraftNeedsDriveHeadWithMetrics(kernel, driveMode, driveMetrics))
	if berr == nil {
		out.WeightBundle = &bundle
	}

	if skipMicro {
		out.Note = "micro inprocess smoke skipped (e2e-only parity)"
	} else {
		inproc, err := ProbeANEInprocessSmoke(ctx, preferred, steps, quick)
		if err != nil {
			if !(runE2E && strings.Contains(err.Error(), "abort trap")) {
				out.Error = err.Error()
				return out, err
			}
			out.Note = "micro inprocess smoke aborted on Metal teardown (e2e leg still runs)"
		} else {
			out.Micro.InprocessAvgEvalMS = inproc.AvgEvalMS
			out.Micro.InprocessAvgMapFillMS = inproc.AvgMapFillMS
			out.Micro.InprocessAvgStepMS = inproc.AvgEvalMS + inproc.AvgMapFillMS
			out.Micro.InprocessSteps = len(inproc.Steps)
			out.Micro.KernelReused = inproc.KernelReused
		}
	}
	out.Micro.HasSidecarGamma = bundle.GammaWeightPath() != ""

	ch, sp := entry.ProxyChannels, entry.ProxySpatial
	if ch <= 0 {
		ch, sp = DraftANEProxyDims(entry.EmbeddingLength)
	}
	if conv, cerr := ProbeANEDraftBenchDims(ctx, ch, sp, quick); cerr == nil {
		out.Micro.ConvOnlyEvalMS = conv.EvalMS
	}

	out.Comparison.ANEMicroStepMS = out.Micro.InprocessAvgStepMS

	if runE2E {
		e2e, e2eErr := probeANEDraftE2E(ctx, entry, quick, e2eTelemetry, driveMode, driveMetrics, convDepth, kernel)
		if e2eErr != nil {
			out.Error = e2eErr.Error()
			return out, e2eErr
		}
		out.E2E = &e2e
		out.Comparison.MetalE2ETokensPerSec = e2e.MetalOnly.TokensPerSec
		out.Comparison.ANEE2ETokensPerSec = e2e.ANEHook.TokensPerSec
		out.Comparison.MetalAcceptance = e2e.MetalOnly.DraftAcceptance
		out.Comparison.ANEAceptance = e2e.ANEHook.DraftAcceptance
		out.Comparison.AcceptanceParity = e2e.AcceptanceParity
		out.Comparison.HookOverheadPct = e2e.HookOverheadPct
	}

	out.OK = true
	return out, nil
}

func findLlamaServerForANEDraft() (string, error) {
	// Prefer unified build (vendor pin or LLAMA_CPP_ROOT sibling); IOSurface handoff requires patched ggml-metal.
	return llm.FindLlamaServer()
}

func probeANEDraftE2E(ctx context.Context, entry ANEDraftEntry, quick, telemetry bool, driveMode, driveMetrics string, convDepth int, kernel string) (ANEDraftABE2E, error) {
	serverBin, err := findLlamaServerForANEDraft()
	if err != nil {
		return ANEDraftABE2E{}, fmt.Errorf("llama-server not found: %w (run ./scripts/build_llama_server.sh)", err)
	}

	basePath := strings.TrimSpace(entry.BaseGGUF)
	draftPath, present := resolveDraftGGUFPath(entry)
	if basePath == "" || !present {
		return ANEDraftABE2E{}, fmt.Errorf("base or draft GGUF missing for %s", entry.Tag)
	}
	_ = draftPath

	port := aneLabPort()
	maxTokens := 32
	if quick {
		maxTokens = 16
	}

	metalRun, err := runDflashServerLeg(ctx, serverBin, entry, port, maxTokens, quick, false, false, "", "", 0, "")
	if err != nil {
		return ANEDraftABE2E{}, err
	}
	aneRun, err := runDflashServerLeg(ctx, serverBin, entry, port, maxTokens, quick, true, telemetry, driveMode, driveMetrics, convDepth, kernel)
	if err != nil {
		return ANEDraftABE2E{}, err
	}

	e2e := ANEDraftABE2E{
		Port:             port,
		MaxTokens:        maxTokens,
		DriveMode:        driveMode,
		MetalOnly:        metalRun,
		ANEHook:          aneRun,
		AcceptanceParity: acceptanceParity(metalRun, aneRun),
	}
	if metalRun.TokensPerSec > 0 && aneRun.TokensPerSec > 0 {
		e2e.HookOverheadPct = calcHookOverheadPct(metalRun.TokensPerSec, aneRun.TokensPerSec)
	}
	return e2e, nil
}

func calcHookOverheadPct(metalTPS, aneTPS float64) float64 {
	if metalTPS <= 0 || aneTPS <= 0 {
		return 0
	}
	return (metalTPS - aneTPS) / metalTPS * 100
}

// probeANEDraftHookOverhead runs ANE hook e2e without B7 shadow drive (handoff+eval only).
func probeANEDraftHookOverhead(ctx context.Context, entry ANEDraftEntry, quick bool, convDepth int, kernel string, metalRun ANEDraftServerRun) (overheadPct, aneTPS float64, run ANEDraftServerRun, err error) {
	serverBin, err := findLlamaServerForANEDraft()
	if err != nil {
		return 0, 0, run, err
	}
	port := aneLabPort()
	maxTokens := 32
	if quick {
		maxTokens = 16
	}
	aneRun, err := runDflashServerLeg(ctx, serverBin, entry, port, maxTokens, quick, true, false, "", "", convDepth, kernel)
	if err != nil {
		return 0, 0, run, err
	}
	if metalRun.TokensPerSec > 0 && aneRun.TokensPerSec > 0 {
		overheadPct = calcHookOverheadPct(metalRun.TokensPerSec, aneRun.TokensPerSec)
	}
	return overheadPct, aneRun.TokensPerSec, aneRun, nil
}

func acceptanceClose(a, b float64) bool {
	if a == 0 && b == 0 {
		return true
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff < 0.02
}

// acceptanceParity compares draft acceptance when both legs report token-level dflash stats.
// Metal-only legs often omit timings.draft_n on short runs; treat that as incomparable (true).
func acceptanceParity(metal, ane ANEDraftServerRun) bool {
	if metal.GenTokens == 0 || ane.GenTokens == 0 {
		return true
	}
	return acceptanceClose(metal.DraftAcceptance, ane.DraftAcceptance)
}

func aneDraftLabNumCtx(entry ANEDraftEntry, quick bool) int {
	if !quick {
		return 0
	}
	if entry.NumCtx > 32768 {
		return 8192
	}
	if entry.EmbeddingLength >= 4096 {
		return 8192
	}
	tag := strings.ToLower(entry.Tag)
	if strings.Contains(tag, "256k") || strings.Contains(tag, "27b") {
		return 8192
	}
	return 0
}

func aneDraftServerLegTimeout(aneHook bool) time.Duration {
	if !aneHook {
		return 3 * time.Minute
	}
	matmulChain := 0
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			matmulChain = n
		}
	}
	switch {
	case matmulChain >= 17:
		return 30 * time.Minute
	case matmulChain >= 13:
		return 15 * time.Minute
	default:
		return 8 * time.Minute
	}
}

func runDflashServerLeg(ctx context.Context, serverBin string, entry ANEDraftEntry, port, maxTokens int, quick, aneHook, telemetry bool, driveMode, driveMetrics string, convDepth int, kernel string) (ANEDraftServerRun, error) {
	run := ANEDraftServerRun{}
	if ctx == nil {
		ctx = context.Background()
	}

	baseGGUF := strings.TrimSpace(entry.BaseGGUF)
	draftGGUF, present := resolveDraftGGUFPath(entry)
	if baseGGUF == "" || !present {
		return run, fmt.Errorf("base or draft GGUF missing for %s", entry.Tag)
	}

	logFile, err := os.CreateTemp("", "zerollama-ane-ab-*.log")
	if err != nil {
		return run, err
	}
	logPath := logFile.Name()
	_ = logFile.Close()
	if os.Getenv("ZEROLLAMA_ANE_KEEP_AB_LOG") == "" {
		defer os.Remove(logPath)
	}

	legTimeout := aneDraftServerLegTimeout(aneHook)
	runCtx, cancel := context.WithTimeout(ctx, legTimeout)
	defer cancel()

	args := []string{
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"-m", baseGGUF,
		"--spec-type", "dflash",
		"--spec-draft-model", draftGGUF,
		"--spec-draft-n-max", "4",
		"-ngl", "99",
	}
	if nctx := aneDraftLabNumCtx(entry, quick); nctx > 0 {
		args = append(args, "-c", strconv.Itoa(nctx))
	}

	cmd := exec.CommandContext(runCtx, serverBin, args...)
	cmd.Dir = filepath.Dir(serverBin)
	shellEnv := os.Environ()
	cmd.Env = shellEnv
	if aneHook {
		kernel = strings.ToLower(strings.TrimSpace(kernel))
		if kernel == "" {
			kernel = "conv"
		}
		cmd.Env = stripANEDraftEnv(cmd.Env)
		ch := entry.ProxyChannels
		if ch <= 0 {
			ch, _ = DraftANEProxyDims(entry.EmbeddingLength)
		}
		sp := entry.ProxySpatial
		if sp <= 0 {
			sp = 16
		}
		matmulIC, matmulOC, matmulSeq := DraftANEMatmulDims(entry)
		matmulChain := 0

		cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT=1")
		needDriveHead := ANEDraftNeedsDriveHeadWithMetrics(kernel, driveMode, driveMetrics)
		manifest, _, _ := MaterializeANEDraftWeightBundleWithDrive(entry, needDriveHead)
		if kernel == "matmul" {
			cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
				return strings.HasPrefix(k, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE")
			})
			cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_KERNEL", "matmul")
			cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_SEQ", strconv.Itoa(matmulSeq))
			cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(matmulOC))
			sp = matmulSeq
			forceChain := 0
			if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN")); v != "" {
				if n, err := strconv.Atoi(v); err == nil && n > 0 {
					forceChain = n
				}
			}
			matmulChain = 1
			useDflashFc := false
			if forceChain == 8 || forceChain == 11 || forceChain == 12 || forceChain == 13 || forceChain == 14 || forceChain == 15 || forceChain == 16 || forceChain == 17 || (forceChain == 0 && IsNativeDflashDraftSidecar(entry)) {
				if icFc, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok {
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "dflash_fc.weight", icFc, ocFc); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "8")
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocFc))
						matmulChain = 8
						ch = icFc
						useDflashFc = true
						for k, v := range ExportDflashTargetMetaEnv(entry) {
							cmd.Env = upsertEnv(cmd.Env, k, v)
						}
						// Native dflash_fc is [n_target_features × n_embd]; ANE uses a top-left slice.
						// Run full matmul on host so B7 shadow matches Metal draft (export row @ full W).
						if fullIc, fullOc, okFull := DraftANEMatmulChain7DflashFcNativeDims(entry); okFull && fullIc > icFc {
							if wFull, _, err := MaterializeANEDraftMatmulWeightFile(entry, "dflash_fc.weight", fullIc, fullOc); err == nil && wFull != "" {
								cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DFLASH_FC_HOST", "1")
								cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DFLASH_FC_FULL_IC", strconv.Itoa(fullIc))
								cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DFLASH_FC_FULL_WEIGHT_FILE", wFull)
							}
						}
					}
				}
			}
			if !useDflashFc && forceChain != 8 && forceChain != 11 && forceChain != 12 && forceChain != 13 && forceChain != 14 && forceChain != 15 && forceChain != 16 && forceChain != 17 {
			if wpath, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", matmulIC, matmulOC); err == nil && wpath != "" {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE="+wpath)
			}
			icUp, ocUp := DraftANEMatmulChain3UpDims(matmulIC, matmulOC)
			icDown, ocDown := DraftANEMatmulChain3DownDims(matmulOC, matmulIC)
			chain3 := false
			if forceChain != 1 && forceChain != 2 {
				if w3, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_down.weight", icDown, ocDown); err == nil && w3 != "" {
					if w2, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_up.weight", icUp, ocUp); err == nil && w2 != "" {
						cmd.Env = append(cmd.Env,
							"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2="+w2,
							"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3="+w3,
							"ZEROLLAMA_ANE_DRAFT_MATMUL_OC2="+strconv.Itoa(ocUp),
							"ZEROLLAMA_ANE_DRAFT_MATMUL_OC3="+strconv.Itoa(ocDown),
						)
						chain3 = true
						matmulChain = 3
						if forceChain != 3 && forceChain != 4 {
							ic4, oc4 := DraftANEMatmulChain4AttnGateDims(matmulIC, matmulOC)
							if w4, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.attn_gate.weight", ic4, oc4); err == nil && w4 != "" {
								cmd.Env = append(cmd.Env,
									"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4="+w4,
									"ZEROLLAMA_ANE_DRAFT_MATMUL_OC4="+strconv.Itoa(oc4),
								)
								matmulChain = 4
								ic5, oc5 := DraftANEMatmulChain5SSMOutDims(matmulIC, matmulOC)
								if w5, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ssm_out.weight", ic5, oc5); err == nil && w5 != "" {
									cmd.Env = append(cmd.Env,
										"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5="+w5,
										"ZEROLLAMA_ANE_DRAFT_MATMUL_OC5="+strconv.Itoa(oc5),
									)
									matmulChain = 5
									if forceChain != 5 {
										ic6, oc6 := DraftANEMatmulChain6QKVDims(matmulIC, matmulOC)
										if w6, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.attn_qkv.weight", ic6, oc6); err == nil && w6 != "" {
											cmd.Env = append(cmd.Env,
												"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6="+w6,
												"ZEROLLAMA_ANE_DRAFT_MATMUL_OC6="+strconv.Itoa(oc6),
											)
											matmulChain = 6
											if forceChain != 6 {
												ic7, oc7 := DraftANEMatmulChain7Blk1GateDims(matmulIC, matmulOC)
												if w7, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.1.ffn_gate.weight", ic7, oc7); err == nil && w7 != "" {
													cmd.Env = append(cmd.Env,
														"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7="+w7,
														"ZEROLLAMA_ANE_DRAFT_MATMUL_OC7="+strconv.Itoa(oc7),
													)
													matmulChain = 7
													if forceChain == 0 || forceChain == 9 || forceChain == 10 {
														ic9, oc9 := DraftANEMatmulChain9Blk1UpDims(matmulIC, matmulOC)
														if w8, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.1.ffn_up.weight", ic9, oc9); err == nil && w8 != "" {
															cmd.Env = append(cmd.Env,
																"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8="+w8,
																"ZEROLLAMA_ANE_DRAFT_MATMUL_OC9="+strconv.Itoa(oc9),
															)
															matmulChain = 9
															if forceChain == 0 || forceChain == 10 { // not 9 — lab chain9 stops at blk.1 SwiGLU
																ic10, oc10 := DraftANEMatmulChain10Blk1DownDims(matmulOC, matmulIC)
																if w9, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.1.ffn_down.weight", ic10, oc10); err == nil && w9 != "" {
																	cmd.Env = append(cmd.Env,
																		"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9="+w9,
																		"ZEROLLAMA_ANE_DRAFT_MATMUL_OC10="+strconv.Itoa(oc10),
																	)
																	matmulChain = 10
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
						switch matmulChain {
						case 10:
							cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "10")
						case 9:
							cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "9")
						case 7:
							cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "7")
						case 6:
							cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "6")
						case 5:
							cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "5")
						case 4:
							cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "4")
						default:
							cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "3")
						}
					}
				}
			}
			if forceChain == 4 {
				ic4, oc4 := DraftANEMatmulChain4AttnGateDims(matmulIC, matmulOC)
				if w4, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.attn_gate.weight", ic4, oc4); err == nil && w4 != "" {
					cmd.Env = append(cmd.Env,
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4="+w4,
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC4="+strconv.Itoa(oc4),
					)
				}
			}
			if !chain3 && forceChain != 1 {
				ic2, oc2 := DraftANEMatmulChain2Dims(matmulIC, matmulOC)
				if w2, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_up.weight", ic2, oc2); err == nil && w2 != "" {
					cmd.Env = append(cmd.Env,
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2="+w2,
						"ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=2",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC2="+strconv.Itoa(oc2),
					)
					matmulChain = 2
				}
			}
			}
			if forceChain == 1 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "1")
				matmulChain = 1
			} else if forceChain == 2 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "2")
				matmulChain = 2
			} else if forceChain == 3 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "3")
				matmulChain = 3
			} else if forceChain == 4 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "4")
				matmulChain = 4
			} else if forceChain == 5 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "5")
				matmulChain = 5
			} else if forceChain == 6 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "6")
				matmulChain = 6
			} else if forceChain == 7 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "7")
				matmulChain = 7
			} else if forceChain == 8 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "8")
				matmulChain = 8
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC7",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC9", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				if !useDflashFc {
					// Lab plumbing: gate top-left slice stands in for dflash_fc when sidecar lacks tensor.
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
					}
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 17 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "17")
				matmulChain = 17
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC9", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				fcOut := matmulOC
				if useDflashFc {
					if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
						fcOut = ocFc
					}
				} else {
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
						fcOut = ocEmbd
					}
				}
				icQ, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
				attnQTensor := ResolveChain11AttnQTensor(entry)
				if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnQTensor, icQ, ocQ); err == nil && wq != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", wq)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", strconv.Itoa(ocQ))
				}
				icK, ocK := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnKTensor := ResolveChain12AttnKTensor(entry)
				if wk, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnKTensor, icK, ocK); err == nil && wk != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3", wk)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", strconv.Itoa(ocK))
				}
				icV, ocV := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnVTensor := ResolveChain12AttnVTensor(entry)
				if wv, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnVTensor, icV, ocV); err == nil && wv != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", wv)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", strconv.Itoa(ocV))
				}
				icWo, ocWo := DraftANEMatmulChain14AttnWoDimsForChain(entry, fcOut, matmulChain)
				attnWoTensor := ResolveChain14AttnWoTensor(entry)
				if wwo, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnWoTensor, icWo, ocWo); err == nil && wwo != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5", wwo)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", strconv.Itoa(ocWo))
				}
				icGate, ocGate := DraftANEMatmulChain15FFNGateDims(entry, fcOut)
				ffnGateTensor := ResolveChain15FFNGateTensor(entry)
				if wg, _, err := MaterializeANEDraftMatmulWeightFile(entry, ffnGateTensor, icGate, ocGate); err == nil && wg != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6", wg)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", strconv.Itoa(ocGate))
				}
				icUp, ocUp := DraftANEMatmulChain16FFNUpDims(entry, fcOut)
				ffnUpTensor := ResolveChain16FFNUpTensor(entry)
				if wu, _, err := MaterializeANEDraftMatmulWeightFile(entry, ffnUpTensor, icUp, ocUp); err == nil && wu != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7", wu)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC7", strconv.Itoa(ocUp))
				}
				icDown, ocDown := DraftANEMatmulChain16FFNDownDims(entry, fcOut)
				ffnDownTensor := ResolveChain16FFNDownTensor(entry)
				if wd, _, err := MaterializeANEDraftMatmulWeightFile(entry, ffnDownTensor, icDown, ocDown); err == nil && wd != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8", wd)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC8", strconv.Itoa(ocDown))
				}
				if head, _, err := MaterializeANEDraftDriveHead(entry); err == nil {
					for k, v := range ExportDriveEnvForHead(head) {
						cmd.Env = upsertEnv(cmd.Env, k, v)
					}
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 16 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "16")
				matmulChain = 16
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC9", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				fcOut := matmulOC
				if useDflashFc {
					if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
						fcOut = ocFc
					}
				} else {
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
						fcOut = ocEmbd
					}
				}
				icQ, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
				attnQTensor := ResolveChain11AttnQTensor(entry)
				if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnQTensor, icQ, ocQ); err == nil && wq != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", wq)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", strconv.Itoa(ocQ))
				}
				icK, ocK := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnKTensor := ResolveChain12AttnKTensor(entry)
				if wk, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnKTensor, icK, ocK); err == nil && wk != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3", wk)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", strconv.Itoa(ocK))
				}
				icV, ocV := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnVTensor := ResolveChain12AttnVTensor(entry)
				if wv, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnVTensor, icV, ocV); err == nil && wv != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", wv)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", strconv.Itoa(ocV))
				}
				icWo, ocWo := DraftANEMatmulChain14AttnWoDimsForChain(entry, fcOut, matmulChain)
				attnWoTensor := ResolveChain14AttnWoTensor(entry)
				if wwo, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnWoTensor, icWo, ocWo); err == nil && wwo != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5", wwo)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", strconv.Itoa(ocWo))
				}
				icGate, ocGate := DraftANEMatmulChain15FFNGateDims(entry, fcOut)
				ffnGateTensor := ResolveChain15FFNGateTensor(entry)
				if wg, _, err := MaterializeANEDraftMatmulWeightFile(entry, ffnGateTensor, icGate, ocGate); err == nil && wg != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6", wg)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", strconv.Itoa(ocGate))
				}
				icUp, ocUp := DraftANEMatmulChain16FFNUpDims(entry, fcOut)
				ffnUpTensor := ResolveChain16FFNUpTensor(entry)
				if wu, _, err := MaterializeANEDraftMatmulWeightFile(entry, ffnUpTensor, icUp, ocUp); err == nil && wu != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7", wu)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC7", strconv.Itoa(ocUp))
				}
				icDown, ocDown := DraftANEMatmulChain16FFNDownDims(entry, fcOut)
				ffnDownTensor := ResolveChain16FFNDownTensor(entry)
				if wd, _, err := MaterializeANEDraftMatmulWeightFile(entry, ffnDownTensor, icDown, ocDown); err == nil && wd != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8", wd)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC8", strconv.Itoa(ocDown))
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 15 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "15")
				matmulChain = 15
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC7", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC9",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				fcOut := matmulOC
				if useDflashFc {
					if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
						fcOut = ocFc
					}
				} else {
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
						fcOut = ocEmbd
					}
				}
				icQ, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
				attnQTensor := ResolveChain11AttnQTensor(entry)
				if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnQTensor, icQ, ocQ); err == nil && wq != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", wq)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", strconv.Itoa(ocQ))
				}
				icK, ocK := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnKTensor := ResolveChain12AttnKTensor(entry)
				if wk, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnKTensor, icK, ocK); err == nil && wk != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3", wk)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", strconv.Itoa(ocK))
				}
				icV, ocV := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnVTensor := ResolveChain12AttnVTensor(entry)
				if wv, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnVTensor, icV, ocV); err == nil && wv != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", wv)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", strconv.Itoa(ocV))
				}
				icWo, ocWo := DraftANEMatmulChain14AttnWoDimsForChain(entry, fcOut, matmulChain)
				attnWoTensor := ResolveChain14AttnWoTensor(entry)
				if wwo, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnWoTensor, icWo, ocWo); err == nil && wwo != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5", wwo)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", strconv.Itoa(ocWo))
				}
				icGate, ocGate := DraftANEMatmulChain15FFNGateDims(entry, fcOut)
				ffnGateTensor := ResolveChain15FFNGateTensor(entry)
				if wg, _, err := MaterializeANEDraftMatmulWeightFile(entry, ffnGateTensor, icGate, ocGate); err == nil && wg != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6", wg)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", strconv.Itoa(ocGate))
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 14 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "14")
				matmulChain = 14
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC7",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC9", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				fcOut := matmulOC
				if useDflashFc {
					if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
						fcOut = ocFc
					}
				} else {
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
						fcOut = ocEmbd
					}
				}
				icQ, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
				attnQTensor := ResolveChain11AttnQTensor(entry)
				if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnQTensor, icQ, ocQ); err == nil && wq != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", wq)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", strconv.Itoa(ocQ))
				}
				icK, ocK := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnKTensor := ResolveChain12AttnKTensor(entry)
				if wk, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnKTensor, icK, ocK); err == nil && wk != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3", wk)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", strconv.Itoa(ocK))
				}
				icV, ocV := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnVTensor := ResolveChain12AttnVTensor(entry)
				if wv, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnVTensor, icV, ocV); err == nil && wv != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", wv)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", strconv.Itoa(ocV))
				}
				icWo, ocWo := DraftANEMatmulChain14AttnWoDimsForChain(entry, fcOut, matmulChain)
				attnWoTensor := ResolveChain14AttnWoTensor(entry)
				if wwo, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnWoTensor, icWo, ocWo); err == nil && wwo != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5", wwo)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", strconv.Itoa(ocWo))
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 13 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "13")
				matmulChain = 13
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC6",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC7", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC9",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				fcOut := matmulOC
				if useDflashFc {
					if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
						fcOut = ocFc
					}
				} else {
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
						fcOut = ocEmbd
					}
				}
				icQ, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
				attnQTensor := ResolveChain11AttnQTensor(entry)
				if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnQTensor, icQ, ocQ); err == nil && wq != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", wq)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", strconv.Itoa(ocQ))
				}
				icK, ocK := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnKTensor := ResolveChain12AttnKTensor(entry)
				if wk, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnKTensor, icK, ocK); err == nil && wk != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3", wk)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", strconv.Itoa(ocK))
				}
				icV, ocV := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnVTensor := ResolveChain12AttnVTensor(entry)
				if wv, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnVTensor, icV, ocV); err == nil && wv != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", wv)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", strconv.Itoa(ocV))
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 12 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "12")
				matmulChain = 12
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC6",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC7", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC9",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				fcOut := matmulOC
				if useDflashFc {
					if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
						fcOut = ocFc
					}
				} else {
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
						fcOut = ocEmbd
					}
				}
				icQ, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
				attnQTensor := ResolveChain11AttnQTensor(entry)
				if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnQTensor, icQ, ocQ); err == nil && wq != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", wq)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", strconv.Itoa(ocQ))
				}
				icK, ocK := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnKTensor := ResolveChain12AttnKTensor(entry)
				if wk, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnKTensor, icK, ocK); err == nil && wk != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3", wk)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", strconv.Itoa(ocK))
				}
				icV, ocV := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
				attnVTensor := ResolveChain12AttnVTensor(entry)
				if wv, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnVTensor, icV, ocV); err == nil && wv != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", wv)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", strconv.Itoa(ocV))
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 11 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "11")
				matmulChain = 11
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					switch k {
					case "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7",
						"ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8", "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9":
						return true
					case "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC3",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC5",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC7",
						"ZEROLLAMA_ANE_DRAFT_MATMUL_OC9", "ZEROLLAMA_ANE_DRAFT_MATMUL_OC10":
						return true
					}
					return false
				})
				fcOut := matmulOC
				if useDflashFc {
					if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
						fcOut = ocFc
					}
				} else {
					icFeat, ocEmbd := matmulIC, matmulOC
					if wfc, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", icFeat, ocEmbd); err == nil && wfc != "" {
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE", wfc)
						cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC", strconv.Itoa(ocEmbd))
						ch = icFeat
						fcOut = ocEmbd
					}
				}
				icQ, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
				attnQTensor := ResolveChain11AttnQTensor(entry)
				if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, attnQTensor, icQ, ocQ); err == nil && wq != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2", wq)
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", strconv.Itoa(ocQ))
				}
				for k, v := range ExportDflashTargetMetaEnv(entry) {
					cmd.Env = upsertEnv(cmd.Env, k, v)
				}
				if len(ExportDflashTargetMetaEnv(entry)) == 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", "1")
				}
			} else if forceChain == 9 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "9")
				matmulChain = 9
				cmd.Env = filterANEDraftEnv(cmd.Env, func(k string) bool {
					return k == "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9" || k == "ZEROLLAMA_ANE_DRAFT_MATMUL_OC10"
				})
			} else if forceChain == 10 {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN", "10")
				matmulChain = 10
			}
			if !useDflashFc && forceChain != 8 && forceChain != 11 && forceChain != 12 && forceChain != 13 && forceChain != 14 && forceChain != 15 && forceChain != 16 && forceChain != 17 {
				ch = matmulIC
			}
		} else {
			applyConvManifestEnv(&cmd.Env, manifest, convDepth)
		}
		dflashGammaWired := false
		if kernel == "matmul" && (matmulChain == 8 || matmulChain == 11 || matmulChain == 12 || matmulChain == 13 || matmulChain == 14 || matmulChain == 15 || matmulChain == 16 || matmulChain == 17) && IsNativeDflashDraftSidecar(entry) {
			normTensor := ResolveChain11HiddenNormTensor(entry)
			gammaDim := DraftANEDflashHiddenNormDim(entry)
			if gp, _, err := MaterializeANEDraftNormGammaFile(entry, normTensor, gammaDim); err == nil && gp != "" {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_GAMMA_FILE", gp)
				dflashGammaWired = true
			}
		}
		if !dflashGammaWired {
			if gamma := manifest.GammaWeightPath(); gamma != "" {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_GAMMA_FILE="+gamma)
			}
		}
		if kernel == "matmul" && (matmulChain == 15 || matmulChain == 16 || matmulChain == 17) && IsNativeDflashDraftSidecar(entry) {
			fcOut := matmulOC
			if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
				fcOut = ocFc
			}
			postNormTensor := ResolveChain15AttnPostNormTensor(entry)
			postNormDim := DraftANEDflashAttnPostNormDim(entry, fcOut)
			if pp, _, err := MaterializeANEDraftNormGammaFile(entry, postNormTensor, postNormDim); err == nil && pp != "" {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ATTN_POST_NORM_FILE", pp)
			}
		}
		if kernel == "matmul" && matmulChain >= 13 && IsNativeDflashDraftSidecar(entry) {
			fcOut := matmulOC
			if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
				fcOut = ocFc
			}
			_, ocAttn := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
			if ap, _, err := MaterializeANEDraftNormGammaFile(entry, ResolveChain13AttnNormTensor(entry), fcOut); err == nil && ap != "" {
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ATTN_NORM_FILE", ap)
			}
			if meta, err := DraftANEDraftAttnHeadMeta(entry); err == nil && meta.HeadDim > 0 {
				nHeadKV := meta.NHeadKV
				headDim := meta.HeadDim
				nHeadQ := meta.NHead
				if !DraftANEDraftAttnUseFullDims(matmulChain, entry) {
					if nh, hd, ok := DraftANEDraftAttnHeadKVForOC(entry, ocAttn); ok {
						nHeadKV = nh
						headDim = hd
					}
				} else if nh := DraftANEDraftAttnQueryHeadCount(entry); nh > 0 {
					nHeadQ = nh
				}
				if nHeadQ > 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_N_HEAD", strconv.Itoa(nHeadQ))
				}
				if nHeadKV > 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_N_HEAD_KV", strconv.Itoa(nHeadKV))
				}
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_N_EMBD_HEAD", strconv.Itoa(headDim))
				if meta.RopeNDims > 0 {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ROPE_N_DIMS", strconv.Itoa(meta.RopeNDims))
				}
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ROPE_FREQ_BASE", strconv.FormatFloat(meta.FreqBase, 'g', -1, 64))
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ROPE_FREQ_SCALE", strconv.FormatFloat(meta.FreqScale, 'g', -1, 64))
				if meta.NeoX {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ROPE_NEOX", "1")
				}
				qNormTensor := ResolveChain13AttnQNormTensor(entry)
				if qp, _, err := MaterializeANEDraftNormGammaFile(entry, qNormTensor, meta.HeadDim); err == nil && qp != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ATTN_Q_NORM_FILE", qp)
				}
				kNormTensor := ResolveChain13AttnKNormTensor(entry)
				if kp, _, err := MaterializeANEDraftNormGammaFile(entry, kNormTensor, meta.HeadDim); err == nil && kp != "" {
					cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_ATTN_K_NORM_FILE", kp)
				}
				cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_HOST_ROPE", "1")
			}
		}
		cmd.Env = append(cmd.Env,
			"ZEROLLAMA_ANE_DRAFT_CHANNELS="+strconv.Itoa(ch),
			"ZEROLLAMA_ANE_DRAFT_SPATIAL="+strconv.Itoa(sp),
		)
		matmulChainForStride := 0
		if kernel == "matmul" {
			matmulChainForStride = matmulChain
		}
		stride := DraftANEHandoffStride(kernel, matmulChainForStride)
		if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE")); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				stride = n
			}
		}
		cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE", strconv.Itoa(stride))
		run.HandoffStride = stride
		if convDepth > 0 && kernel != "matmul" {
			cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_CONV_DEPTH="+strconv.Itoa(convDepth))
		}
		if telemetry {
			cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TELEMETRY=1")
		}
		if driveMode != "" {
			cmd.Env = upsertEnv(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DRIVE", driveMode)
			if needDriveHead {
				if headManifest, _, err := MaterializeANEDraftWeightBundleWithDrive(entry, true); err == nil {
					for k, v := range ExportEnvForManifest(headManifest, "") {
						if strings.HasPrefix(k, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE") && kernel == "matmul" {
							continue
						}
						if k == "ZEROLLAMA_ANE_DRAFT" || k == "ZEROLLAMA_ANE_DRAFT_CHANNELS" || k == "ZEROLLAMA_ANE_DRAFT_SPATIAL" {
							continue
						}
						cmd.Env = upsertEnv(cmd.Env, k, v)
					}
				}
				if driveMode == "shadow" {
					metrics := strings.TrimSpace(strings.ToLower(driveMetrics))
					if metrics == "" {
						metrics = "tokens"
					}
					cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DRIVE_METRICS="+metrics)
				}
			} else if kernel == "matmul" {
				metrics := strings.TrimSpace(strings.ToLower(driveMetrics))
				if metrics == "" {
					metrics = "hidden"
				}
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DRIVE_METRICS="+metrics)
			}
		}
	}
	cmd.Env = prependLlamaServerLibPath(cmd.Env, serverBin)
	cmd.Env = dedupeEnvLastWins(cmd.Env)

	logOut, err := os.OpenFile(logPath, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return run, err
	}
	cmd.Stdout = logOut
	cmd.Stderr = logOut

	if err := cmd.Start(); err != nil {
		logOut.Close()
		return run, err
	}

	stop := func() {
		if cmd.Process != nil {
			_ = cmd.Process.Signal(syscall.SIGTERM)
			done := make(chan struct{})
			go func() {
				_ = cmd.Wait()
				close(done)
			}()
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
			}
			cmd.Process = nil
		}
		if logOut != nil {
			_ = logOut.Sync()
			_ = logOut.Close()
			logOut = nil
		}
	}
	defer stop()

	copyABLegLog := func() {
		dst := aneDraftABKeepLogDest(os.Getenv("ZEROLLAMA_ANE_KEEP_AB_LOG"), aneHook, driveMode)
		if dst == "" {
			return
		}
		logBytes, _ := os.ReadFile(logPath)
		if len(logBytes) == 0 {
			return
		}
		_ = os.WriteFile(dst, logBytes, 0o644)
		snapshotANEDraftEnv(cmd.Env, dst+".env")
	}
	defer copyABLegLog()

	healthURL := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	healthWait := legTimeout / 2
	if healthWait < 3*time.Minute {
		healthWait = 3 * time.Minute
	}
	if healthWait > legTimeout-30*time.Second {
		healthWait = legTimeout - 30*time.Second
	}
	deadline := time.Now().Add(healthWait)
	for time.Now().Before(deadline) {
		req, _ := http.NewRequestWithContext(runCtx, http.MethodGet, healthURL, nil)
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		select {
		case <-runCtx.Done():
			return run, runCtx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}

	body := []byte(fmt.Sprintf(`{"model":"ab","messages":[{"role":"user","content":"Write a short poem about apples."}],"max_tokens":%d,"stream":false}`, maxTokens))
	req, err := http.NewRequestWithContext(runCtx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port), bytes.NewReader(body))
	if err != nil {
		return run, err
	}
	req.Header.Set("Content-Type", "application/json")

	t0 := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		run.Error = err.Error()
		return run, err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		run.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, truncate(string(respBody), 200))
		return run, fmt.Errorf("%s", run.Error)
	}

	var cc chatCompletionResp
	if err := json.Unmarshal(respBody, &cc); err != nil {
		run.Error = "parse completion json: " + err.Error()
		return run, err
	}

	elapsed := time.Since(t0)
	run.OK = true
	fillServerRunFromCompletion(&run, cc, elapsed.Seconds()*1000)

	stop()
	time.Sleep(750 * time.Millisecond)

	logBytes, _ := os.ReadFile(logPath)
	logText := string(logBytes)
	if g, a, gt, at, ok := parseDflashStatistics(logText); ok {
		run.GenDrafts = g
		run.AccDrafts = a
		run.GenTokens = gt
		run.AccTokens = at
		if gt > 0 {
			run.DraftAcceptance = draftAcceptance(gt, at)
		}
	}
	if ar := draftAcceptRateRE.FindAllStringSubmatch(logText, -1); len(ar) > 0 {
		last := ar[len(ar)-1]
		if len(last) == 2 {
			if v, err := strconv.ParseFloat(last[1], 64); err == nil {
				if run.DraftAcceptance == 0 {
					run.DraftAcceptance = v
				}
			}
		}
	}
	if aneHook {
		run.HandoffSteps = countANEHandoffsFromLog(logText)
		run.Conv2Chained = aneConv2ChainedRE.MatchString(logText)
		if m := aneConvDepthRE.FindStringSubmatch(logText); len(m) == 3 {
			if cap, err := strconv.Atoi(m[1]); err == nil {
				run.ConvDepthCap = cap
			}
			if active, err := strconv.Atoi(m[2]); err == nil {
				run.ActiveConvDepth = active
			}
		} else if convDepth > 0 {
			run.ConvDepthCap = convDepth
			active := inferActiveConvDepthFromLog(logText)
			if active == 0 || active > convDepth {
				active = convDepth
			}
			run.ActiveConvDepth = active
		}
		if telemetry {
			run.GoldenCosine, run.GoldenSteps = parseGoldenCosineFromLog(logText)
		}
		if kernel == "matmul" {
			run.MatmulChain = parseMatmulChainFromLog(logText)
		}
		if driveMode == "shadow" {
			var cosSum float64
			var cosN int
			run.DriveShadowSteps, run.DriveShadowMatches, cosSum, cosN = parseB7ShadowFromLog(logText)
			if cosN > 0 {
				run.DriveShadowHiddenCos = cosSum / float64(cosN)
				run.DriveShadowHiddenSteps = cosN
			}
		}
	}

	return run, nil
}

func aneDraftABKeepLogDest(keep string, aneHook bool, driveMode string) string {
	if keep == "" {
		return ""
	}
	leg := "metal"
	if aneHook {
		if strings.TrimSpace(driveMode) != "" {
			leg = "ane-drive"
		} else {
			leg = "ane-hook"
		}
	}
	if keep == "1" {
		return fmt.Sprintf("/tmp/zerollama-ane-ab-%s.log", leg)
	}
	if strings.HasSuffix(keep, ".log") {
		return strings.TrimSuffix(keep, ".log") + "-" + leg + ".log"
	}
	return keep + "-" + leg
}

func snapshotANEDraftEnv(env []string, dst string) {
	if dst == "" {
		return
	}
	var b strings.Builder
	for _, e := range env {
		k, _, _ := strings.Cut(e, "=")
		if strings.HasPrefix(k, "ZEROLLAMA_ANE_DRAFT") {
			b.WriteString(e)
			b.WriteByte('\n')
		}
	}
	_ = os.WriteFile(dst, []byte(b.String()), 0o644)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// RunANEDraftABJSON writes B4 A/B JSON to w.
func RunANEDraftABJSON(ctx context.Context, w io.Writer, preferred string, steps int, quick, e2e, e2eTelemetry bool, driveMode string, convDepth int) error {
	res, err := ProbeANEDraftAB(ctx, preferred, steps, quick, e2e, e2eTelemetry, driveMode, "", convDepth, "conv", false)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}

// CountANEHandoffsFromLog exports handoff line counting for tests.
func CountANEHandoffsFromLog(logText string) int {
	return countANEHandoffsFromLog(logText)
}

// ParseDflashStatisticsFromLog exports log parsing for tests.
func ParseDflashStatisticsFromLog(logText string) (genDrafts, accDrafts, genTokens, accTokens uint64, ok bool) {
	return parseDflashStatistics(logText)
}

// ReadServerLogTail reads last lines from a path (test helper).
func ReadServerLogTail(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sc := bufio.NewScanner(bytes.NewReader(b))
	var lines []string
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if len(lines) > 40 {
		lines = lines[len(lines)-40:]
	}
	return strings.Join(lines, "\n"), nil
}
