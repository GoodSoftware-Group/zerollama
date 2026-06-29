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
	aneHandoffStepRE    = regexp.MustCompile(`common_ane_draft_handoff_after_decode: step=\d+ ggml iosurface handoff ok`)
	aneConv2ChainedRE   = regexp.MustCompile(`B6 dual conv1 chain active|B8 triple conv1 chain active|B9 quad conv1 chain active|B10 pent conv1 chain active|B11 hex conv1 chain active|B12 hept conv1 chain active|B13 oct conv1 chain active`)
	aneConvDepthRE      = regexp.MustCompile(`conv depth cap=(\d+) active_convs=(\d+)`)
	aneGoldenCosineRE   = regexp.MustCompile(`B6 golden step=\d+ mode=\w+ mse_ref_vs_ane=[0-9.eE+-]+ cosine=([0-9.-]+)`)
	aneB7ShadowRE       = regexp.MustCompile(`B7 shadow step=\d+ seq=\d+ ane_tok=\d+ metal_tok=\d+ match=(\d+)(?: hidden_cos=([0-9.eE+-]+))?`)
	aneMatmulChain4RE   = regexp.MustCompile(`chain4=swiglu\+down\+attn_gate|mode=matmul_chain4`)
	aneMatmulChain5RE   = regexp.MustCompile(`chain5=swiglu\+down\+attn_gate\+ssm_out|mode=matmul_chain5`)
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
func ProbeANEDraftAB(ctx context.Context, preferred string, steps int, quick, runE2E, e2eTelemetry bool, driveMode, driveMetrics string, convDepth int, kernel string) (ANEDraftABResult, error) {
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

	inproc, err := ProbeANEInprocessSmoke(ctx, preferred, steps, quick)
	if err != nil {
		out.Error = err.Error()
		return out, err
	}
	out.Micro.InprocessAvgEvalMS = inproc.AvgEvalMS
	out.Micro.InprocessAvgMapFillMS = inproc.AvgMapFillMS
	out.Micro.InprocessAvgStepMS = inproc.AvgEvalMS + inproc.AvgMapFillMS
	out.Micro.InprocessSteps = len(inproc.Steps)
	out.Micro.KernelReused = inproc.KernelReused
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

	metalRun, err := runDflashServerLeg(ctx, serverBin, entry, port, maxTokens, false, false, "", "", 0, "")
	if err != nil {
		return ANEDraftABE2E{}, err
	}
	aneRun, err := runDflashServerLeg(ctx, serverBin, entry, port, maxTokens, true, telemetry, driveMode, driveMetrics, convDepth, kernel)
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
	aneRun, err := runDflashServerLeg(ctx, serverBin, entry, port, maxTokens, true, false, "", "", convDepth, kernel)
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

func runDflashServerLeg(ctx context.Context, serverBin string, entry ANEDraftEntry, port, maxTokens int, aneHook, telemetry bool, driveMode, driveMetrics string, convDepth int, kernel string) (ANEDraftServerRun, error) {
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

	runCtx, cancel := context.WithTimeout(ctx, 3*time.Minute)
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

	cmd := exec.CommandContext(runCtx, serverBin, args...)
	cmd.Env = os.Environ()
	if aneHook {
		kernel = strings.ToLower(strings.TrimSpace(kernel))
		if kernel == "" {
			kernel = "conv"
		}
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
			cmd.Env = append(cmd.Env,
				"ZEROLLAMA_ANE_DRAFT_KERNEL=matmul",
				"ZEROLLAMA_ANE_DRAFT_MATMUL_SEQ="+strconv.Itoa(matmulSeq),
				"ZEROLLAMA_ANE_DRAFT_MATMUL_OC="+strconv.Itoa(matmulOC),
			)
			sp = matmulSeq
			if wpath, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.0.ffn_gate.weight", matmulIC, matmulOC); err == nil && wpath != "" {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE="+wpath)
			}
			icUp, ocUp := DraftANEMatmulChain3UpDims(matmulIC, matmulOC)
			icDown, ocDown := DraftANEMatmulChain3DownDims(matmulOC, matmulIC)
			forceChain := 0
			if v, ok := aneDraftEnvValue(os.Environ(), "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN"); ok {
				if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
					forceChain = n
				}
			}
			matmulChain = 1
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
								}
							}
						}
						switch matmulChain {
						case 5:
							cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=5")
						case 4:
							cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=4")
						default:
							cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=3")
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
			if forceChain == 1 {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=1")
				matmulChain = 1
			} else if forceChain == 2 {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=2")
				matmulChain = 2
			} else if forceChain == 3 {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=3")
				matmulChain = 3
			} else if forceChain == 4 {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=4")
				matmulChain = 4
			} else if forceChain == 5 {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=5")
				matmulChain = 5
			}
			ch = matmulIC
		} else {
			applyConvManifestEnv(&cmd.Env, manifest, convDepth)
		}
		if gamma := manifest.GammaWeightPath(); gamma != "" {
			cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_GAMMA_FILE="+gamma)
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
		if !aneDraftEnvHasKey(cmd.Env, "ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE") {
			cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE="+strconv.Itoa(stride))
		}
		run.HandoffStride = stride
		if v, ok := aneDraftEnvValue(cmd.Env, "ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE"); ok {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				run.HandoffStride = n
			}
		}
		if convDepth > 0 && kernel != "matmul" {
			cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_CONV_DEPTH="+strconv.Itoa(convDepth))
		}
		if telemetry {
			cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_TELEMETRY=1")
		}
		if driveMode != "" {
			cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DRIVE="+driveMode)
			if needDriveHead {
				cmd.Env = append(cmd.Env, "ZEROLLAMA_ANE_DRAFT_DRIVE_VOCAB_CAP=8192")
				if headManifest, _, err := MaterializeANEDraftWeightBundleWithDrive(entry, true); err == nil {
					for k, v := range ExportEnvForManifest(headManifest, "") {
						if strings.HasPrefix(k, "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE") && kernel == "matmul" {
							continue
						}
						if k == "ZEROLLAMA_ANE_DRAFT" || k == "ZEROLLAMA_ANE_DRAFT_CHANNELS" || k == "ZEROLLAMA_ANE_DRAFT_SPATIAL" {
							continue
						}
						cmd.Env = append(cmd.Env, k+"="+v)
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

	healthURL := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	deadline := time.Now().Add(90 * time.Second)
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
	time.Sleep(150 * time.Millisecond)

	logBytes, _ := os.ReadFile(logPath)
	logText := string(logBytes)
	if keep := os.Getenv("ZEROLLAMA_ANE_KEEP_AB_LOG"); keep != "" {
		dst := keep
		if keep == "1" {
			dst = "/tmp/zerollama-ane-ab-last.log"
		}
		_ = os.WriteFile(dst, logBytes, 0o644)
	}
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

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// RunANEDraftABJSON writes B4 A/B JSON to w.
func RunANEDraftABJSON(ctx context.Context, w io.Writer, preferred string, steps int, quick, e2e, e2eTelemetry bool, driveMode string, convDepth int) error {
	res, err := ProbeANEDraftAB(ctx, preferred, steps, quick, e2e, e2eTelemetry, driveMode, "", convDepth, "conv")
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
