package llm

import (
	"bytes"
	"log/slog"
	"os/exec"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

const specDraftBackendSamplingFlag = "--spec-draft-backend-sampling"
const noSpecDraftBackendSamplingFlag = "--no-spec-draft-backend-sampling"

var llamaServerHelpCache sync.Map // serverBin -> help text

// llamaServerHelp returns cached “llama-server --help“ output for flag probing.
func llamaServerHelp(serverBin string) string {
	if serverBin == "" {
		return ""
	}
	if cached, ok := llamaServerHelpCache.Load(serverBin); ok {
		return cached.(string)
	}
	cmd := exec.Command(serverBin, "--help")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	// Match subprocess library paths so Metal builds can print --help
	// (otherwise dyld fails and we falsely fall back to draft-dflash).
	SetupLlamaServerCommandEnv(cmd, serverBin, nil, nil)
	err := cmd.Run()
	text := out.String()
	// Some Metal builds abort after printing --help (SIGABRT); still use the text.
	if text == "" || (!strings.Contains(text, "--spec-type") && strings.Contains(text, "Library not loaded")) {
		if err != nil {
			slog.Debug("llama-server --help failed; assuming flags unsupported", "binary", serverBin, "error", err)
		}
		llamaServerHelpCache.Store(serverBin, "")
		return ""
	}
	if err != nil {
		slog.Debug("llama-server --help exited non-zero; using captured help text", "binary", serverBin, "error", err)
	}
	llamaServerHelpCache.Store(serverBin, text)
	return text
}

func llamaServerSupportsFlag(serverBin, flag string) bool {
	if flag == "" {
		return false
	}
	// WHY fail closed: without a binary path we cannot probe; upstream always
	// passes a resolved exe — guessing "supported" breaks eliza fork / older builds.
	if serverBin == "" {
		return false
	}
	return strings.Contains(llamaServerHelp(serverBin), flag)
}

func appendSpecDraftBackendSamplingArg(params []string, serverBin string) []string {
	// WHY probe: eliza fork / older llama-server builds lack this flag; passing it
	// aborts startup before model load (see qwen3.6-mtp on eliza-llama.cpp).
	if !llamaServerSupportsFlag(serverBin, specDraftBackendSamplingFlag) {
		slog.Debug(
			"llama-server lacks draft backend sampling flag; skipping",
			"binary", serverBin,
			"flag", specDraftBackendSamplingFlag,
		)
		return params
	}
	return append(params, specDraftBackendSamplingFlag)
}

// appendNoSpecDraftBackendSamplingArg forces draft sampling back onto the CPU path.
//
// WHY: the ggml-org 5f55650a pin (b10199+1) refactored the backend (GPU-side) draft
// sampler kernels (llama_sampler_*_backend_apply in src/llama-sampler.cpp) to flatten
// logits via ggml_reshape_1d before argmax/top-k/top-p/min-p. That refactor is broken
// for draft-mtp specifically: with --spec-draft-backend-sampling (the default, and what
// appendSpecDraftBackendSamplingArg used to force on), draft acceptance collapses to
// ~0% from the very first turn of a fresh conversation and generation degenerates into
// multilingual token salad (qwen3.6 MTP, 2026-07-30 incident). Verified fix (isolated,
// single standalone llama-server, fresh short conversation): launching the identical
// binary/model with --no-spec-draft-backend-sampling (CPU-side draft sampling) restores
// 42-64% draft acceptance and coherent output, matching pre-rebase (f95de977 / b10159)
// behavior. CPU-side sampling for the tiny 1-layer MTP draft head is cheap, so this has
// no meaningful perf cost. Root cause is upstream, not ours to fix here; revisit once
// ggml-org's backend sampler refactor is corrected upstream and drop this override.
//
// CAUTION - this does NOT fully clear draft-mtp for qwen3.6 in production: a live
// re-enable + revert on 2026-07-30 showed a SEPARATE failure mode survives this fix.
// On a real long multi-turn conversation, "forcing full prompt re-processing due to
// lack of cache data (likely due to SWA or hybrid/recurrent memory ...)" fires (qwen3.6
// is a hybrid Gated-Delta-Net/SWA arch) and desyncs the MTP draft context's state for
// the rest of that runner's slot lifetime — subsequent unrelated requests on the same
// slot then also produced multilingual token salad, even with backend sampling off.
// This matches ggml-org/llama.cpp#23322 (low/zero MTP acceptance under SWA/hybrid-memory
// cache invalidation) and is likely the original incident's root cause on top of the
// backend-sampling regression. Needs its own fix/investigation (checkpoint restore
// should probably also resync/clear ctx_dft, not just ctx_tgt) before draft-mtp is safe
// to re-enable for qwen3.6-64k/:35b/:27b in prod. Until then those tags stay pinned to
// spec_type=none/draft_num_predict=0.
func appendNoSpecDraftBackendSamplingArg(params []string, serverBin string) []string {
	if !llamaServerSupportsFlag(serverBin, noSpecDraftBackendSamplingFlag) {
		slog.Debug(
			"llama-server lacks the no-backend-sampling override; leaving backend sampling at its default",
			"binary", serverBin,
			"flag", noSpecDraftBackendSamplingFlag,
		)
		return params
	}
	return append(params, noSpecDraftBackendSamplingFlag)
}

const specDmAdaptiveFlag = "--spec-dm-adaptive"

// appendSpecDmAdaptiveArg passes Bee B1 profit controller when ZEROLLAMA_SPEC_DM_ADAPTIVE
// is set and the binary advertises --spec-dm-adaptive (patch 0102). Default: omit (off).
func appendSpecDmAdaptiveArg(params []string, serverBin string) []string {
	mode := envconfig.SpecDmAdaptive()
	if mode == "" {
		return params
	}
	if !llamaServerSupportsFlag(serverBin, specDmAdaptiveFlag) {
		slog.Debug(
			"llama-server lacks adaptive draft-max flag; skipping",
			"binary", serverBin,
			"flag", specDmAdaptiveFlag,
		)
		return params
	}
	return append(params, specDmAdaptiveFlag, mode)
}

// resolveDFlashSpecType picks the CLI --spec-type token supported by serverBin.
func resolveDFlashSpecType(serverBin string) string {
	help := llamaServerHelp(serverBin)
	if help == "" {
		// Prefer the newer ggml-org name when we cannot probe.
		return "draft-dflash"
	}
	// Match the enumerated token list carefully — "draft-dflash" contains "dflash".
	if strings.Contains(help, "draft-dflash") {
		return "draft-dflash"
	}
	if strings.Contains(help, ",dflash") || strings.Contains(help, " dflash") || strings.HasSuffix(strings.TrimSpace(help), "dflash") {
		return "dflash"
	}
	// Help line is often: --spec-type none,...,dflash
	if strings.Contains(help, "spec-type") && strings.Contains(help, "dflash") {
		return "dflash"
	}
	return "draft-dflash"
}

// resetLlamaServerHelpCache clears cached help text (tests only).
func resetLlamaServerHelpCache() {
	llamaServerHelpCache = sync.Map{}
}

func launchFlashAttentionMode(launch llamaServerLaunchConfig) ml.FlashAttentionType {
	enabled := envconfig.FlashAttention(false)
	userSet := enabled == envconfig.FlashAttention(true)
	if userSet {
		if enabled {
			return ml.FlashAttentionEnabled
		}
		return ml.FlashAttentionDisabled
	}
	// L1 profile flash_attn=true → force on (matches Python -fa on).
	if launch.forceFlashAttn {
		return ml.FlashAttentionEnabled
	}
	return LlamaServerFlashAttention(launch.gpus)
}

func appendFlashAttentionArgsForLaunch(params []string, launch llamaServerLaunchConfig) []string {
	switch launchFlashAttentionMode(launch) {
	case ml.FlashAttentionEnabled:
		return append(params, "--flash-attn", "on")
	case ml.FlashAttentionDisabled:
		return append(params, "--flash-attn", "off")
	default:
		return append(params, "--flash-attn", "auto")
	}
}
