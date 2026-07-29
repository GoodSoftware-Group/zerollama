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

var llamaServerHelpCache sync.Map // serverBin -> help text

// llamaServerHelp returns cached ``llama-server --help`` output for flag probing.
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
