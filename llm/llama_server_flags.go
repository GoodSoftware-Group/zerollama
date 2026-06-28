package llm

import (
	"bytes"
	"log/slog"
	"os/exec"
	"strings"
	"sync"
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
	if err := cmd.Run(); err != nil {
		slog.Debug("llama-server --help failed; assuming flags unsupported", "binary", serverBin, "error", err)
		llamaServerHelpCache.Store(serverBin, "")
		return ""
	}
	text := out.String()
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

// resetLlamaServerHelpCache clears cached help text (tests only).
func resetLlamaServerHelpCache() {
	llamaServerHelpCache = sync.Map{}
}
