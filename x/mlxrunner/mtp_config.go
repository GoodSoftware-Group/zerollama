package mlxrunner

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// MTP load policy via ZEROLLAMA_MLX_MTP (mtplx-shaped fail-closed expect; F0750):
//
//	"" / off / auto — soft: missing draft heads → plain AR (historical default)
//	require         — fail Load if the checkpoint ships no draft / SelfDraft head
//
// Matches mtplx `--generation-mode mtp` requiring `--load-mtp`: do not silently
// advertise MTP then decode AR when heads are absent.
func mtpLoadMode() string {
	return strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_MTP")))
}

// MTPRequire reports fail-closed MTP expect at load.
func MTPRequire() bool {
	switch mtpLoadMode() {
	case "require", "1", "on", "true", "yes", "mtp":
		return true
	default:
		return false
	}
}

// MTP history policy (mtplx-shaped; F0751) via ZEROLLAMA_MLX_MTP_HISTORY:
//
//	committed   — feed full prompt through draft KV / frontier (product default)
//	last_window — only the trailing window enters draft KV (see LAST_WINDOW)
//	auto        — committed if prompt < threshold, else last_window
//	cycle       — diagnostic alias: treated as committed (no separate cycle port yet)
//
// Window / threshold (mtplx defaults):
//
//	ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW            default 8192
//	ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD   default 16384
const (
	mtpHistoryCommitted  = "committed"
	mtpHistoryLastWindow = "last_window"
	mtpHistoryAuto       = "auto"
	mtpHistoryCycle      = "cycle"

	defaultMTPHistoryLastWindow  = 8192
	defaultMTPHistoryLWThreshold = 16384
)

// mtpHistoryPlan is the resolved per-request history policy.
type mtpHistoryPlan struct {
	Policy       string // committed | last_window (never auto after resolve)
	WindowTokens int    // >0 only for last_window
	// HistoryStart is the absolute look-ahead token index at which draft-KV
	// pair writes may begin (1-based prompt index of first kept look-ahead).
	// Pairs for look-ahead absolute positions < HistoryStart are skipped.
	HistoryStart int
}

func normalizeMTPHistoryPolicy(policy string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(policy))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	switch normalized {
	case "", "full":
		return mtpHistoryCommitted, nil
	case "lastwindow", "window":
		return mtpHistoryLastWindow, nil
	case "none", "off":
		return mtpHistoryCycle, nil
	case mtpHistoryAuto, mtpHistoryCycle, mtpHistoryCommitted, mtpHistoryLastWindow:
		return normalized, nil
	default:
		return "", fmt.Errorf("ZEROLLAMA_MLX_MTP_HISTORY must be auto|committed|last_window|cycle (got %q)", policy)
	}
}

func mtpHistoryEnvPolicy() string {
	return strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_MTP_HISTORY"))
}

func mtpHistoryLastWindowTokens() int {
	return envIntDefault("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW", defaultMTPHistoryLastWindow)
}

func mtpHistoryLastWindowThreshold() int {
	return envIntDefault("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD", defaultMTPHistoryLWThreshold)
}

func envIntDefault(key string, def int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return def
	}
	return n
}

// Greedy trio (mtplx 2.9.2 #313/#315/#318): at temperature 0, draft by
// on-device argmax (no Categorical on a one-hot) and accept by token
// equality (no Bernoulli p/q). Measured +2.5–9.8% below 12k tokens and a
// small loss at 16k/32k, so a context fence keeps it off there.
//
//	ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN       default on
//	ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT    default on
//	ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT  default 12288 (0 = no fence)
const defaultGreedyTrioMaxContext = 12288

func envEnabledDefaultOn(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

func greedyTrioMaxContext() int {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT")))
	if raw == "" {
		return defaultGreedyTrioMaxContext
	}
	if raw == "0" || raw == "off" || raw == "none" || raw == "unlimited" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return defaultGreedyTrioMaxContext
	}
	return n
}

func greedyTrioContextOK(promptTokens int) bool {
	fence := greedyTrioMaxContext()
	return fence <= 0 || promptTokens < fence
}

func greedyDraftChainEnabled(promptTokens int) bool {
	return envEnabledDefaultOn("ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN") && greedyTrioContextOK(promptTokens)
}

func batchedGreedyAcceptEnabled(promptTokens int) bool {
	return envEnabledDefaultOn("ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT") && greedyTrioContextOK(promptTokens)
}

// resolveMTPHistoryPolicy mirrors mtplx _resolve_mtp_history_policy.
// requested is usually "auto" or "committed"; empty uses env or committed.
func resolveMTPHistoryPolicy(requested string, promptTokens int) (mtpHistoryPlan, error) {
	req := strings.TrimSpace(requested)
	if req == "" {
		req = mtpHistoryEnvPolicy()
	}
	if req == "" {
		req = mtpHistoryCommitted
	}
	normalized, err := normalizeMTPHistoryPolicy(req)
	if err != nil {
		return mtpHistoryPlan{}, err
	}
	// Env overrides when caller asked for committed or auto (mtplx hot-path rule).
	if env := mtpHistoryEnvPolicy(); env != "" && (normalized == mtpHistoryCommitted || normalized == mtpHistoryAuto) {
		if n2, err := normalizeMTPHistoryPolicy(env); err == nil {
			normalized = n2
		}
	}
	if normalized == mtpHistoryAuto {
		if promptTokens >= mtpHistoryLastWindowThreshold() {
			normalized = mtpHistoryLastWindow
		} else {
			normalized = mtpHistoryCommitted
		}
	}
	if normalized == mtpHistoryCycle {
		// Diagnostic cycle path not ported; keep committed-stream semantics.
		normalized = mtpHistoryCommitted
	}

	plan := mtpHistoryPlan{Policy: normalized}
	if normalized == mtpHistoryLastWindow {
		plan.WindowTokens = mtpHistoryLastWindowTokens()
		// prompt_ids[1:] history; keep last WindowTokens look-ahead tokens.
		historyLen := promptTokens - 1
		if historyLen < 0 {
			historyLen = 0
		}
		keep := plan.WindowTokens
		if keep > historyLen {
			keep = historyLen
		}
		dropped := historyLen - keep
		// First look-ahead absolute index kept (token at prompt index dropped+1).
		plan.HistoryStart = dropped + 1
	}
	return plan, nil
}
