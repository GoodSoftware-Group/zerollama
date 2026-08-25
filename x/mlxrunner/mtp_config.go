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
