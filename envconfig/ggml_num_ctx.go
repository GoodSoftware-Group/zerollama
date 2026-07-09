package envconfig

import (
	"log/slog"
	"strconv"
	"strings"
)

// GgmlClampNumCtxEnabled reports ZEROLLAMA_GGML_CLAMP_NUM_CTX (legacy server-wide default).
// Prefer per-request options.ggml_clamp_num_ctx so each client controls its own load policy.
func GgmlClampNumCtxEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_GGML_CLAMP_NUM_CTX")))
	if v == "" {
		return false
	}
	switch v {
	case "0", "false", "no", "off":
		return false
	case "1", "true", "yes", "on", "auto":
		return true
	default:
		slog.Warn("invalid ZEROLLAMA_GGML_CLAMP_NUM_CTX; treating as off", "value", v)
		return false
	}
}

// GgmlSuggestCtxMaxCap is the upper bound for ggml suggest binary search (default 131072).
// Why cap: matches runtime VRAM_SUGGEST_CTX_MAX; avoids searching to 262K when train
// metadata allows it but VRAM does not.
func GgmlSuggestCtxMaxCap() int {
	raw := strings.TrimSpace(Var("ZEROLLAMA_GGML_SUGGEST_CTX_MAX"))
	if raw == "" {
		return 131072
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 512 {
		return 131072
	}
	return n
}

// GgmlVRAMMargin multiplies load estimates before comparing to free VRAM (default 1.05).
// Why >1.0: GraphSize + file-size proxy underestimates projector/graph spikes; margin
// keeps suggest conservative without requiring exact tensor offload math in suggest path.
func GgmlVRAMMargin() float64 {
	raw := strings.TrimSpace(Var("ZEROLLAMA_GGML_VRAM_MARGIN"))
	if raw == "" {
		return 1.05
	}
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil || f < 1.0 {
		return 1.05
	}
	return f
}

func ggmlClampNumCtxDisplay() string {
	v := strings.TrimSpace(Var("ZEROLLAMA_GGML_CLAMP_NUM_CTX"))
	if v == "" {
		return "0"
	}
	return v
}
