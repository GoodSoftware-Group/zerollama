package agentstats

import (
	"strconv"
	"strings"
	"time"
)

// MaybeRecordRunnerLine parses mlxrunner subprocess slog text lines and records
// cache/prefill events to the agent stats file.
//
// fast_path = live-session LCP rewind; same_branch = trie restore without page-in/out.
// Both differ from a plain cache hit where utilization_pct is high but TTFT is still slow.
func MaybeRecordRunnerLine(line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	msg := slogField(line, "msg")
	switch msg {
	case "cache hit", "cache miss":
		Record("mlx_cache", map[string]any{
			"event":            "mlx_cache",
			"msg":              msg,
			"total":            intField(line, "total"),
			"matched":          intField(line, "matched"),
			"cached":           intField(line, "cached"),
			"left":             intField(line, "left"),
			"utilization_pct":  floatField(line, "utilization_pct"),
			"fast_path":        boolField(line, "fast_path"),
			"same_branch":      boolField(line, "same_branch"),
			"rewound_to":       intField(line, "rewound_to"),
		})
	case "prefill complete":
		Record("mlx_prefill", map[string]any{
			"event":          "mlx_prefill",
			"prompt_tokens":  intField(line, "prompt_tokens"),
			"cached_tokens":  intField(line, "cached_tokens"),
			"prefill_tokens": intField(line, "prefill_tokens"),
			"elapsed_ms":     durationMs(line, "elapsed"),
			"chunk_size":     intField(line, "chunk_size"),
			"tok_per_sec":    floatField(line, "tok_per_sec"),
		})
	default:
		if strings.Contains(msg, "KV restore missed") || strings.Contains(msg, "prefix cache miss despite prompt_cache_key") {
			Record("mlx_cache_warn", map[string]any{
				"event": "mlx_cache_warn",
				"msg":   msg,
				"line":  line,
			})
		}
	}
}

func slogField(line, key string) string {
	needle := key + "="
	i := strings.Index(line, needle)
	if i < 0 {
		return ""
	}
	rest := line[i+len(needle):]
	if len(rest) == 0 {
		return ""
	}
	if rest[0] == '"' {
		rest = rest[1:]
		if j := strings.Index(rest, `"`); j >= 0 {
			return rest[:j]
		}
		return rest
	}
	if j := strings.IndexAny(rest, " \t"); j >= 0 {
		return rest[:j]
	}
	return rest
}

func intField(line, key string) int {
	v, _ := strconv.Atoi(slogField(line, key))
	return v
}

func floatField(line, key string) float64 {
	v, _ := strconv.ParseFloat(slogField(line, key), 64)
	return v
}

func boolField(line, key string) bool {
	return slogField(line, key) == "true"
}

func durationMs(line, key string) int {
	raw := slogField(line, key)
	if raw == "" {
		return 0
	}
	if d, err := time.ParseDuration(raw); err == nil {
		return int(d.Milliseconds())
	}
	return 0
}
