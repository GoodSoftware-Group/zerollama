package server

import (
	"os"
	"strings"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
)

const (
	loopTrimMinPeriod = 8
	loopTrimMaxPeriod = 48
	loopTrimRepeats   = 3
)

func loopTrimEnabled() bool {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_LOOP_TRIM")))
	return raw != "0" && raw != "off" && raw != "false"
}

// applyLoopTrim drops a triple-repeated 8–48-rune cycle from non-stream
// output (mlx-serve loopTrimmedIds). Streaming already flushed those tokens.
func applyLoopTrim(content string, details *api.FinishDetails) string {
	if content == "" || details == nil || details.Type != "repetition_loop" || !loopTrimEnabled() {
		return content
	}
	if trimmed, ok := trimTripleRepeatRunes(content); ok {
		return trimmed
	}
	return content
}

func trimTripleRepeatRunes(s string) (string, bool) {
	runes := []rune(s)
	n := len(runes)
	maxP := loopTrimMaxPeriod
	if maxP > n/loopTrimRepeats {
		maxP = n / loopTrimRepeats
	}
	for p := loopTrimMinPeriod; p <= maxP; p++ {
		base := n - loopTrimRepeats*p
		first := runes[base : base+p]
		ok := true
		for r := 1; r < loopTrimRepeats; r++ {
			if !equalRunes(first, runes[base+r*p:base+(r+1)*p]) {
				ok = false
				break
			}
		}
		if ok {
			return string(runes[:base]), true
		}
	}
	return s, false
}

func equalRunes(a, b []rune) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func utf8LogprobToken(s string) string {
	if s == "" || utf8.ValidString(s) {
		return s
	}
	return strings.ToValidUTF8(s, "\uFFFD")
}
