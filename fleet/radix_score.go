package fleet

import (
	"os"
	"strings"
)

// fleetRadixScoreEnabled gates L3-R8 radix residency soft-scoring (default on).
func fleetRadixScoreEnabled() bool {
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_RADIX_SCORE"))
	if s == "" {
		return true
	}
	return s == "1" || strings.EqualFold(s, "on") || strings.EqualFold(s, "true")
}

// fleetRadixHashScoreEnabled gates L3-R9 / LA13 content-hash longest-prefix scoring (default on).
func fleetRadixHashScoreEnabled() bool {
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_RADIX_HASH_SCORE"))
	if s == "" {
		return true
	}
	return s == "1" || strings.EqualFold(s, "on") || strings.EqualFold(s, "true")
}

// longestPrefixHashMatch counts how many leading client block hashes are present
// on the peer. Hash chains are rooted at block 0, so a break stops the match.
func longestPrefixHashMatch(want, have []string) int {
	if len(want) == 0 || len(have) == 0 {
		return 0
	}
	set := make(map[string]struct{}, len(have))
	for _, h := range have {
		h = strings.TrimSpace(h)
		if h == "" {
			continue
		}
		set[h] = struct{}{}
	}
	n := 0
	for _, h := range want {
		h = strings.TrimSpace(h)
		if h == "" {
			break
		}
		if _, ok := set[h]; !ok {
			break
		}
		n++
	}
	return n
}
