package envconfig

import "strings"

// ANEDraftEnabled reports ZEROLLAMA_ANE_DRAFT (future hybrid draft-on-ANE path).
// Default off — lab subprocesses only until scheduler wiring lands.
func ANEDraftEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_ANE_DRAFT")))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
