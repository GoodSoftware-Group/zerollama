package envconfig

import "strings"

// ThinkingGate returns ZEROLLAMA_THINKING_GATE for minefield trap 29.
//
// Values:
//   - "" / "default" / "off" — client may enable thinking per request (default)
//   - "deny" — HTTP 400 when a request tries to turn thinking on
//   - "strip" — force think off; ignore client enable kwargs
//
// Use deny or strip on lanes sized for non-thinking max_tokens so a client
// cannot re-enable thinking and blow the ceiling (trap 12 signature).
func ThinkingGate() string {
	v := strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_THINKING_GATE")))
	switch v {
	case "deny", "strip":
		return v
	case "", "default", "off", "0", "false":
		return ""
	default:
		return ""
	}
}
