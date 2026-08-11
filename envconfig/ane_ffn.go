package envconfig

import (
	"strconv"
	"strings"
)

// ANEFFNEnabled reports ZEROLLAMA_ANE_FFN (lab FFN/MUL_MAT intercept).
// Default off — never enable on production :11434 / :8081.
func ANEFFNEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_ANE_FFN")))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// ANEFFNMode returns shadow|force|off. When master switch is off, always "off".
func ANEFFNMode() string {
	if !ANEFFNEnabled() {
		return "off"
	}
	v := strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_ANE_FFN_MODE")))
	switch v {
	case "force":
		return "force"
	case "off":
		return "off"
	default:
		return "shadow"
	}
}

// ANEFFNLabPort returns ZEROLLAMA_ANE_FFN_LAB_PORT (or _PORT). 0 = unset.
func ANEFFNLabPort() int {
	for _, k := range []string{"ZEROLLAMA_ANE_FFN_LAB_PORT", "ZEROLLAMA_ANE_FFN_PORT"} {
		v := strings.TrimSpace(Var(k))
		if v == "" {
			continue
		}
		n, err := strconv.Atoi(v)
		if err == nil {
			return n
		}
	}
	return 0
}

// ANEFFNIsProductionPort is true for reserved production listeners.
func ANEFFNIsProductionPort(port int) bool {
	return port == 11434 || port == 8081
}

// ANEFFNName returns ZEROLLAMA_ANE_FFN_NAME (shexp|ffn|dense|any|custom).
// Empty means no name filter (match all weight names).
func ANEFFNName() string {
	return strings.TrimSpace(Var("ZEROLLAMA_ANE_FFN_NAME"))
}

// ANEFFNSwiglu reports ZEROLLAMA_ANE_FFN_SWIGLU (fuse up+gate+GLU+down).
func ANEFFNSwiglu() bool {
	v := strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_ANE_FFN_SWIGLU")))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func aneFFNFlag(name string) bool {
	v := strings.ToLower(strings.TrimSpace(Var(name)))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// ANEFFNInt8 reports ZEROLLAMA_ANE_FFN_INT8 (int8 weight blobs in force replace).
func ANEFFNInt8() bool { return aneFFNFlag("ZEROLLAMA_ANE_FFN_INT8") }

// ANEFFNW8A8 reports ZEROLLAMA_ANE_FFN_W8A8 (hid W8A8; implies int8 weights).
func ANEFFNW8A8() bool { return aneFFNFlag("ZEROLLAMA_ANE_FFN_W8A8") }

// ANEFFNW8A8X reports ZEROLLAMA_ANE_FFN_W8A8_X (dual W8A8 x+hid).
func ANEFFNW8A8X() bool { return aneFFNFlag("ZEROLLAMA_ANE_FFN_W8A8_X") }

// ANEFFNInt8In reports ZEROLLAMA_ANE_FFN_INT8_IN (host int8 acts; implies W8A8_X).
func ANEFFNInt8In() bool { return aneFFNFlag("ZEROLLAMA_ANE_FFN_INT8_IN") }

// ANEFFNAllowServe reports whether FFN intercept may run for this serve port.
// Force and shadow both refuse production ports.
func ANEFFNAllowServe(servePort int) bool {
	if !ANEFFNEnabled() || ANEFFNMode() == "off" {
		return false
	}
	if ANEFFNIsProductionPort(servePort) {
		return false
	}
	lab := ANEFFNLabPort()
	if lab > 0 && servePort > 0 && servePort != lab {
		return false
	}
	return true
}
