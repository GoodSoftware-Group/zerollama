package version

import (
	"fmt"
	"runtime"
	"strings"
	"unicode"
)

// Version is the registry / User-Agent semver. registry.ollama.ai returns 412
// for ollama/0.0.x and for non-semver strings (git hashes). Gallery tags such
// as embeddinggemma and Qwen3-Embedding 4B/8B need a current Ollama UA
// (≥0.30.x; 0.40.x matches current upstream clients).
//
// Do not ldflag this to `git describe`. Put the commit in Commit instead.
var Version string = "0.40.2"

// Commit is an optional git describe / SHA from -ldflags. Operators see it on
// /api/version and `zerollama -v`. It is never sent to registry.ollama.ai.
var Commit string = ""

// Semver returns a gallery-safe ollama version. If Version was overwritten
// with a git hash (the Pangolin-class footgun), fall back to the default.
func Semver() string {
	if looksLikeOllamaSemver(Version) {
		return Version
	}
	return "0.40.2"
}

// HTTPUserAgent is the registry and API User-Agent. Always ollama/<semver>.
func HTTPUserAgent() string {
	return fmt.Sprintf("ollama/%s (%s %s) Go/%s", Semver(), runtime.GOARCH, runtime.GOOS, runtime.Version())
}

func looksLikeOllamaSemver(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	// "0.40.2" or "0.40.2-rc1"; reject "cdf3c93" / "0.30.11-12-gabcdef".
	if strings.Contains(s, "-g") || strings.Contains(s, "-dirty") {
		return false
	}
	parts := strings.Split(s, ".")
	if len(parts) < 2 || len(parts) > 3 {
		return false
	}
	for i, p := range parts {
		if p == "" {
			return false
		}
		if i == len(parts)-1 {
			p, _, _ = strings.Cut(p, "-")
			p, _, _ = strings.Cut(p, "+")
		}
		for _, r := range p {
			if !unicode.IsDigit(r) {
				return false
			}
		}
	}
	return unicode.IsDigit(rune(parts[0][0]))
}
