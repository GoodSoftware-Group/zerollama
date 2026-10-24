package version

import (
	"fmt"
	"runtime"
	"strings"
	"unicode"
)

// OllamaCompat is the Ollama gallery / registry compatibility semver.
//
// registry.ollama.ai and related endpoints gate pulls on User-Agent
// `ollama/<semver>` (HTTP 412 for git hashes, 0.0.x, or too-old clients).
// New gallery models (decision/System One, embeddings, …) often require a
// current upstream release stamp even when our fork tip lags the tag.
//
// Bump this when syncing/merging from upstream ollama (`../ollama-upstream`
// or a release tag) to the newest semver our pull/create/load paths can
// speak. Prefer the latest upstream *release* tag (e.g. v0.40.10 → "0.40.10"),
// not an rc, unless we deliberately need that client gate.
//
// Never ldflag a git hash into this value. Put git describe in Commit.
const OllamaCompat = "0.40.10"

// Version is the product stamp (`zerollama -v`, GET /api/version "version").
// Defaults to OllamaCompat so operators and the registry see one coherent
// number. May be overwritten via -ldflags only with another valid ollama-style
// semver — never with `git describe` (use Commit for that).
var Version string = OllamaCompat

// Commit is an optional git describe / SHA from -ldflags. Operators see it on
// /api/version and `zerollama -v`. It is never sent to registry.ollama.ai.
var Commit string = ""

// Semver returns the gallery-safe ollama version used in User-Agent and
// /api/version. If Version was overwritten with a git hash (the Pangolin-class
// footgun), fall back to OllamaCompat.
func Semver() string {
	if looksLikeOllamaSemver(Version) {
		return Version
	}
	return OllamaCompat
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
	// "0.40.10" or "0.40.10-rc1"; reject "cdf3c93" / "0.30.11-12-gabcdef".
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
