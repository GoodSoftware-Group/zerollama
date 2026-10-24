package version

import (
	"strings"
	"testing"
)

func TestLooksLikeOllamaSemver(t *testing.T) {
	ok := []string{"0.40.2", "0.30.11", "0.6.5", "0.40.2-rc1", "1.0", OllamaCompat}
	for _, s := range ok {
		if !looksLikeOllamaSemver(s) {
			t.Errorf("looksLikeOllamaSemver(%q) = false, want true", s)
		}
	}
	bad := []string{"", "cdf3c93", "v0.40.2", "0.30.11-12-gcdf3c93", "0.40.2-dirty", "ollama"}
	for _, s := range bad {
		if looksLikeOllamaSemver(s) {
			t.Errorf("looksLikeOllamaSemver(%q) = true, want false", s)
		}
	}
}

func TestSemverFallsBackFromGitHash(t *testing.T) {
	old := Version
	t.Cleanup(func() { Version = old })
	Version = "cdf3c93"
	if got := Semver(); got != OllamaCompat {
		t.Fatalf("Semver() = %q, want OllamaCompat %q", got, OllamaCompat)
	}
	ua := HTTPUserAgent()
	wantPrefix := "ollama/" + OllamaCompat + " "
	if !strings.HasPrefix(ua, wantPrefix) {
		t.Fatalf("HTTPUserAgent() = %q, want prefix %q", ua, wantPrefix)
	}
	if strings.Contains(ua, "cdf3c93") {
		t.Fatalf("User-Agent leaked git hash: %q", ua)
	}
}

func TestSemverKeepsExplicit(t *testing.T) {
	old := Version
	t.Cleanup(func() { Version = old })
	Version = "0.40.2"
	if got := Semver(); got != "0.40.2" {
		t.Fatalf("Semver() = %q, want 0.40.2", got)
	}
}

func TestOllamaCompatIsGallerySafe(t *testing.T) {
	if !looksLikeOllamaSemver(OllamaCompat) {
		t.Fatalf("OllamaCompat %q is not a gallery-safe semver", OllamaCompat)
	}
	if Version != OllamaCompat && looksLikeOllamaSemver(Version) {
		// Default package init: Version tracks OllamaCompat until ldflags.
		t.Logf("Version=%q OllamaCompat=%q (ldflag override OK if semver)", Version, OllamaCompat)
	}
}
