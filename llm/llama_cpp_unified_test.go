package llm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsLegacyLlamaCppCheckout(t *testing.T) {
	if !IsLegacyLlamaCppCheckout("/Users/x/Sites/inference/eliza-llama.cpp") {
		t.Fatal("expected eliza-llama.cpp to be legacy")
	}
	if IsLegacyLlamaCppCheckout("/Users/x/Sites/inference/llama.cpp") {
		t.Fatal("expected llama.cpp to not be legacy")
	}
}

func TestUnifiedLlamaCppRootEnv(t *testing.T) {
	t.Setenv("LLAMA_CPP_ROOT", "/tmp/unified-llama-cpp-test")
	got := UnifiedLlamaCppRoot()
	if got != "/tmp/unified-llama-cpp-test" {
		t.Fatalf("UnifiedLlamaCppRoot = %q", got)
	}
}

func TestLlamaCppUnificationReportLegacyEnv(t *testing.T) {
	root := t.TempDir()
	legacy := filepath.Join(filepath.Dir(root), "eliza-llama.cpp")
	t.Setenv("LLAMA_CPP_ROOT", legacy)
	t.Setenv("LLAMA_SERVER_BIN", filepath.Join(legacy, "build", "bin", "llama-server"))

	report := LlamaCppUnificationReport()
	if !report.LegacyCheckout {
		t.Fatal("expected legacy checkout warning")
	}
	if !report.Warn {
		t.Fatal("expected warn")
	}
	if !report.ServerUnderRoot {
		t.Fatal("server should be under configured root")
	}
}

func TestLlamaCppPathUsesLegacyCheckout(t *testing.T) {
	if !LlamaCppPathUsesLegacyCheckout("/x/eliza-llama.cpp/build/bin/llama-server") {
		t.Fatal("expected legacy path")
	}
}

func TestApplyUnifiedLlamaCppEnvRedirectsLegacyServerBin(t *testing.T) {
	unified := t.TempDir()
	legacyDir := filepath.Join(filepath.Dir(unified), "eliza-llama.cpp")
	if err := os.MkdirAll(filepath.Join(legacyDir, "build", "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	legacyBin := filepath.Join(legacyDir, "build", "bin", "llama-server")
	if err := os.WriteFile(legacyBin, []byte("stub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(unified, "build", "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	unifiedBin := filepath.Join(unified, "build", "bin", "llama-server")
	if err := os.WriteFile(unifiedBin, []byte("ok"), 0o755); err != nil {
		t.Fatal(err)
	}

	t.Setenv("LLAMA_CPP_ROOT", unified)
	t.Setenv("LLAMA_SERVER_BIN", legacyBin)

	msgs := ApplyUnifiedLlamaCppEnv()
	if len(msgs) == 0 {
		t.Fatal("expected migration messages")
	}
	if got := os.Getenv("LLAMA_SERVER_BIN"); got != unifiedBin {
		t.Fatalf("LLAMA_SERVER_BIN = %q, want %q", got, unifiedBin)
	}
}

func TestApplyUnifiedLlamaCppEnvRedirectsSiblingToVendor(t *testing.T) {
	vendor := vendorLlamaCppRoot()
	if vendor == "" {
		t.Skip("no patched vendor checkout")
	}
	sibling := siblingLlamaCppRoot()
	if sibling == "" {
		t.Skip("no sibling path")
	}
	if err := os.MkdirAll(filepath.Join(vendor, "build", "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	vendorBin := filepath.Join(vendor, "build", "bin", "llama-server")
	if err := os.WriteFile(vendorBin, []byte("ok"), 0o755); err != nil {
		t.Fatal(err)
	}

	t.Setenv("LLAMA_CPP_ROOT", sibling)
	t.Setenv("LLAMA_SERVER_BIN", filepath.Join(sibling, "build", "bin", "llama-server"))

	msgs := ApplyUnifiedLlamaCppEnv()
	if got := os.Getenv("LLAMA_CPP_ROOT"); got != vendor {
		t.Fatalf("LLAMA_CPP_ROOT = %q, want vendor %q (msgs=%v)", got, vendor, msgs)
	}
	if got := os.Getenv("LLAMA_SERVER_BIN"); got != vendorBin {
		t.Fatalf("LLAMA_SERVER_BIN = %q, want %q", got, vendorBin)
	}
}

func TestPinnedLlamaCppCommitFallback(t *testing.T) {
	// Without repo markers in temp cwd, still returns default pin.
	dir := t.TempDir()
	t.Chdir(dir)
	_ = os.Unsetenv("LLAMA_CPP_ROOT")
	if got := PinnedLlamaCppCommit(); got != defaultUnifiedLlamaCppCommit {
		t.Fatalf("PinnedLlamaCppCommit = %q", got)
	}
}

func TestPinRefsMatch(t *testing.T) {
	if !pinRefsMatch("c84b3020", "c84b30200c8d512c00c9d61c96bed078f1c0024d") {
		t.Fatal("expected short/full pin match")
	}
	if pinRefsMatch("b9781", "c84b30200c8d512c00c9d61c96bed078f1c0024d") {
		t.Fatal("expected mismatch")
	}
}

func TestVendorLlamaCppRootPrefersPatchedVendor(t *testing.T) {
	repo := zerollamaRepoRoot()
	if repo == "" {
		t.Skip("not in zerollama repo")
	}
	t.Setenv("LLAMA_CPP_ROOT", "")
	got := UnifiedLlamaCppRoot()
	vendor := filepath.Join(repo, "vendor", "llama-cpp-"+readVendorLlamaCppPin())
	if _, err := os.Stat(filepath.Join(vendor, "CMakeLists.txt")); err == nil {
		if got != vendor {
			t.Fatalf("UnifiedLlamaCppRoot = %q, want vendor %q", got, vendor)
		}
	}
}
