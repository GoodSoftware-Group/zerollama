package llm

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFakeLlamaServer(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	// Mach-O magic + pad to satisfy isUsableLlamaServerBin size gate.
	buf := make([]byte, 1<<20)
	copy(buf, []byte{0xCF, 0xFA, 0xED, 0xFE})
	if err := os.WriteFile(path, buf, 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestIsLegacyLlamaCppCheckout(t *testing.T) {
	if !IsLegacyLlamaCppCheckout("/Users/x/Sites/inference/eliza-llama.cpp") {
		t.Fatal("expected eliza-llama.cpp to be legacy")
	}
	if IsLegacyLlamaCppCheckout("/Users/x/Sites/inference/llama.cpp") {
		t.Fatal("expected llama.cpp to not be legacy")
	}
}

func TestIsUsableLlamaServerBinRejectsStub(t *testing.T) {
	dir := t.TempDir()
	tiny := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(tiny, []byte{0xCF, 0xFA, 0xED, 0xFE, 'x'}, 0o755); err != nil {
		t.Fatal(err)
	}
	if isUsableLlamaServerBin(tiny) {
		t.Fatal("tiny mach-o stub should be rejected")
	}
	realish := filepath.Join(dir, "llama-server-big")
	writeFakeLlamaServer(t, realish)
	if !isUsableLlamaServerBin(realish) {
		t.Fatal("1MiB mach-o fixture should be accepted")
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
	legacyBin := filepath.Join(legacyDir, "build", "bin", "llama-server")
	writeFakeLlamaServer(t, legacyBin)
	unifiedBin := filepath.Join(unified, "build", "bin", "llama-server")
	writeFakeLlamaServer(t, unifiedBin)

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
	vendorBin := filepath.Join(vendor, "build", "bin", "llama-server")
	// Do not clobber a real multi-MB build; only install a fixture if absent/unusable.
	hadReal := isUsableLlamaServerBin(vendorBin)
	if !hadReal {
		writeFakeLlamaServer(t, vendorBin)
		t.Cleanup(func() { _ = os.Remove(vendorBin) })
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

func TestFindVendorLlamaServerFallback(t *testing.T) {
	path, ok := findVendorLlamaServerFallback()
	if !ok {
		t.Skip("no built vendor llama-server on this machine")
	}
	if !isUsableLlamaServerBin(path) {
		t.Fatalf("fallback %q is not usable", path)
	}
}
