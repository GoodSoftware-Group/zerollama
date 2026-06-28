package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDoctorCheckLlamaPatchesInTree(t *testing.T) {
	repo := doctorRepoRoot()
	if repo == "" {
		t.Skip("no repo root")
	}
	c := doctorCheckLlamaPatches(repo)
	if c.Status == "fail" {
		t.Fatalf("doctor llama patches: %s (%s)", c.Detail, c.FixHint)
	}
	if !strings.Contains(c.Detail, "/kv/seq-copy") {
		t.Fatalf("expected seq-copy in detail, got %q", c.Detail)
	}
}

func TestDoctorExpectedVendorHead(t *testing.T) {
	repo := doctorRepoRoot()
	head := doctorExpectedVendorHead(repo)
	if head == "" {
		t.Fatal("LLAMA_CPP_VENDOR_HEAD missing or empty")
	}
	if len(head) < 40 {
		t.Fatalf("unexpected vendor head: %q", head)
	}
}

func TestDoctorBinaryEmbedsSeqCopy(t *testing.T) {
	repo := doctorRepoRoot()
	fetchHead := doctorMakefileSyncFetchHead(repo)
	vendorBin := filepath.Join(repo, "vendor", "llama-cpp-"+fetchHead, "build", "bin", "llama-server")
	if st, err := os.Stat(vendorBin); err != nil || st.IsDir() {
		t.Skip("vendor llama-server not built")
	}
	if !doctorBinaryEmbedsSeqCopy(vendorBin) {
		t.Fatal("vendor llama-server missing /kv/seq-copy string")
	}
}
