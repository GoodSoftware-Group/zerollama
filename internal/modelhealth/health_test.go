package modelhealth

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBlobAccessible(t *testing.T) {
	dir := t.TempDir()
	okPath := filepath.Join(dir, "ok")
	if err := os.WriteFile(okPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !blobAccessible(okPath) {
		t.Fatal("expected existing file accessible")
	}

	missing := filepath.Join(dir, "missing")
	if blobAccessible(missing) {
		t.Fatal("expected missing file inaccessible")
	}

	link := filepath.Join(dir, "broken-link")
	if err := os.Symlink(filepath.Join(dir, "nope"), link); err != nil {
		t.Fatal(err)
	}
	if blobAccessible(link) {
		t.Fatal("expected broken symlink inaccessible")
	}
}

func TestIsBenchable(t *testing.T) {
	if !IsBenchable(Report{Status: StatusOK}) {
		t.Fatal("ok should be benchable")
	}
	if !IsBenchable(Report{Status: StatusRepairable}) {
		t.Fatal("repairable should be benchable (sync may fix)")
	}
	if IsBenchable(Report{Status: StatusOrphaned}) {
		t.Fatal("orphaned should not be benchable")
	}
}
