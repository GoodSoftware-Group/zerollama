package gpatch_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/weightedit"
	_ "github.com/ollama/ollama/x/weightedit/gpatch"
)

func testdata(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(file), "testdata")
}

func TestFixtureApplyRestore(t *testing.T) {
	dir := testdata(t)
	patchPath := filepath.Join(dir, "fixture.gpatch")
	ckpt := filepath.Join(dir, "checkpoint")
	f, err := os.Open(patchPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	edit, codec, err := weightedit.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	if codec.Format() != "geodesic-patch" {
		t.Fatalf("format=%s", codec.Format())
	}
	if edit.ID == "" {
		t.Fatal("empty patch id")
	}
	if edit.ChangedWeights() != 2 {
		t.Fatalf("changed=%d", edit.ChangedWeights())
	}

	store, err := weightedit.OpenDirectory(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := weightedit.Bind(store, edit, "source")
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "patched")
	if _, err := weightedit.Apply(bound, out); err != nil {
		t.Fatal(err)
	}
	patched, err := weightedit.OpenDirectory(out)
	if err != nil {
		t.Fatal(err)
	}
	restored := filepath.Join(t.TempDir(), "restored")
	if _, err := weightedit.Restore(patched, edit, restored); err != nil {
		t.Fatal(err)
	}
}
