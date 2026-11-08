package kokoro

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

func TestVoiceIDsFromBin(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "voices.bin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for _, name := range []string{"af_heart.npy", "bf_emma.npy", "subdir/am_adam.npy"} {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte{0}); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()
	got := VoiceIDsFromBin(path)
	want := []string{"af_heart", "am_adam", "bf_emma"}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestVoiceIDsFromBinMissing(t *testing.T) {
	if VoiceIDsFromBin(filepath.Join(t.TempDir(), "nope.bin")) != nil {
		t.Fatal("expected nil")
	}
}
