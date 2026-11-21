package voicepipe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDumpEarWAVWritesClipAndTranscript(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(earDumpEnv, dir)
	wav, err := WrapPCM16(16000, make([]byte, 16000*2/5))
	if err != nil {
		t.Fatal(err)
	}
	path := DumpEarWAV("guess", wav)
	if path == "" || !strings.Contains(filepath.Base(path), "-guess-") {
		t.Fatalf("path %q", path)
	}
	if st, err := os.Stat(path); err != nil {
		t.Fatal(err)
	} else if st.Mode().Perm() != 0o600 {
		t.Fatalf("wav mode %o", st.Mode().Perm())
	}
	NoteEar(path, "Hello.", "qwen3-asr", true, nil)
	notePath := strings.TrimSuffix(path, ".wav") + ".txt"
	note, err := os.ReadFile(notePath)
	if err != nil {
		t.Fatal(err)
	}
	text := string(note)
	if !strings.Contains(text, "transcript: Hello.") || !strings.Contains(text, "transcribed: true") || !strings.Contains(text, "backend: qwen3-asr") {
		t.Fatalf("note %q", text)
	}
	if st, err := os.Stat(notePath); err != nil {
		t.Fatal(err)
	} else if st.Mode().Perm() != 0o600 {
		t.Fatalf("note mode %o", st.Mode().Perm())
	}
}

func TestDumpEarWAVDisabled(t *testing.T) {
	t.Setenv(earDumpEnv, "")
	wav, err := WrapPCM16(16000, make([]byte, 320))
	if err != nil {
		t.Fatal(err)
	}
	if path := DumpEarWAV("commit", wav); path != "" {
		t.Fatalf("wrote %s with dump disabled", path)
	}
}
