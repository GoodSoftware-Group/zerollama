package modality

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

func TestSoundtrackAbsent(t *testing.T) {
	if !soundtrackAbsent("Output file #0 does not contain any stream") {
		t.Fatal("expected no-stream stderr to count as silence")
	}
	if !soundtrackAbsent("Stream map '0:a' matches no streams") {
		t.Fatal("expected map miss to count as silence")
	}
	if soundtrackAbsent("Invalid data found when processing input") {
		t.Fatal("a decode error is not silence")
	}
}

func TestAttachVideoSoundtracks(t *testing.T) {
	ffmpeg := envconfig.FFmpegBin()
	if _, err := exec.LookPath(ffmpeg); err != nil {
		t.Skipf("ffmpeg not installed (%s)", ffmpeg)
	}
	dir := t.TempDir()
	silent := filepath.Join(dir, "silent.mp4")
	heard := filepath.Join(dir, "heard.mp4")
	gen := func(path string, extra ...string) {
		t.Helper()
		args := []string{"-hide_banner", "-loglevel", "error", "-y",
			"-f", "lavfi", "-i", "testsrc=duration=1:size=32x32:rate=5"}
		args = append(args, extra...)
		args = append(args, "-pix_fmt", "yuv420p", "-shortest", path)
		if out, err := exec.Command(ffmpeg, args...).CombinedOutput(); err != nil {
			t.Fatalf("fixture %s: %v: %s", path, err, out)
		}
	}
	gen(silent)
	gen(heard, "-f", "lavfi", "-i", "sine=frequency=440:duration=1")

	read := func(path string) []byte {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	req := &api.ChatRequest{Messages: []api.Message{{
		Videos: []api.VideoData{read(silent), read(heard)},
	}}}
	if err := AttachVideoSoundtracks(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages[0].AudioClips) != 1 {
		t.Fatalf("clips %d, want the sine track only", len(req.Messages[0].AudioClips))
	}
	if _, ok := llm.AudioFormat(req.Messages[0].AudioClips[0]); !ok {
		t.Fatal("soundtrack is not a wav")
	}
	if len(req.Messages[0].Videos) != 2 {
		t.Fatal("soundtrack extract should keep the containers for frame sampling")
	}
}
