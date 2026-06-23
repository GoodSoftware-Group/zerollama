package modality

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// Phase D partial: real ffmpeg round-trip without checked-in MP4 fixtures.
// Skips when ffmpeg is not on PATH (CI minimal trees, dev laptops without ffmpeg).
func TestSampleVideoToPNGs_ffmpegGolden(t *testing.T) {
	ffmpeg := envconfig.FFmpegBin()
	if _, err := exec.LookPath(ffmpeg); err != nil {
		t.Skipf("ffmpeg not installed (%s)", ffmpeg)
	}

	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	dir := t.TempDir()
	mp4 := filepath.Join(dir, "lavfi.mp4")
	cmd := exec.Command(ffmpeg,
		"-hide_banner", "-loglevel", "error", "-y",
		"-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
		"-pix_fmt", "yuv420p", mp4,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("generate fixture: %v: %s", err, out)
	}
	data, err := os.ReadFile(mp4)
	if err != nil {
		t.Fatal(err)
	}

	policy := VideoSamplingPolicy{
		Mode:          model.VideoSampleModeFPS,
		FPS:           1,
		MaxFrames:     3,
		MaxBytes:      1 << 20,
		FFmpegTimeout: 30 * time.Second,
	}
	frames, err := sampleVideoToPNGs(context.Background(), policy, "", data, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames.frames) != 3 {
		t.Fatalf("frame count=%d want 3 (2s @ 1fps capped by max_frames)", len(frames.frames))
	}
	for i, f := range frames.frames {
		if len(f) < 8 || f[0] != 0x89 || f[1] != 0x50 {
			t.Fatalf("frame %d: not PNG magic", i)
		}
	}

	req := &api.ChatRequest{
		Messages: []api.Message{{Videos: []api.VideoData{data}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages[0].VideoSpans) != 1 {
		t.Fatal("expected one video span")
	}
	sp := req.Messages[0].VideoSpans[0]
	if sp.FrameCount != 3 {
		t.Fatalf("frame_count=%d want 3", sp.FrameCount)
	}
	if len(sp.GridTHW) != 3 || sp.GridTHW[0] != 3 {
		t.Fatalf("expected native grid_thw from ffmpeg frames, got %v", sp.GridTHW)
	}

	// Second sample should hit global expand cache (no second ffmpeg run).
	frames2, err := sampleVideoToPNGs(context.Background(), policy, "", data, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames2.frames) != 3 {
		t.Fatalf("cached frame count=%d want 3", len(frames2.frames))
	}
}
