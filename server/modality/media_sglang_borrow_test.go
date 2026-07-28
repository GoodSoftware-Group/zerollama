package modality

import (
	"context"
	"errors"
	"net/http"
	"os/exec"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestMediaHTTPStatus_taxonomy(t *testing.T) {
	if MediaHTTPStatus(ClientMedia("bad clip")) != http.StatusBadRequest {
		t.Fatal("client → 400")
	}
	if MediaHTTPStatus(ServerMediaf("missing ffmpeg: %v", exec.ErrNotFound)) != http.StatusInternalServerError {
		t.Fatal("server → 500")
	}
	if MediaHTTPStatus(errors.New("legacy string")) != http.StatusBadRequest {
		t.Fatal("unknown expand errors default to 400")
	}
}

func TestExpandVideos_emptyIsClient(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		t.Fatal("hook should not run for empty video")
		return nil, nil
	}
	err := ExpandVideosInChatRequest(context.Background(), VideoSamplingPolicy{
		MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20,
	}, &api.ChatRequest{Messages: []api.Message{{Videos: []api.VideoData{{}}}}})
	if !IsClientMedia(err) {
		t.Fatalf("want ClientMediaError, got %T %v", err, err)
	}
	if MediaHTTPStatus(err) != http.StatusBadRequest {
		t.Fatalf("status=%d", MediaHTTPStatus(err))
	}
}

func TestExpandVideos_parallelPreservesOrder(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	var concurrent atomic.Int32
	var maxConcurrent atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		n := concurrent.Add(1)
		for {
			cur := maxConcurrent.Load()
			if n <= cur || maxConcurrent.CompareAndSwap(cur, n) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		concurrent.Add(-1)
		id := data[0]
		return []api.ImageData{{id}}, nil
	}

	t.Setenv("OLLAMA_MM_IO_WORKERS", "4")
	policy := VideoSamplingPolicy{
		Mode: "fps", FPS: 1, MaxFrames: 8,
		MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20,
	}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Videos: []api.VideoData{{1}, {2}, {3}},
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages[0].Images) != 3 {
		t.Fatalf("images=%d", len(req.Messages[0].Images))
	}
	for i, img := range req.Messages[0].Images {
		if len(img) == 0 || img[0] != byte(i+1) {
			t.Fatalf("image[%d]=%v want id %d", i, img, i+1)
		}
	}
	if maxConcurrent.Load() < 2 {
		t.Fatalf("expected parallel decode, maxConcurrent=%d", maxConcurrent.Load())
	}
}

func TestIsAudioContainer(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want bool
	}{
		{"empty", nil, false},
		{"wav", []byte("RIFF....WAVE"), false},
		{"mp4", append(make([]byte, 4), []byte("ftypisom")...), true},
		{"webm", []byte{0x1a, 0x45, 0xdf, 0xa3, 0x00}, true},
		{"avi", []byte("RIFF....AVI "), true},
		{"amr", []byte("#!AMR\nxxxx"), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsAudioContainer(tc.data); got != tc.want {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestExpandAudioClips_leavesWAV(t *testing.T) {
	// Minimal RIFF/WAVE header (12+ bytes) — AudioFormat accepts it.
	wav := []byte("RIFF....WAVE")
	req := &api.ChatRequest{Messages: []api.Message{{AudioClips: []api.AudioData{wav}}}}
	if err := ExpandAudioClipsInChatRequest(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if string(req.Messages[0].AudioClips[0]) != string(wav) {
		t.Fatal("wav should be unchanged")
	}
}

func TestExpandAudioClips_emptyIsClient(t *testing.T) {
	req := &api.ChatRequest{Messages: []api.Message{{AudioClips: []api.AudioData{{}}}}}
	err := ExpandAudioClipsInChatRequest(context.Background(), req)
	if !IsClientMedia(err) {
		t.Fatalf("got %T %v", err, err)
	}
}

func TestDemuxAudioContainer_webmWithTone(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	// 0.2s sine in WebM/Opus — browser MediaRecorder-shaped container.
	cmd := exec.Command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
		"-f", "lavfi", "-i", "sine=frequency=440:duration=0.2",
		"-c:a", "libopus", "-f", "webm", "pipe:1")
	webm, err := cmd.Output()
	if err != nil {
		t.Skipf("could not synthesize webm: %v", err)
	}
	if !IsAudioContainer(webm) {
		t.Fatal("synthesized webm should sniff as container")
	}
	req := &api.ChatRequest{Messages: []api.Message{{AudioClips: []api.AudioData{webm}}}}
	if err := ExpandAudioClipsInChatRequest(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	out := req.Messages[0].AudioClips[0]
	if format, ok := llm.AudioFormat(out); !ok || format != "wav" {
		t.Fatalf("demuxed format=%q ok=%v len=%d", format, ok, len(out))
	}
}

func TestDemuxAudioContainer_videoOnlyIsClient(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	cmd := exec.Command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
		"-f", "lavfi", "-i", "color=c=black:s=32x32:d=0.2",
		"-an", "-c:v", "libvpx", "-f", "webm", "pipe:1")
	webm, err := cmd.Output()
	if err != nil {
		t.Skipf("could not synthesize video-only webm: %v", err)
	}
	req := &api.ChatRequest{Messages: []api.Message{{AudioClips: []api.AudioData{webm}}}}
	err = ExpandAudioClipsInChatRequest(context.Background(), req)
	if !IsClientMedia(err) {
		t.Fatalf("want client error, got %T %v", err, err)
	}
	if MediaHTTPStatus(err) != http.StatusBadRequest {
		t.Fatalf("status=%d", MediaHTTPStatus(err))
	}
}
