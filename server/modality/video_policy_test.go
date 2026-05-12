package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestBuildFFmpegVideoFilter(t *testing.T) {
	t.Parallel()
	fps := BuildFFmpegVideoFilter(VideoSamplingPolicy{Mode: model.VideoSampleModeFPS, FPS: 2.5})
	if want := "fps=2.5"; fps != want {
		t.Fatalf("fps mode: got %q want %q", fps, want)
	}
	str := BuildFFmpegVideoFilter(VideoSamplingPolicy{Mode: model.VideoSampleModeStride, Stride: 30})
	if want := "select='not(mod(n\\,30))',setpts=N/FRAME_RATE/TB"; str != want {
		t.Fatalf("stride mode: got %q want %q", str, want)
	}
}

func TestResolveVideoPolicy_manifest(t *testing.T) {
	t.Parallel()
	cfg := model.ConfigV2{
		VideoSampling: &model.VideoSampling{
			Mode:      model.VideoSampleModeStride,
			Stride:    10,
			MaxFrames: 8,
		},
		TokensPerImage: 512,
	}
	p := ResolveVideoPolicy(cfg)
	if p.Mode != model.VideoSampleModeStride || p.Stride != 10 || p.MaxFrames != 8 {
		t.Fatalf("unexpected policy: %+v", p)
	}
	if p.TokensPerImage != 512 || !p.ManifestOverride {
		t.Fatalf("tokens / manifest: %+v", p)
	}

	onlyStride := model.ConfigV2{VideoSampling: &model.VideoSampling{Stride: 12}}
	p2 := ResolveVideoPolicy(onlyStride)
	if p2.Mode != model.VideoSampleModeStride || p2.Stride != 12 {
		t.Fatalf("stride without mode should select stride: %+v", p2)
	}

	badMode := model.ConfigV2{VideoSampling: &model.VideoSampling{Mode: "typo"}}
	p3 := ResolveVideoPolicy(badMode)
	if p3.Mode != model.VideoSampleModeFPS {
		t.Fatalf("invalid mode should fall back to fps: %+v", p3)
	}
}

func TestPreflightVideoVisionBudget(t *testing.T) {
	t.Parallel()
	policy := VideoSamplingPolicy{MaxFrames: 32, TokensPerImage: 768}
	req := &api.ChatRequest{
		Messages: []api.Message{
			{Videos: []api.VideoData{make(api.VideoData, 1)}},
		},
	}
	// 1 video * 32 * 768 = 24576 > 2048
	if err := PreflightVideoVisionBudget(policy, 2048, req); err == nil {
		t.Fatal("expected error when vision estimate exceeds num_ctx")
	}
	if err := PreflightVideoVisionBudget(policy, 50000, req); err != nil {
		t.Fatal(err)
	}

	// Long history with many images but no video on those turns must not trip preflight alone.
	heavyHistory := &api.ChatRequest{
		Messages: []api.Message{
			{Role: "user", Images: make([]api.ImageData, 100)},
			{Role: "assistant", Content: "ok"},
			{Videos: []api.VideoData{make(api.VideoData, 1)}},
		},
	}
	if err := PreflightVideoVisionBudget(policy, 50000, heavyHistory); err != nil {
		t.Fatalf("preflight should ignore images on messages without video: %v", err)
	}
}
