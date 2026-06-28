package modality

import (
	"bytes"
	"context"
	"image"
	"image/png"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestVisionTokensFromGridTHW_qwenStyle(t *testing.T) {
	t.Parallel()
	// [4, 24, 32] with merge=2 → 4*24*32/4 = 768
	got := VisionTokensFromGridTHW([]int{4, 24, 32}, 2)
	if got != 768 {
		t.Fatalf("got %d want 768", got)
	}
}

func TestValidateVideoSpanGridTHW(t *testing.T) {
	t.Parallel()
	if err := validateVideoSpanGridTHW(api.VideoSpan{FrameCount: 4, GridTHW: []int{4, 24, 32}}); err != nil {
		t.Fatal(err)
	}
	if err := validateVideoSpanGridTHW(api.VideoSpan{FrameCount: 3, GridTHW: []int{4, 24, 32}}); err == nil {
		t.Fatal("expected frame_count mismatch error")
	}
	if err := validateVideoSpanGridTHW(api.VideoSpan{GridTHW: []int{1, 2}}); err == nil {
		t.Fatal("expected len error")
	}
}

func TestEstimateMultimodalTokens_gridTHW(t *testing.T) {
	t.Parallel()
	policy := VideoSamplingPolicy{TokensPerImage: 768}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Role: "user",
			Images: make([]api.ImageData, 4),
			VideoSpans: []api.VideoSpan{{
				FrameCount: 4,
				GridTHW:    []int{4, 24, 32},
			}},
		}},
	}
	got := EstimateMultimodalTokens(policy, req)
	if got.VideoTokens != 768 {
		t.Fatalf("video_tokens=%d want 768 from grid_thw", got.VideoTokens)
	}
	if got.ImageTokens != 0 {
		t.Fatalf("image_tokens=%d want 0", got.ImageTokens)
	}
}

func TestPreflightVideoVisionBudget_gridTHW(t *testing.T) {
	policy := VideoSamplingPolicy{TokensPerImage: 768}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Role:       "user",
			Images:     make([]api.ImageData, 4),
			VideoSpans: []api.VideoSpan{{FrameCount: 4, GridTHW: []int{4, 24, 32}}},
		}},
	}
	if err := PreflightVideoVisionBudget(policy, 700, req); err == nil {
		t.Fatal("expected exceed at 700")
	}
	if err := PreflightVideoVisionBudget(policy, 800, req); err != nil {
		t.Fatalf("expected pass at 800: %v", err)
	}
}

func TestComputeVideoGridTHWFromFrames_64x64(t *testing.T) {
	t.Parallel()
	// Minimal 64×64 RGBA PNG (smart resize → 56×56 → grid 4×4 per frame).
	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	frames := []api.ImageData{buf.Bytes(), buf.Bytes()}
	grid := computeVideoGridTHWFromFrames(frames, VideoSamplingPolicy{})
	if len(grid) != 3 {
		t.Fatalf("grid=%v", grid)
	}
	// 64×64 triggers shortest_edge upscale in Qwen smart resize → 280×280 → grid 20×20.
	if grid[0] != 2 || grid[1] != 20 || grid[2] != 20 {
		t.Fatalf("grid=%v want [2,20,20]", grid)
	}
	tokens := VisionTokensFromGridTHW(grid, 2)
	if tokens != 200 {
		t.Fatalf("tokens=%d want 200", tokens)
	}
}

func TestExpandVideosInChatRequest_setsGridTHWFromHook(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	pngFrame := buf.Bytes()

	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		return []api.ImageData{pngFrame, pngFrame}, nil
	}

	policy := VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	req := &api.ChatRequest{
		Messages: []api.Message{{Videos: []api.VideoData{{1}}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages[0].VideoSpans) != 1 {
		t.Fatal("expected one span")
	}
	sp := req.Messages[0].VideoSpans[0]
	if sp.FrameCount != 2 {
		t.Fatalf("frame_count=%d", sp.FrameCount)
	}
	if len(sp.GridTHW) != 3 || sp.GridTHW[0] != 2 {
		t.Fatalf("grid_thw=%v", sp.GridTHW)
	}
	if sp.GridTHWExplicit {
		t.Fatal("ffmpeg expand must not mark grid as client-explicit")
	}
	got := GridTHWPerRaster(req.Messages[0])
	for _, g := range got {
		if g != nil {
			t.Fatalf("server estimate must not forward to runner: %v", got)
		}
	}
}

func TestVideoExpandCache_preservesGridTHW(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	pngFrame := buf.Bytes()
	grid := []int{2, 20, 20}

	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		return []api.ImageData{pngFrame, pngFrame}, nil
	}

	policy := VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	video := []byte{0x01}
	req1 := &api.ChatRequest{
		Options:  map[string]any{"prompt_cache_key": "grid-cache-1"},
		Messages: []api.Message{{Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req1); err != nil {
		t.Fatal(err)
	}
	if len(req1.Messages[0].VideoSpans[0].GridTHW) != 3 {
		t.Fatalf("turn1 grid=%v", req1.Messages[0].VideoSpans[0].GridTHW)
	}

	// Force global eviction; session cache should still return cached grid without PNG decode.
	resetVideoExpandCache()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		t.Fatal("decode should not run on session cache hit")
		return nil, nil
	}
	req2 := &api.ChatRequest{
		Options:  map[string]any{"prompt_cache_key": "grid-cache-1"},
		Messages: []api.Message{{Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req2); err != nil {
		t.Fatal(err)
	}
	got := req2.Messages[0].VideoSpans[0].GridTHW
	if len(got) != 3 || got[0] != grid[0] || got[1] != grid[1] || got[2] != grid[2] {
		t.Fatalf("turn2 grid=%v want %v", got, grid)
	}
}

func TestResolveVideoPolicy_visionGridManifest(t *testing.T) {
	cfg := model.ConfigV2{
		VisionPatchSize:        16,
		VisionSpatialMergeSize: 4,
	}
	p := ResolveVideoPolicy(cfg)
	if p.visionPatchSize() != 16 || p.visionSpatialMergeSize() != 4 {
		t.Fatalf("patch=%d merge=%d", p.visionPatchSize(), p.visionSpatialMergeSize())
	}
	if p.visionGridFactor() != 64 {
		t.Fatalf("factor=%d want 64", p.visionGridFactor())
	}
}

func TestValidatePreexpandedVideoMessage_gridTHW(t *testing.T) {
	msg := api.Message{
		Images:     make([]api.ImageData, 4),
		VideoSpans: []api.VideoSpan{{FrameCount: 4, GridTHW: []int{4, 24, 32}}},
	}
	if err := validatePreexpandedVideoMessage(&msg); err != nil {
		t.Fatal(err)
	}
	if !msg.VideoSpans[0].GridTHWExplicit {
		t.Fatal("expected GridTHWExplicit after client pre-expanded validate")
	}
	bad := api.Message{
		Images:     make([]api.ImageData, 4),
		VideoSpans: []api.VideoSpan{{FrameCount: 4, GridTHW: []int{3, 24, 32}}},
	}
	if err := validatePreexpandedVideoMessage(&bad); err == nil {
		t.Fatal("expected grid/frame mismatch")
	}
}
