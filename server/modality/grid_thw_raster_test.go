package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGridTHWPerRaster_stillsAndVideoFrames(t *testing.T) {
	msg := api.Message{
		Images: make([]api.ImageData, 6),
		VideoSpans: []api.VideoSpan{
			{FrameCount: 3, GridTHW: []int{3, 24, 32}, GridTHWExplicit: true},
			{FrameCount: 1, GridTHW: []int{1, 16, 16}, GridTHWExplicit: true},
		},
	}
	got := GridTHWPerRaster(msg)
	if len(got) != 6 {
		t.Fatalf("len=%d want 6", len(got))
	}
	if got[0] != nil || got[1] != nil {
		t.Fatalf("still grids should be nil: %v %v", got[0], got[1])
	}
	want := []int{1, 24, 32}
	for i := 2; i <= 4; i++ {
		if len(got[i]) != 3 || got[i][0] != want[0] || got[i][1] != want[1] || got[i][2] != want[2] {
			t.Fatalf("frame %d grid=%v want %v", i, got[i], want)
		}
	}
	if len(got[5]) != 3 || got[5][1] != 16 || got[5][2] != 16 {
		t.Fatalf("clip2 frame grid=%v", got[5])
	}
}

func TestGridTHWPerRaster_forwardsServerEstimate(t *testing.T) {
	msg := api.Message{
		Images: make([]api.ImageData, 2),
		VideoSpans: []api.VideoSpan{
			{FrameCount: 2, GridTHW: []int{2, 8, 8}}, // ffmpeg estimate, not explicit
		},
	}
	got := GridTHWPerRaster(msg)
	want := []int{1, 8, 8}
	for i := 0; i < 2; i++ {
		if len(got[i]) != 3 || got[i][0] != want[0] || got[i][1] != want[1] || got[i][2] != want[2] {
			t.Fatalf("frame %d grid=%v want %v (server estimate must forward after M-RoPE hint honor)", i, got[i], want)
		}
	}
}

func TestGridTHWHasHints(t *testing.T) {
	if GridTHWHasHints([][]int{nil, nil}) {
		t.Fatal("expected false")
	}
	if !GridTHWHasHints([][]int{nil, {1, 2, 3}}) {
		t.Fatal("expected true")
	}
}
