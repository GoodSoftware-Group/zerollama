package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGemma4PaddedMediaScheduleFromMessage_stillsAndVideo(t *testing.T) {
	msg := api.Message{
		Images: make([]api.ImageData, 5),
		VideoSpans: []api.VideoSpan{
			{FrameCount: 3},
			{FrameCount: 1},
		},
		AudioClips: []api.AudioData{{1}, {2}},
	}
	got := Gemma4PaddedMediaScheduleFromMessage(msg)
	if got.StillImageCount != 1 {
		t.Fatalf("still=%d want 1", got.StillImageCount)
	}
	if len(got.VideoFrameCounts) != 2 || got.VideoFrameCounts[0] != 3 || got.VideoFrameCounts[1] != 1 {
		t.Fatalf("video=%v", got.VideoFrameCounts)
	}
	if got.AudioClipCount != 2 {
		t.Fatalf("audio=%d want 2", got.AudioClipCount)
	}
}

func TestGemma4PaddedMediaScheduleForChat_latestUser(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Images: []api.ImageData{{}}, VideoSpans: []api.VideoSpan{{FrameCount: 1}}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Images: make([]api.ImageData, 4), VideoSpans: []api.VideoSpan{{FrameCount: 4}}},
	}
	got := Gemma4PaddedMediaScheduleForChat(msgs)
	if got.StillImageCount != 0 || len(got.VideoFrameCounts) != 1 || got.VideoFrameCounts[0] != 4 {
		t.Fatalf("got %+v", got)
	}
}
