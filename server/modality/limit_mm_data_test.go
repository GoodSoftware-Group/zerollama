package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPreflightLimitMMDataPerRequest_latestUserOnly(t *testing.T) {
	limits := LimitMMDataPerRequest{Video: 1}
	req := &api.ChatRequest{Messages: []api.Message{
		{Role: "user", Content: "old", Videos: []api.VideoData{{1}, {2}}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "new", Videos: []api.VideoData{{3}}},
	}}
	if err := PreflightLimitMMDataPerRequest(limits, req); err != nil {
		t.Fatalf("latest user has 1 video: %v", err)
	}
	req.Messages[2].Videos = []api.VideoData{{3}, {4}}
	if err := PreflightLimitMMDataPerRequest(limits, req); err == nil {
		t.Fatal("expected limit error for 2 videos on latest user")
	}
}

func TestPreflightLimitMMDataPerRequest_preexpandedStillImages(t *testing.T) {
	limits := LimitMMDataPerRequest{Image: 1, Video: 1}
	req := &api.ChatRequest{Messages: []api.Message{{
		Role:        "user",
		Content:     "clip",
		Images:      []api.ImageData{{1}, {2}, {3}},
		VideoSpans:  []api.VideoSpan{{FrameCount: 2}},
	}}}
	if err := PreflightLimitMMDataPerRequest(limits, req); err != nil {
		t.Fatalf("1 still + 1 video span: %v", err)
	}
	req.Messages[0].Images = append(req.Messages[0].Images, api.ImageData{4})
	if err := PreflightLimitMMDataPerRequest(limits, req); err == nil {
		t.Fatal("expected image limit error")
	}
}
