package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPreprocessedLayoutCache_restoreOnTurn2(t *testing.T) {
	resetSessionVideoExpandCache()

	policy := VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	sessionKey := "preprocessed-agent-1"
	opts := map[string]any{"prompt_cache_key": sessionKey}
	padded := []int{301, 302, 303, 304}
	images := []api.ImageData{{0x01}, {0x02}, {0x03}}

	turn1 := &api.ChatRequest{
		Options: opts,
		Messages: []api.Message{{
			Role:           "user",
			Images:         images,
			VideoSpans:     []api.VideoSpan{{FrameCount: 3, GridTHW: []int{3, 24, 32}}},
			PaddedInputIDs: padded,
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, turn1); err != nil {
		t.Fatal(err)
	}

	turn2 := &api.ChatRequest{
		Options: opts,
		Messages: []api.Message{
			{Role: "user", Images: images, VideoSpans: []api.VideoSpan{{FrameCount: 3, GridTHW: []int{3, 24, 32}}}},
			{Role: "assistant", Content: "seen it"},
			{Role: "user", Images: images, VideoSpans: []api.VideoSpan{{FrameCount: 3, GridTHW: []int{3, 24, 32}}}},
		},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, turn2); err != nil {
		t.Fatal(err)
	}
	latest := turn2.Messages[2]
	if len(latest.PaddedInputIDs) != len(padded) {
		t.Fatalf("restored len=%d want %d", len(latest.PaddedInputIDs), len(padded))
	}
	for i := range padded {
		if latest.PaddedInputIDs[i] != padded[i] {
			t.Fatalf("restored[%d]=%d want %d", i, latest.PaddedInputIDs[i], padded[i])
		}
	}
}

func TestPreprocessedLayoutCache_missWithoutSessionKey(t *testing.T) {
	resetSessionVideoExpandCache()
	policy := VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	images := []api.ImageData{{0xaa}, {0xbb}}
	padded := []int{1, 2}

	turn1 := &api.ChatRequest{
		Messages: []api.Message{{
			Images:         images,
			VideoSpans:     []api.VideoSpan{{FrameCount: 2}},
			PaddedInputIDs: padded,
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, turn1); err != nil {
		t.Fatal(err)
	}

	turn2 := &api.ChatRequest{
		Messages: []api.Message{{
			Images:     images,
			VideoSpans: []api.VideoSpan{{FrameCount: 2}},
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, turn2); err != nil {
		t.Fatal(err)
	}
	if len(turn2.Messages[0].PaddedInputIDs) != 0 {
		t.Fatal("expected no restore without session key")
	}
}

func TestPreprocessedLayoutCache_multiSpanNoCache(t *testing.T) {
	resetSessionVideoExpandCache()
	policy := VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	sessionKey := "multi-span"
	opts := map[string]any{"prompt_cache_key": sessionKey}

	req := &api.ChatRequest{
		Options: opts,
		Messages: []api.Message{{
			Images:         []api.ImageData{{1}, {2}, {3}, {4}},
			VideoSpans:     []api.VideoSpan{{FrameCount: 2}, {FrameCount: 2}},
			PaddedInputIDs: []int{9, 8, 7},
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	layoutKey := sessionPreprocessedLayoutKey(req.Messages[0])
	if layoutKey != "" {
		t.Fatalf("multi-span should not produce layout key, got %q", layoutKey)
	}
}

func TestPreprocessedMessageFingerprint_stable(t *testing.T) {
	msg := api.Message{
		Images:     []api.ImageData{{0x11}, {0x22}, {0x33}},
		VideoSpans: []api.VideoSpan{{FrameCount: 2, GridTHW: []int{2, 16, 16}}},
	}
	fp1 := preprocessedMessageFingerprint(msg)
	fp2 := preprocessedMessageFingerprint(msg)
	if fp1 == "" || fp1 != fp2 {
		t.Fatalf("fp1=%q fp2=%q", fp1, fp2)
	}
	msg.Images[2][0] = 0xff
	if preprocessedMessageFingerprint(msg) == fp1 {
		t.Fatal("fingerprint should change when frame bytes change")
	}
}
