package llamarunner

import (
	"reflect"
	"testing"
)

func TestInputsFromQwen3VLPromptTokens_visionBlock(t *testing.T) {
	tokens := []int{1, qwenVLVisionStart, qwenVLImagePad, qwenVLImagePad, qwenVLVisionEnd, 2}
	chunks := [][]visionChunk{{
		{tokens: []int{qwenVLVisionStart}},
		{embed: []float32{0.1, 0.2}},
		{embed: []float32{0.3, 0.4}},
		{tokens: []int{qwenVLVisionEnd}},
	}}
	got := inputsFromQwen3VLPromptTokens(tokens, chunks)
	want := []input{
		{token: 1},
		{token: qwenVLVisionStart},
		{embed: []float32{0.1, 0.2}},
		{embed: []float32{0.3, 0.4}},
		{token: qwenVLVisionEnd},
		{token: 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v want %+v", got, want)
	}
}

func TestInputsFromQwen3VLPromptTokens_padFallback(t *testing.T) {
	tokens := []int{qwenVLImagePad, qwenVLImagePad, qwenVLImagePad}
	chunks := [][]visionChunk{{{embed: []float32{0.1}}}}
	got := inputsFromQwen3VLPromptTokens(tokens, chunks)
	if len(got) != 3 || got[0].embed == nil || got[1].token != qwenVLImagePad {
		t.Fatalf("got %+v", got)
	}
}

func TestInputsFromQwen3VLPromptTokens_twoVisionBlocks(t *testing.T) {
	tokens := []int{
		qwenVLVisionStart, qwenVLImagePad, qwenVLVisionEnd,
		qwenVLVisionStart, qwenVLImagePad, qwenVLVisionEnd,
	}
	chunks := [][]visionChunk{
		{{embed: []float32{0.1}}},
		{{embed: []float32{0.2}}},
	}
	got := inputsFromQwen3VLPromptTokens(tokens, chunks)
	if len(got) != 2 || got[0].embed == nil || got[1].embed == nil {
		t.Fatalf("got %+v", got)
	}
}

func TestInputsFromGemma4PromptTokens_imageSlots(t *testing.T) {
	const imageSlot = 999
	slots := Gemma4SoftTokens{Image: imageSlot}
	tokens := []int{1, imageSlot, 2, imageSlot, 3}
	chunks := [][]visionChunk{{{embed: []float32{0.1}}}, {{embed: []float32{0.2}}}}
	got := inputsFromGemma4PromptTokens(tokens, chunks, slots, Gemma4PaddedMediaSchedule{})
	want := []input{
		{token: 1},
		{embed: []float32{0.1}},
		{token: 2},
		{embed: []float32{0.2}},
		{token: 3},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v want %+v", got, want)
	}
}

func TestInputsFromGemma4PromptTokens_videoClip(t *testing.T) {
	const (
		imageSlot = 100
		videoSlot = 200
	)
	slots := Gemma4SoftTokens{Image: imageSlot, Video: videoSlot}
	tokens := []int{1, videoSlot, 2}
	chunks := [][]visionChunk{
		{{embed: []float32{0.1}}},
		{{embed: []float32{0.2}}},
		{{embed: []float32{0.3}}},
	}
	schedule := Gemma4PaddedMediaSchedule{VideoFrameCounts: []int{3}}
	got := inputsFromGemma4PromptTokens(tokens, chunks, slots, schedule)
	if len(got) != 5 {
		t.Fatalf("len=%d want 5", len(got))
	}
	if got[0].token != 1 || got[1].embed == nil || got[2].embed == nil || got[3].embed == nil || got[4].token != 2 {
		t.Fatalf("got %+v", got)
	}
}
