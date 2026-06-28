package llm

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func fakeDetokenize(_ context.Context, tokens []int) (string, error) {
	var b strings.Builder
	for i, t := range tokens {
		if i > 0 {
			b.WriteByte('|')
		}
		fmt.Fprintf(&b, "%d", t)
	}
	return b.String(), nil
}

func TestBuildLlamaServerPaddedMultimodalPrompt_visionBlock(t *testing.T) {
	tokens := []int{10, qwenVLVisionStart, 151655, 151655, qwenVLVisionEnd, 20}
	got, n, err := buildLlamaServerPaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, "<MEDIA>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("media count = %d, want 1", n)
	}
	want := "10<MEDIA>20"
	if got != want {
		t.Fatalf("prompt_string = %q, want %q", got, want)
	}
}

func TestBuildLlamaServerPaddedMultimodalPrompt_twoBlocks(t *testing.T) {
	tokens := []int{
		qwenVLVisionStart, 151655, qwenVLVisionEnd,
		99,
		qwenVLVisionStart, 151655, qwenVLVisionEnd,
	}
	got, n, err := buildLlamaServerPaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("media count = %d, want 2", n)
	}
	want := "<M>99<M>"
	if got != want {
		t.Fatalf("prompt_string = %q, want %q", got, want)
	}
}

func TestTruncateCompletionTokens(t *testing.T) {
	tokens := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	out, truncated, original := truncateCompletionTokens(tokens, 7, 2)
	if !truncated || original != 10 {
		t.Fatalf("truncated=%v original=%d", truncated, original)
	}
	want := []int{0, 1, 5, 6, 7, 8, 9}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v want %v", out, want)
	}
}

func TestTruncateCompletionTokens_preservesVisionBlocks(t *testing.T) {
	tokens := []int{
		0, 1,
		qwenVLVisionStart, 151655, 151655, qwenVLVisionEnd,
		2, 3, 4, 5, 6, 7, 8, 9,
	}
	out, truncated, original := truncateCompletionTokens(tokens, 7, 2)
	if !truncated || original != len(tokens) {
		t.Fatalf("truncated=%v original=%d", truncated, original)
	}
	for i, tok := range out {
		if tok == qwenVLVisionStart {
			j := i + 1
			for j < len(out) && out[j] != qwenVLVisionEnd {
				j++
			}
			if j >= len(out) || out[j] != qwenVLVisionEnd {
				t.Fatalf("orphan vision_start at %d in %v", i, out)
			}
		}
	}
}

func TestBuildLlamaServerPaddedMultimodalPrompt_unclosedVision(t *testing.T) {
	tokens := []int{10, qwenVLVisionStart, 151655}
	_, n, err := buildLlamaServerPaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, "<MEDIA>")
	if err == nil {
		t.Fatalf("expected error for unclosed vision block, got media=%d", n)
	}
}

func TestBuildLlamaServerGemma4PaddedMultimodalPrompt_imageSlots(t *testing.T) {
	const imageSlot = 888
	slots := Gemma4SoftTokens{Image: imageSlot}
	tokens := []int{10, imageSlot, 20, imageSlot, 30}
	got, n, err := buildLlamaServerGemma4PaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, slots, Gemma4PaddedMediaSchedule{}, "<MEDIA>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("media count = %d, want 2", n)
	}
	want := "10<MEDIA>20<MEDIA>30"
	if got != want {
		t.Fatalf("prompt_string = %q, want %q", got, want)
	}
}

func TestBuildLlamaServerGemma4PaddedMultimodalPrompt_videoSlots(t *testing.T) {
	const (
		imageSlot = 100
		videoSlot = 200
	)
	slots := Gemma4SoftTokens{Image: imageSlot, Video: videoSlot}
	tokens := []int{1, imageSlot, 2, videoSlot, 3}
	schedule := Gemma4PaddedMediaSchedule{VideoFrameCounts: []int{3}}
	got, n, err := buildLlamaServerGemma4PaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, slots, schedule, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("media count = %d, want 4 (1 image + 3 video frames)", n)
	}
	want := "1<M>2<M><M><M>3"
	if got != want {
		t.Fatalf("prompt_string = %q, want %q", got, want)
	}
}

func TestGemma4PromptHasSoftSlots(t *testing.T) {
	slots := Gemma4SoftTokens{Image: 42, Video: 43, Audio: 44}
	if !gemma4PromptHasSoftSlots([]int{1, 43, 3}, slots) {
		t.Fatal("expected true for video slot")
	}
	if gemma4PromptHasSoftSlots([]int{1, 2, 3}, slots) {
		t.Fatal("expected false")
	}
}

func TestGemma4PromptHasImageSlots(t *testing.T) {
	if !gemma4PromptHasImageSlots([]int{1, 42, 3}, 42) {
		t.Fatal("expected true")
	}
	if gemma4PromptHasImageSlots([]int{1, 2, 3}, 42) {
		t.Fatal("expected false")
	}
}

func TestCompletionMediaFromRequest_images(t *testing.T) {
	media := completionMediaFromRequest(CompletionRequest{
		Images: []ImageData{{ID: 0, Data: []byte("png")}, {ID: 1, Data: []byte("jpg")}},
	})
	if len(media) != 2 || media[1].ID != 1 {
		t.Fatalf("got %+v", media)
	}
}

func TestBuildLlamaServerLfm2PaddedMultimodalPrompt_block(t *testing.T) {
	const start, end = 500, 501
	slots := Lfm2VisionTokens{Start: start, End: end, UseBlock: true}
	tokens := []int{10, start, 502, 503, end, 20}
	got, n, err := buildLlamaServerLfm2PaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, slots, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || got != "10<M>20" {
		t.Fatalf("n=%d got=%q", n, got)
	}
}

func TestBuildLlamaServerLfm2PaddedMultimodalPrompt_flatImageRun(t *testing.T) {
	const imageTok = 396
	slots := Lfm2VisionTokens{Image: imageTok, UseBlock: false}
	tokens := []int{1, imageTok, imageTok, imageTok, 2}
	got, n, err := buildLlamaServerLfm2PaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, slots, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || got != "1<M>2" {
		t.Fatalf("n=%d got=%q", n, got)
	}
}

func TestBuildLlamaServerMistral3PaddedMultimodalPrompt(t *testing.T) {
	slots := Mistral3VisionTokens{Img: 10, Break: 12, End: 13}
	tokens := []int{1, 10, 11, 12, 11, 13, 2}
	got, n, err := buildLlamaServerMistral3PaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, slots, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || got != "1<M>2" {
		t.Fatalf("n=%d got=%q", n, got)
	}
}

func TestBuildLlamaServerDeepseekOcrPaddedMultimodalPrompt(t *testing.T) {
	const imageTok = 128815
	tokens := []int{5, imageTok, imageTok, 6, imageTok, 7}
	got, n, err := buildLlamaServerDeepseekOcrPaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, imageTok, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 || got != "5<M>6<M>7" {
		t.Fatalf("n=%d got=%q", n, got)
	}
}

func TestBuildLlamaServerSlotPaddedMultimodalPrompt(t *testing.T) {
	const slot = 128256
	tokens := []int{1, slot, 2, slot, 3}
	got, n, err := buildLlamaServerSlotPaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, slot, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 || got != "1<M>2<M>3" {
		t.Fatalf("n=%d got=%q", n, got)
	}
}

func TestBuildLlamaServerLlama4PaddedMultimodalPrompt(t *testing.T) {
	tokens := []int{
		1,
		llama4ImageBoundary, llama4ImageToken, llama4PatchToken, llama4ImageBoundary,
		2,
	}
	got, n, err := buildLlamaServerLlama4PaddedMultimodalPrompt(context.Background(), fakeDetokenize, tokens, "<M>")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || got != "1<M>2" {
		t.Fatalf("n=%d got=%q", n, got)
	}
}
