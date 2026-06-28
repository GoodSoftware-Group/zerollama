package llamarunner

import (
	"reflect"
	"testing"
)

func TestInputsFromImageSlotTokens_gemma3(t *testing.T) {
	const slot = 255999
	tokens := []int{1, slot, 2, slot, 3}
	chunks := [][]visionChunk{{{embed: []float32{0.1}}}, {{embed: []float32{0.2}}}}
	got := inputsFromImageSlotTokens(tokens, chunks, slot)
	want := []input{{token: 1}, {embed: []float32{0.1}}, {token: 2}, {embed: []float32{0.2}}, {token: 3}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v want %+v", got, want)
	}
}

func TestInputsFromLlama4ImageBlocks(t *testing.T) {
	tokens := []int{
		1, llama4ImageBoundaryToken, llama4PatchToken, llama4PatchToken, llama4ImageBoundaryToken, 2,
	}
	chunks := [][]visionChunk{{{embed: []float32{0.5}}}}
	got := inputsFromLlama4ImageBlocks(tokens, chunks)
	want := []input{{token: 1}, {embed: []float32{0.5}}, {token: 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v want %+v", got, want)
	}
}

func TestInputsFromMistral3PaddedTokens(t *testing.T) {
	slots := mistral3VisionTokens{Img: 10, Break: 12, End: 13}
	tokens := []int{10, 12, 12, 13}
	chunks := [][]visionChunk{{{embed: []float32{0.1}}}}
	got := inputsFromMistral3PaddedTokens(tokens, chunks, slots)
	if len(got) != 1 || got[0].embed == nil {
		t.Fatalf("got %+v", got)
	}
}

func TestInputsFromDeepseekOcrPaddedTokens(t *testing.T) {
	const imageToken = 999
	tokens := []int{1, imageToken, imageToken, imageToken, 2}
	chunks := [][]visionChunk{{{embed: []float32{0.1}}}}
	got := inputsFromDeepseekOcrPaddedTokens(tokens, chunks, imageToken)
	want := []input{{token: 1}, {embed: []float32{0.1}}, {token: 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v want %+v", got, want)
	}
}

func TestSupportsPaddedLayoutConsume_allFamilies(t *testing.T) {
	modes := []string{
		PaddedLayoutConsumeQwen3VLHFRunner,
		PaddedLayoutConsumeGemma4ImgRunner,
		PaddedLayoutConsumeGemma3ImgRunner,
		PaddedLayoutConsumeMllamaImgRunner,
		PaddedLayoutConsumeLlama4ImgRunner,
		PaddedLayoutConsumeLfm2ImgRunner,
		PaddedLayoutConsumeGlmocrImgRunner,
		PaddedLayoutConsumeMistral3ImgRunner,
		PaddedLayoutConsumeDeepseekOcrImgRunner,
	}
	for _, m := range modes {
		if !supportsPaddedLayoutConsume(m) {
			t.Fatalf("expected support for %q", m)
		}
	}
	if supportsPaddedLayoutConsume("deferred") {
		t.Fatal("deferred should not be supported")
	}
}
