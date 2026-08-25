package modality

import (
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestStripCoveredImageData(t *testing.T) {
	images := []llm.ImageData{
		{ID: 0, Data: []byte("a")},
		{ID: 1, Data: []byte("b")},
		{ID: 2, Data: []byte("c")},
	}
	spans := []MediaSpan{
		{Start: 0, End: 100},
		{Start: 150, End: 250},
		{Start: 300, End: 400},
	}
	out := StripCoveredImageData(images, spans, 250)
	if out[0].Data != nil || out[1].Data != nil {
		t.Fatal("expected first two payloads stripped")
	}
	if out[2].Data == nil {
		t.Fatal("tail item should keep payload")
	}
	if images[0].Data == nil {
		t.Fatal("original slice must not be mutated")
	}
}

func TestStripCoveredImageData_zeroComputed(t *testing.T) {
	images := []llm.ImageData{{Data: []byte("x")}}
	spans := []MediaSpan{{End: 10}}
	out := StripCoveredImageData(images, spans, 0)
	if out[0].Data == nil {
		t.Fatal("nothing stripped when numComputed=0")
	}
}
