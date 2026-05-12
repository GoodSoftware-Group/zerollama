package modality

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestBackendFor(t *testing.T) {
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{
			model.ModalityTranscribe: model.BackendWhisper,
			model.ModalitySpeech:     model.BackendPiper,
		},
	}
	if g, w := BackendFor(cfg, model.ModalityTranscribe), model.BackendWhisper; g != w {
		t.Fatalf("transcribe: got %q want %q", g, w)
	}
	if g, w := BackendFor(cfg, model.ModalitySpeech), model.BackendPiper; g != w {
		t.Fatalf("speech: got %q want %q", g, w)
	}
	if BackendFor(model.ConfigV2{}, model.ModalityImage) != "" {
		t.Fatal("expected empty default")
	}
}
