package server

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

func TestGpusForSeeLanePinsVision(t *testing.T) {
	t.Setenv("ZEROLLAMA_SEE_GPU", "0")
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "CUDA0", Library: "CUDA"}},
		{DeviceID: ml.DeviceID{ID: "CUDA1", Library: "CUDA"}},
	}
	vision := &Model{}
	vision.Config.Capabilities = []string{string(model.CapabilityVision), string(model.CapabilityCompletion)}
	got := gpusForSeeLane(vision, gpus)
	if len(got) != 1 || got[0].ID != "CUDA0" {
		t.Fatalf("vision gpus %#v", got)
	}
	text := &Model{}
	text.Config.Capabilities = []string{string(model.CapabilityCompletion)}
	if got := gpusForSeeLane(text, gpus); len(got) != 2 {
		t.Fatalf("text should keep both, got %d", len(got))
	}
}

func TestGpusForSeeLaneOffByDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_SEE_GPU", "")
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "CUDA0", Library: "CUDA"}},
		{DeviceID: ml.DeviceID{ID: "CUDA1", Library: "CUDA"}},
	}
	vision := &Model{}
	vision.Config.Capabilities = []string{string(model.CapabilityVision)}
	if got := gpusForSeeLane(vision, gpus); len(got) != 2 {
		t.Fatalf("lane off, got %d", len(got))
	}
}
