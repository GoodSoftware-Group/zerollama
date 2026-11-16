package create

import (
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type lfm2VLImport struct{}

func newLfm2VLImportTransform(string, sourceModelConfig) (tensorImportTransform, error) {
	return lfm2VLImport{}, nil
}

func (lfm2VLImport) skipTensor(string) bool { return false }

func (lfm2VLImport) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	return []*safetensors.TensorData{td}, nil
}

func (lfm2VLImport) quantizationType(name string, shape []int32, quantize string) string {
	// The SigLIP2 tower and the pixel-shuffle projector stay in source precision.
	if strings.Contains(name, "vision_tower") || strings.Contains(name, "multi_modal_projector") {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}
