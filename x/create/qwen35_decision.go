package create

import "strings"

type qwen35DecisionImportTransform struct {
	qwen35ImportTransform
}

func newQwen35DecisionImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	base, err := newQwen35ImportTransform(modelDir, cfg)
	if err != nil {
		return qwen35DecisionImportTransform{}, err
	}
	t, ok := base.(qwen35ImportTransform)
	if !ok {
		return qwen35DecisionImportTransform{qwen35ImportTransform: qwen35ImportTransform{}}, nil
	}
	return qwen35DecisionImportTransform{qwen35ImportTransform: t}, nil
}

func (qwen35DecisionImportTransform) quantizationType(name string, shape []int32, requested string) string {
	// Only backbone tensors are quantized. Decision heads retain source precision.
	if !strings.HasPrefix(name, "model.") && !strings.HasPrefix(name, "language_model.") {
		return ""
	}
	return qwen35ImportTransform{}.quantizationType(name, shape, requested)
}
