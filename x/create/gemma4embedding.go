package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/safetensors"
)

// Gemma4Embedding (EmbeddingGemma v2) import transform. The trunk is a
// 24-layer bidirectional encoder; the embedding_projection output head feeds
// mean-pooling + L2-norm downstream in the runner, so keep it unquantized.
type gemma4EmbeddingImportTransform struct {
	embeddingDim int
	contextLen   int
	hasVision    bool
	hasAudio     bool
}

func newGemma4EmbeddingImportTransform(modelDir string, _ sourceModelConfig) (tensorImportTransform, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("gemma4embedding: read config.json: %w", err)
	}
	var cfg struct {
		TextConfig struct {
			EmbeddingDim          int `json:"embedding_dim"`
			MaxPositionEmbeddings int `json:"max_position_embeddings"`
		} `json:"text_config"`
		VisionConfig json.RawMessage `json:"vision_config"`
		AudioConfig  json.RawMessage `json:"audio_config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("gemma4embedding: parse config.json: %w", err)
	}
	return gemma4EmbeddingImportTransform{
		embeddingDim: cfg.TextConfig.EmbeddingDim,
		contextLen:   cfg.TextConfig.MaxPositionEmbeddings,
		hasVision:    len(cfg.VisionConfig) > 0 && string(cfg.VisionConfig) != "null",
		hasAudio:     len(cfg.AudioConfig) > 0 && string(cfg.AudioConfig) != "null",
	}, nil
}

func (t gemma4EmbeddingImportTransform) skipTensor(string) bool { return false }

func (t gemma4EmbeddingImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	return []*safetensors.TensorData{td}, nil
}

func (t gemma4EmbeddingImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	switch {
	case name == "embedding_projection.weight" ||
		name == "model.embedding_projection.weight" ||
		name == "language_model.embedding_projection.weight":
		return ""
	default:
		return GetTensorQuantization(name, shape, quantize)
	}
}

// EmbeddingManifestCaps returns manifest capabilities and dimensions for EmbeddingGemma2Model.
func EmbeddingManifestCaps(modelDir string) ([]string, int, int, error) {
	tr, err := newGemma4EmbeddingImportTransform(modelDir, sourceModelConfig{})
	if err != nil {
		return nil, 0, 0, err
	}
	t := tr.(gemma4EmbeddingImportTransform)
	if t.embeddingDim <= 0 {
		return nil, 0, 0, fmt.Errorf("missing text_config.embedding_dim")
	}
	if t.contextLen <= 0 {
		return nil, 0, 0, fmt.Errorf("missing text_config.max_position_embeddings")
	}
	caps := []string{"embedding"}
	if t.hasVision {
		caps = append(caps, "vision")
	}
	if t.hasAudio {
		caps = append(caps, "audio")
	}
	ctxLen := min(t.contextLen, 8192)
	return caps, t.embeddingDim, ctxLen, nil
}
