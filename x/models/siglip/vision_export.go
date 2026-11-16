package siglip

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
)

// Tower is a SigLIP vision encoder that returns patch tokens. LFM2-VL consumes
// those tokens through its own projector, so the MAP head is not applied.
type Tower struct {
	inner *visionTower
}

// LoadTower reads a vision tower. extraPrefixes are tried before the standalone
// SigLIP names and should include the trailing dot, for example
// "model.vision_tower.vision_model.".
func LoadTower(linears model.LinearFactory, tensors map[string]*mlx.Array, cfg *VisionConfig, extraPrefixes ...string) (*Tower, error) {
	keys := make([]string, 0, len(extraPrefixes))
	for _, prefix := range extraPrefixes {
		keys = append(keys, prefix+"embeddings.patch_embedding.weight")
	}
	tower, err := loadVisionKeys(linears, tensors, cfg, keys)
	if err != nil {
		return nil, err
	}
	return &Tower{inner: tower}, nil
}

// NaFlex reports whether the loaded patch embedding is the linear NaFlex form.
func (t *Tower) NaFlex() bool {
	return t != nil && t.inner != nil && t.inner.PatchLinear != nil
}

// PatchTokens runs the encoder through post-layernorm. patches is [1, N, C]
// or [N, C] and N must be gridH*gridW. The result is [1, N, hidden].
func (t *Tower) PatchTokens(patches *mlx.Array, gridH, gridW int) (*mlx.Array, error) {
	if t == nil || t.inner == nil || t.inner.PatchLinear == nil {
		return nil, fmt.Errorf("siglip tower has no linear patch embedding")
	}
	if gridH < 1 || gridW < 1 {
		return nil, fmt.Errorf("invalid patch grid %dx%d", gridH, gridW)
	}
	if patches.NumDims() == 2 {
		patches = mlx.Reshape(patches, 1, int32(patches.Dim(0)), int32(patches.Dim(1)))
	}
	h := t.inner.PatchLinear.Forward(patches.AsType(t.inner.posDtype()))
	return t.inner.run(h, gridH, gridW), nil
}
