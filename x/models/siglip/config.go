package siglip

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// VisionConfig is the resolved vision tower. Zero ImageSize means NaFlex:
// the image is resized per request so the patch count stays within MaxPatches.
type VisionConfig struct {
	HiddenSize       int32
	IntermediateSize int32
	NumHiddenLayers  int32
	NumHeads         int32
	NumChannels      int32
	ImageSize        int32
	PatchSize        int32
	NumPatches       int32
	HiddenAct        string
	LayerNormEps     float32
	UseHead          bool
	NaFlex           bool
}

// TextConfig is the bidirectional text tower. SigLIP pools the final
// position, which training fills with padding, so MaxPositionEmbeddings is
// the sequence length the tower always runs.
type TextConfig struct {
	HiddenSize            int32
	IntermediateSize      int32
	NumHiddenLayers       int32
	NumHeads              int32
	MaxPositionEmbeddings int32
	VocabSize             int32
	ProjectionSize        int32
	PadTokenID            int32
	HiddenAct             string
	LayerNormEps          float32
}

// Config is the checkpoint after defaults are filled in. Published SigLIP
// configs omit most fields; the numbers match transformers' Siglip*Config.
type Config struct {
	Architecture string
	Family       string
	Vision       *VisionConfig
	Text         *TextConfig
}

type rawTower struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int32   `json:"hidden_size"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumChannels           int32   `json:"num_channels"`
	ImageSize             int32   `json:"image_size"`
	PatchSize             int32   `json:"patch_size"`
	NumPatches            int32   `json:"num_patches"`
	HiddenAct             string  `json:"hidden_act"`
	LayerNormEps          float32 `json:"layer_norm_eps"`
	VisionUseHead         *bool   `json:"vision_use_head"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	VocabSize             int32   `json:"vocab_size"`
	ProjectionSize        int32   `json:"projection_size"`
	PadTokenID            *int32  `json:"pad_token_id"`
}

type rawConfig struct {
	Architectures []string        `json:"architectures"`
	ModelType     string          `json:"model_type"`
	VisionConfig  json.RawMessage `json:"vision_config"`
	TextConfig    json.RawMessage `json:"text_config"`
	rawTower
}

func parseConfig(data []byte) (Config, error) {
	var raw rawConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return Config{}, fmt.Errorf("parse siglip config: %w", err)
	}
	arch := ""
	if len(raw.Architectures) > 0 {
		arch = raw.Architectures[0]
	}
	family := familyOf(arch, raw.ModelType)
	cfg := Config{Architecture: arch, Family: family}

	visionRaw, textRaw, err := nestedTowers(raw)
	if err != nil {
		return Config{}, err
	}
	if wantVision(arch, raw.ModelType, visionRaw) {
		vision, err := resolveVision(visionRaw, family)
		if err != nil {
			return Config{}, err
		}
		cfg.Vision = vision
	}
	if wantText(arch, raw.ModelType, textRaw) {
		cfg.Text = resolveText(textRaw)
	}
	if cfg.Vision == nil && cfg.Text == nil {
		return Config{}, fmt.Errorf("siglip config has neither a text nor a vision tower")
	}
	return cfg, nil
}

func familyOf(arch, modelType string) string {
	switch {
	case strings.Contains(strings.ToLower(arch), "siglip2"):
		return "siglip2"
	case modelType == "siglip2" || strings.HasPrefix(modelType, "siglip2_"):
		return "siglip2"
	default:
		return "siglip"
	}
}

func nestedTowers(raw rawConfig) (vision, text rawTower, err error) {
	if len(raw.VisionConfig) > 0 && string(raw.VisionConfig) != "null" {
		if err = json.Unmarshal(raw.VisionConfig, &vision); err != nil {
			return rawTower{}, rawTower{}, fmt.Errorf("parse vision_config: %w", err)
		}
	} else if strings.Contains(raw.ModelType, "vision") || strings.Contains(strings.ToLower(raw.ArchitecturesJoin()), "vision") {
		vision = raw.rawTower
	}
	if len(raw.TextConfig) > 0 && string(raw.TextConfig) != "null" {
		if err = json.Unmarshal(raw.TextConfig, &text); err != nil {
			return rawTower{}, rawTower{}, fmt.Errorf("parse text_config: %w", err)
		}
	} else if strings.Contains(raw.ModelType, "text") || strings.Contains(strings.ToLower(raw.ArchitecturesJoin()), "text") {
		text = raw.rawTower
	}
	return vision, text, nil
}

func (r rawConfig) ArchitecturesJoin() string {
	return strings.Join(r.Architectures, " ")
}

func wantVision(arch, modelType string, tower rawTower) bool {
	if strings.Contains(strings.ToLower(arch), "text") && !strings.Contains(strings.ToLower(arch), "vision") {
		return false
	}
	if strings.Contains(modelType, "text") && !strings.Contains(modelType, "vision") && tower.ModelType == "" && tower.HiddenSize == 0 && tower.PatchSize == 0 {
		return false
	}
	if strings.Contains(strings.ToLower(arch), "vision") || strings.Contains(modelType, "vision") {
		return true
	}
	return tower.ModelType != "" || tower.PatchSize != 0 || tower.HiddenSize != 0 || tower.NumPatches != 0
}

func wantText(arch, modelType string, tower rawTower) bool {
	if strings.Contains(strings.ToLower(arch), "vision") && !strings.Contains(strings.ToLower(arch), "text") {
		return false
	}
	if strings.Contains(modelType, "vision") && !strings.Contains(modelType, "text") && tower.ModelType == "" && tower.VocabSize == 0 {
		return false
	}
	if strings.Contains(strings.ToLower(arch), "text") || strings.Contains(modelType, "text") {
		return true
	}
	return tower.ModelType != "" || tower.VocabSize != 0 || tower.MaxPositionEmbeddings != 0
}

// ParseVisionConfig resolves a nested vision_config object, including SigLIP2
// towers that omit image_size.
func ParseVisionConfig(data []byte) (*VisionConfig, error) {
	var raw rawTower
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse vision config: %w", err)
	}
	family := "siglip"
	if strings.Contains(raw.ModelType, "siglip2") {
		family = "siglip2"
	}
	return resolveVision(raw, family)
}

func resolveVision(raw rawTower, family string) (*VisionConfig, error) {
	v := &VisionConfig{
		HiddenSize:       or32(raw.HiddenSize, 768),
		IntermediateSize: or32(raw.IntermediateSize, 3072),
		NumHiddenLayers:  or32(raw.NumHiddenLayers, 12),
		NumHeads:         or32(raw.NumAttentionHeads, 12),
		NumChannels:      or32(raw.NumChannels, 3),
		PatchSize:        or32(raw.PatchSize, 16),
		HiddenAct:        orStr(raw.HiddenAct, "gelu_pytorch_tanh"),
		LayerNormEps:     orF32(raw.LayerNormEps, 1e-6),
		UseHead:          raw.VisionUseHead == nil || *raw.VisionUseHead,
		NaFlex:           family == "siglip2" || strings.Contains(raw.ModelType, "siglip2"),
	}
	if !v.NaFlex {
		v.ImageSize = or32(raw.ImageSize, 224)
		v.NumPatches = (v.ImageSize / v.PatchSize) * (v.ImageSize / v.PatchSize)
	} else {
		v.NumPatches = or32(raw.NumPatches, 256)
		v.ImageSize = 0
	}
	if v.HiddenSize%v.NumHeads != 0 {
		return nil, fmt.Errorf("vision hidden_size %d is not divisible by %d heads", v.HiddenSize, v.NumHeads)
	}
	side := int32(math.Round(math.Sqrt(float64(v.NumPatches))))
	if side*side != v.NumPatches {
		return nil, fmt.Errorf("vision num_patches %d is not a square grid", v.NumPatches)
	}
	return v, nil
}

func resolveText(raw rawTower) *TextConfig {
	t := &TextConfig{
		HiddenSize:            or32(raw.HiddenSize, 768),
		IntermediateSize:      or32(raw.IntermediateSize, 3072),
		NumHiddenLayers:       or32(raw.NumHiddenLayers, 12),
		NumHeads:              or32(raw.NumAttentionHeads, 12),
		MaxPositionEmbeddings: or32(raw.MaxPositionEmbeddings, 64),
		VocabSize:             or32(raw.VocabSize, 32000),
		HiddenAct:             orStr(raw.HiddenAct, "gelu_pytorch_tanh"),
		LayerNormEps:          orF32(raw.LayerNormEps, 1e-6),
		PadTokenID:            1,
	}
	if raw.PadTokenID != nil {
		t.PadTokenID = *raw.PadTokenID
	}
	t.ProjectionSize = raw.ProjectionSize
	if t.ProjectionSize == 0 {
		t.ProjectionSize = t.HiddenSize
	}
	return t
}

func or32(v, fallback int32) int32 {
	if v == 0 {
		return fallback
	}
	return v
}

func orF32(v, fallback float32) float32 {
	if v == 0 {
		return fallback
	}
	return v
}

func orStr(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}
