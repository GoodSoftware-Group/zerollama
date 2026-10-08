// Package clef implements Cloudflare's Qwen3.5 joint-schema decision model.
package clef

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	mlxmodel "github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/models/qwen3_5"
)

type config struct {
	HiddenSize    int `json:"hidden_size"`
	Width         int `json:"width"`
	RoutingLayers int `json:"routing_layers"`
	Layers        int `json:"layers"`
	Heads         int `json:"heads"`
	Feedforward   int `json:"feedforward"`
}

type Model struct {
	*qwen3_5.Model
	Head            head
	OutputEmbedding nn.EmbeddingLayer
	config          config
}

func init() { base.Register("ClefForDecision", newModel) }

func newModel(root *mlxmodel.Root) (base.Model, error) {
	var backboneConfig struct {
		ModelType string `json:"model_type"`
	}
	if err := root.Manifest.ReadConfigJSON("config.json", &backboneConfig); err != nil {
		return nil, err
	}
	if backboneConfig.ModelType != "qwen3_5" {
		return nil, fmt.Errorf("unsupported Clef backbone %q", backboneConfig.ModelType)
	}
	var cfg config
	if err := root.Manifest.ReadConfigJSON("joint_head_config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.HiddenSize <= 0 || cfg.Width <= 0 || cfg.Heads <= 0 || cfg.Width%cfg.Heads != 0 || cfg.Feedforward <= 0 || cfg.Layers < 1 || cfg.Layers > 16 || cfg.RoutingLayers < 1 || cfg.RoutingLayers > 16 {
		return nil, fmt.Errorf("invalid Clef joint head configuration")
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), config: cfg}
	if cfg.HiddenSize != int(m.Config.HiddenSize) {
		return nil, fmt.Errorf("Clef head and backbone hidden sizes differ")
	}
	return m, nil
}

// PrepareMedia expands text segments for Clef scoring. Image segments require
// the Qwen3.5 vision tower (upstream vision.go); text-only Clef works without it.
func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	for _, s := range segments {
		if len(s.Data) > 0 {
			return nil, fmt.Errorf("Clef image scoring requires Qwen3.5 MLX vision support (Mac lab)")
		}
	}
	var tokens []int32
	for _, s := range segments {
		tokens = append(tokens, s.Tokens...)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty Clef prompt")
	}
	return &base.PreparedRequest{Tokens: tokens}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	c := m.Config
	m.OutputEmbedding = mlxmodel.MakeEmbeddingLayer(tensors, "lm_head", c.QuantGroupSize, c.QuantBits, c.QuantMode, c.TensorQuant)
	if m.OutputEmbedding == nil {
		return fmt.Errorf("Clef requires its untied output embedding")
	}
	return m.Head.load(tensors, m.config)
}
