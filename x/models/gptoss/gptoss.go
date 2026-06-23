// Package gptoss provides the GPT-OSS MoE text model implementation for MLX.
package gptoss

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("GptOssForCausalLM", NewModel)
}

type ropeScalingConfig struct {
	Factor float32 `json:"factor"`
}

// Config holds GPT-OSS model configuration (HuggingFace config.json).
type Config struct {
	HiddenSize            int32    `json:"hidden_size"`
	NumHiddenLayers       int32    `json:"num_hidden_layers"`
	IntermediateSize      int32    `json:"intermediate_size"`
	NumAttentionHeads     int32    `json:"num_attention_heads"`
	NumKeyValueHeads      int32    `json:"num_key_value_heads"`
	HeadDim               int32    `json:"head_dim"`
	VocabSize             int32    `json:"vocab_size"`
	MaxPositionEmbeddings int32    `json:"max_position_embeddings"`
	RMSNormEps            float32  `json:"rms_norm_eps"`
	RopeTheta             float32  `json:"rope_theta"`
	AttentionBias         bool     `json:"attention_bias"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings"`
	SlidingWindow         int32    `json:"sliding_window"`
	LayerTypes            []string `json:"layer_types"`
	NumLocalExperts       int32    `json:"num_local_experts"`
	NumExpertsPerTok      int32    `json:"num_experts_per_tok"`
	ExpertsPerToken       int32    `json:"experts_per_token"`
	InitialContextLength  int32    `json:"initial_context_length"`
	RopeScalingFactor     float32  `json:"rope_scaling_factor"`
	RopeScaling           ropeScalingConfig `json:"rope_scaling"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	Scale         float32 `json:"-"`
	RopeInvScale  float32 `json:"-"`
}

func parseConfig(configData []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}

	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers <= 0 {
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads <= 0 {
		return Config{}, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim <= 0 {
		if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return Config{}, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 150000
	}
	if cfg.NumExpertsPerTok <= 0 {
		cfg.NumExpertsPerTok = cfg.ExpertsPerToken
	}
	if cfg.NumExpertsPerTok <= 0 {
		cfg.NumExpertsPerTok = 4
	}
	if cfg.NumLocalExperts <= 0 {
		return Config{}, fmt.Errorf("invalid num_local_experts: %d", cfg.NumLocalExperts)
	}
	if cfg.IntermediateSize <= 0 {
		cfg.IntermediateSize = cfg.HiddenSize
	}
	if cfg.InitialContextLength <= 0 {
		cfg.InitialContextLength = 4096
	}
	if cfg.RopeScalingFactor <= 0 && cfg.RopeScaling.Factor > 0 {
		cfg.RopeScalingFactor = cfg.RopeScaling.Factor
	}
	if cfg.RopeScalingFactor <= 0 {
		cfg.RopeScalingFactor = 1
	}
	cfg.RopeInvScale = 1.0 / cfg.RopeScalingFactor

	if len(cfg.LayerTypes) == 0 {
		cfg.LayerTypes = make([]string, cfg.NumHiddenLayers)
		for i := range cfg.NumHiddenLayers {
			if i%2 == 0 {
				cfg.LayerTypes[i] = "sliding_attention"
			} else {
				cfg.LayerTypes[i] = "full_attention"
			}
		}
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("layer_types has %d entries, want %d", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}

	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

func (cfg *Config) layerIsSliding(i int32) bool {
	return cfg.LayerTypes[i] == "sliding_attention"
}

// Model is the GPT-OSS MoE model.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

// Layer is a pre-norm decoder block with sparse MoE.
type Layer struct {
	AttentionNorm *nn.RMSNorm
	Attention     *Attention
	MLPNorm       *nn.RMSNorm
	MLP           *SparseMoE
	IsSliding     bool
}

// Attention implements GPT-OSS attention with sinks and NeoX RoPE.
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	Sinks *mlx.Array
}

// NewModel creates a GPT-OSS model from a manifest root.
func NewModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()
	model.ApplyQuantizationFromConfig(configData, &model.QuantConfigFields{
		QuantGroupSize: &cfg.QuantGroupSize,
		QuantBits:      &cfg.QuantBits,
		QuantMode:      &cfg.QuantMode,
		TensorQuant:    cfg.TensorQuant,
	})

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if genConfigData, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}
	if tokConfigData, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}

	for i := range cfg.NumHiddenLayers {
		m.Layers[i] = &Layer{IsSliding: cfg.layerIsSliding(i)}
	}

	return m, nil
}

// LoadWeights assigns tensors to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	cfg := m.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, "model.embed_tokens", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: model.embed_tokens.weight")
	}
	m.EmbedTokens = embedTokens

	normWeight := tensors["model.norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: model.norm.weight")
	}
	m.Norm = nn.NewRMSNorm(normWeight, cfg.RMSNormEps)

	if cfg.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		return fmt.Errorf("missing lm_head.weight")
	}

	useQuantizedExperts := supportsGatherQMM(cfg.QuantMode, cfg.QuantBits)
	if !useQuantizedExperts && cfg.TensorQuant != nil {
		for _, tq := range cfg.TensorQuant {
			if tq == nil {
				continue
			}
			_, bits, mode := model.QuantizationParams(tq.QuantType)
			if supportsGatherQMM(mode, bits) {
				useQuantizedExperts = true
				break
			}
		}
	}

	for i := range cfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("model.layers.%d", i)
		layer := &Layer{IsSliding: cfg.layerIsSliding(i)}

		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.AttentionNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.MLPNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}

		attn := &Attention{
			QProj: linears.Make(layerPrefix + ".self_attn.q_proj"),
			KProj: linears.Make(layerPrefix + ".self_attn.k_proj"),
			VProj: linears.Make(layerPrefix + ".self_attn.v_proj"),
			OProj: linears.Make(layerPrefix + ".self_attn.o_proj"),
		}
		if sinks := tensors[layerPrefix+".self_attn.sinks"]; sinks != nil {
			attn.Sinks = sinks
		} else if sinks := tensors[layerPrefix+".self_attn.sinks.weight"]; sinks != nil {
			attn.Sinks = sinks
		}
		layer.Attention = attn

		moe := &SparseMoE{Router: linears.Make(layerPrefix + ".mlp.router")}
		switchMLP, err := loadLayerExperts(tensors, cfg, useQuantizedExperts, layerPrefix)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		moe.SwitchMLP = switchMLP
		layer.MLP = moe

		if layer.AttentionNorm == nil || layer.MLPNorm == nil {
			return fmt.Errorf("layer %d: missing layernorm weights", i)
		}
		if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		if attn.Sinks == nil {
			return fmt.Errorf("layer %d: missing attention sinks", i)
		}
		if moe.Router == nil {
			return fmt.Errorf("layer %d: missing mlp router", i)
		}

		m.Layers[i] = layer
	}

	return nil
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Transpose(mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)
	v = mlx.Transpose(mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)

	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, cfg.RopeInvScale, positions)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, cfg.RopeInvScale, positions)

	var kv nn.SDPAOption
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}

	opts := []nn.SDPAOption{kv, nn.WithMask(nn.CausalMask())}
	// MLX Metal SDPA+sinks panics on GPT-OSS GQA (64Q/8KV). Skip sinks until the
	// reference path is wired; inference works without them.
	_ = a.Sinks
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, opts...)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	h := mlx.Add(x, l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), b, c, positions, B, L, cfg))
	return mlx.Add(h, l.MLP.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps), cfg))
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := m.EmbedTokens.Forward(b.InputIDs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}

	return m.Norm.Forward(h, m.RMSNormEps)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

func (m *Model) NumLayers() int {
	return len(m.Layers)
}

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if m.SlidingWindow > 0 && layer.IsSliding {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}
