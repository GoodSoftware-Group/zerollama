// Package lfm2 provides the Liquid LFM2 text model for MLX.
// Port of mlx-lm mlx_lm/models/lfm2.py (hybrid ShortConv + full attention).
package lfm2

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("Lfm2ForCausalLM", newModel)
}

// Config holds LFM2 model configuration from config.json.
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	VocabSize             int32   `json:"vocab_size"`
	NormEps               float32 `json:"norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	ConvBias              bool    `json:"conv_bias"`
	ConvLCache            int32   `json:"conv_L_cache"`
	BlockDim              int32   `json:"block_dim"`
	BlockFFDim            int32   `json:"block_ff_dim"`
	BlockMultipleOf       int32   `json:"block_multiple_of"`
	BlockFFNDimMultiplier float32 `json:"block_ffn_dim_multiplier"`
	BlockAutoAdjustFFDim  bool    `json:"block_auto_adjust_ff_dim"`
	FullAttnIdxs          []int   `json:"full_attn_idxs"`
	LayerTypes            []string `json:"layer_types"`

	HeadDim        int32                             `json:"-"`
	IntermediateSize int32                           `json:"-"`
	Scale          float32                           `json:"-"`
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`
}

// Model is the LFM2 causal LM.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config

	weightPrefix string
}

// Layer is one LFM2 decoder block (attention or ShortConv + shared MLP).
type Layer struct {
	IsAttention   bool
	SelfAttn      *Attention
	Conv          *ShortConv
	FeedForward   *MLP
	OperatorNorm  *nn.RMSNorm
	FFNNorm       *nn.RMSNorm
}

// Attention is LFM2 full attention (GQA + Q/K RMSNorm + RoPE).
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm
}

// ShortConv is the gated depthwise causal convolution branch.
type ShortConv struct {
	Conv    *nn.Conv1d
	InProj  nn.LinearLayer
	OutProj nn.LinearLayer
}

// MLP is SwiGLU with LFM2 naming (w1/w3/w2).
type MLP struct {
	W1 nn.LinearLayer
	W3 nn.LinearLayer
	W2 nn.LinearLayer
}

func resolveWeightPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{"", "language_model."} {
		if tensors[prefix+"model.embed_tokens.weight"] != nil {
			return prefix
		}
	}
	return ""
}

func effectiveFFDim(cfg *Config) int32 {
	ff := cfg.BlockFFDim
	if cfg.BlockAutoAdjustFFDim {
		ff = int32(2 * ff / 3)
		if cfg.BlockFFNDimMultiplier != 0 {
			ff = int32(float32(ff) * cfg.BlockFFNDimMultiplier)
		}
		mult := cfg.BlockMultipleOf
		if mult <= 0 {
			mult = 1
		}
		ff = mult * ((ff + mult - 1) / mult)
	}
	return ff
}

func resolveFullAttnIdxs(cfg *Config) []int {
	if len(cfg.FullAttnIdxs) > 0 {
		return cfg.FullAttnIdxs
	}
	var idxs []int
	for i, t := range cfg.LayerTypes {
		if t == "full_attention" {
			idxs = append(idxs, i)
		}
	}
	return idxs
}

func sanitizeConvWeight(w *mlx.Array) *mlx.Array {
	if w == nil || w.NumDims() != 3 {
		return w
	}
	// Torch depthwise is often [C, 1, K]; MLX wants [C, K, 1].
	if w.Dim(2) > w.Dim(1) {
		return mlx.Transpose(w, 0, 2, 1)
	}
	return w
}

func newModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if cfg.HiddenSize <= 0 {
		return nil, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.NormEps == 0 {
		cfg.NormEps = 1e-5
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1_000_000
	}
	if cfg.ConvLCache <= 0 {
		cfg.ConvLCache = 3
	}
	if cfg.BlockDim <= 0 {
		cfg.BlockDim = cfg.HiddenSize
	}
	if cfg.BlockFFDim <= 0 {
		return nil, fmt.Errorf("invalid block_ff_dim: %d", cfg.BlockFFDim)
	}
	cfg.FullAttnIdxs = resolveFullAttnIdxs(&cfg)
	if len(cfg.FullAttnIdxs) == 0 {
		return nil, fmt.Errorf("no full_attn_idxs / full_attention layer_types in config")
	}
	cfg.IntermediateSize = effectiveFFDim(&cfg)
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		// config.json quantization{bits,group_size} from mlx-lm convert —
		// blob metadata is often empty for these imports.
		var quantCfg struct {
			Quantization struct {
				GroupSize int `json:"group_size"`
				Bits      int `json:"bits"`
			} `json:"quantization"`
		}
		_ = json.Unmarshal(configData, &quantCfg)
		if quantCfg.Quantization.Bits > 0 {
			cfg.QuantBits = quantCfg.Quantization.Bits
			cfg.QuantGroupSize = quantCfg.Quantization.GroupSize
			if cfg.QuantGroupSize <= 0 {
				cfg.QuantGroupSize = 64
			}
			cfg.QuantMode = "affine"
		} else {
			cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
		}
	}
	cfg.TensorQuant = root.AllTensorQuant()

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

	return &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}, nil
}

// LoadWeights assigns tensors to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	m.weightPrefix = resolveWeightPrefix(tensors)
	prefix := m.weightPrefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, prefix+"model.embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: %smodel.embed_tokens.weight", prefix)
	}
	m.EmbedTokens = embedTokens

	normWeight := tensors[prefix+"model.embedding_norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %smodel.embedding_norm.weight", prefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.NormEps)

	// LFM2 always ties the LM head to embeddings (mlx-lm as_linear).
	if lmHead := linears.Make(prefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	for i := range int(m.NumHiddenLayers) {
		layerPrefix := fmt.Sprintf("%smodel.layers.%d", prefix, i)
		isAttn := slices.Contains(m.FullAttnIdxs, i)
		layer := &Layer{IsAttention: isAttn}

		if w := tensors[layerPrefix+".operator_norm.weight"]; w != nil {
			layer.OperatorNorm = nn.NewRMSNorm(w, m.NormEps)
		} else {
			return fmt.Errorf("layer %d: missing operator_norm", i)
		}
		if w := tensors[layerPrefix+".ffn_norm.weight"]; w != nil {
			layer.FFNNorm = nn.NewRMSNorm(w, m.NormEps)
		} else {
			return fmt.Errorf("layer %d: missing ffn_norm", i)
		}

		layer.FeedForward = &MLP{
			W1: linears.Make(layerPrefix + ".feed_forward.w1"),
			W3: linears.Make(layerPrefix + ".feed_forward.w3"),
			W2: linears.Make(layerPrefix + ".feed_forward.w2"),
		}
		if layer.FeedForward.W1 == nil || layer.FeedForward.W3 == nil || layer.FeedForward.W2 == nil {
			return fmt.Errorf("layer %d: missing feed_forward projections", i)
		}

		if isAttn {
			attn := &Attention{
				QProj: linears.Make(layerPrefix + ".self_attn.q_proj"),
				KProj: linears.Make(layerPrefix + ".self_attn.k_proj"),
				VProj: linears.Make(layerPrefix + ".self_attn.v_proj"),
				OProj: linears.Make(layerPrefix + ".self_attn.out_proj"),
			}
			if w := tensors[layerPrefix+".self_attn.q_layernorm.weight"]; w != nil {
				attn.QNorm = nn.NewRMSNorm(w, m.NormEps)
			}
			if w := tensors[layerPrefix+".self_attn.k_layernorm.weight"]; w != nil {
				attn.KNorm = nn.NewRMSNorm(w, m.NormEps)
			}
			if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil {
				return fmt.Errorf("layer %d: missing attention projections", i)
			}
			if attn.QNorm == nil || attn.KNorm == nil {
				return fmt.Errorf("layer %d: missing attention q/k layernorm", i)
			}
			layer.SelfAttn = attn
		} else {
			convW := sanitizeConvWeight(tensors[layerPrefix+".conv.conv.weight"])
			if convW == nil {
				return fmt.Errorf("layer %d: missing conv.conv.weight", i)
			}
			var convBias *mlx.Array
			if m.ConvBias {
				convBias = tensors[layerPrefix+".conv.conv.bias"]
			}
			groups := int32(convW.Dim(0))
			sc := &ShortConv{
				Conv:    nn.NewConv1d(convW, convBias, 1, 0, 1, groups),
				InProj:  linears.Make(layerPrefix + ".conv.in_proj"),
				OutProj: linears.Make(layerPrefix + ".conv.out_proj"),
			}
			if sc.InProj == nil || sc.OutProj == nil {
				return fmt.Errorf("layer %d: missing conv projections", i)
			}
			layer.Conv = sc
		}

		m.Layers[i] = layer
	}

	return nil
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
	return m.Norm.Forward(h, m.NormEps)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

func (m *Model) NumLayers() int { return len(m.Layers) }

func (m *Model) MaxContextLength() int { return int(m.MaxPositionEmbeddings) }

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	convTail := m.ConvLCache - 1
	for i, layer := range m.Layers {
		if layer.IsAttention {
			caches[i] = cache.NewKVCache()
		} else {
			// ShortConv only uses conv state; dummy delta dims satisfy RecurrentCache.
			caches[i] = cache.NewRecurrentCache(convTail, m.HiddenSize, 1, 1, 1)
		}
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	normed := l.OperatorNorm.Forward(x, cfg.NormEps)
	var r *mlx.Array
	if l.IsAttention {
		r = l.SelfAttn.Forward(normed, b, c, positions, B, L, cfg)
	} else {
		r = l.Conv.Forward(normed, b, c, B, L, cfg)
	}
	h := mlx.Add(x, r)
	return mlx.Add(h, l.FeedForward.Forward(l.FFNNorm.Forward(h, cfg.NormEps)))
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = a.QNorm.Forward(q, cfg.NormEps)
	k = a.KNorm.Forward(k, cfg.NormEps)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions)

	var kv nn.SDPAOption
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (s *ShortConv) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	hidden := cfg.HiddenSize
	bcx := s.InProj.Forward(x)
	gateB := mlx.SliceStartStop(bcx, []int32{0, 0, 0}, []int32{B, L, hidden})
	gateC := mlx.SliceStartStop(bcx, []int32{0, 0, hidden}, []int32{B, L, 2 * hidden})
	xPart := mlx.SliceStartStop(bcx, []int32{0, 0, 2 * hidden}, []int32{B, L, 3 * hidden})
	bx := mlx.Mul(gateB, xPart)

	convTail := int(cfg.ConvLCache - 1)
	var rc *cache.RecurrentCache
	var hist *nn.RecurrentHistory
	opts := make([]nn.RecurrentOption, 0, 2)
	if typed, ok := c.(*cache.RecurrentCache); ok {
		rc = typed
		hist = rc.Get(b, bx.DType())
		opts = append(opts, nn.WithRecurrentHistory(hist))
		if splits := rc.SnapshotSplits(int(L)); len(splits) > 0 {
			opts = append(opts, nn.WithSnapshotSplits(splits))
		}
	} else {
		opts = append(opts, nn.WithRecurrentState(
			mlx.Zeros(bx.DType(), int(B), convTail, int(hidden)),
			mlx.Zeros(mlx.DTypeFloat32, int(B), 1, 1, 1),
		))
	}

	convOut, convStates := nn.CausalConv1D(b, bx, s.Conv, convTail, opts...)
	y := mlx.Mul(gateC, convOut)
	out := s.OutProj.Forward(y)

	if rc != nil {
		deltas := make([]*mlx.Array, len(convStates))
		delta := hist.DeltaState()
		for i := range deltas {
			deltas[i] = delta
		}
		rc.Put(b, convStates, deltas)
	}
	return out
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.W2.Forward(mlx.SwiGLU(m.W1.Forward(x), m.W3.Forward(x)))
}
