package deepseekv4

import (
	"encoding/json"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

// Config is HuggingFace / mlx-lm config.json for DeepseekV4ForCausalLM (Flash).
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	MoEIntermediateSize   int32   `json:"moe_intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	QLoraRank             int32   `json:"q_lora_rank"`
	QKRopeHeadDim         int32   `json:"qk_rope_head_dim"`
	OGroups               int32   `json:"o_groups"`
	OLoraRank             int32   `json:"o_lora_rank"`
	NRoutedExperts        int32   `json:"n_routed_experts"`
	NSharedExperts        int32   `json:"n_shared_experts"`
	NumExpertsPerTok      int32   `json:"num_experts_per_tok"`
	NormTopKProb          bool    `json:"norm_topk_prob"`
	RoutedScalingFactor   float32 `json:"routed_scaling_factor"`
	NumHashLayers         int32   `json:"num_hash_layers"`
	HCEps                 float32 `json:"hc_eps"`
	HCMult                int32   `json:"hc_mult"`
	HCSinkhornIters       int32   `json:"hc_sinkhorn_iters"`
	CompressRopeTheta     float32 `json:"compress_rope_theta"`
	CompressRatios        []int32 `json:"compress_ratios"`
	ScoringFunc           string  `json:"scoring_func"`
	SwigluLimit           float32 `json:"swiglu_limit"`
	SlidingWindow         int32   `json:"sliding_window"`
	IndexHeadDim          int32   `json:"index_head_dim"`
	IndexNHeads           int32   `json:"index_n_heads"`
	IndexTopK             int32   `json:"index_topk"`

	RopeScaling *nn.RopeParameters `json:"rope_scaling"`

	Quantization json.RawMessage `json:"quantization"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"` // why: 2-bit vs 4-bit packed cols collide
	Scale          float32                           `json:"-"`
	QKNopeHeadDim  int32                             `json:"-"`
}

func (c *Config) finish() {
	if c.HCMult <= 0 {
		c.HCMult = 4
	}
	if c.HCSinkhornIters <= 0 {
		c.HCSinkhornIters = 20
	}
	if c.HCEps == 0 {
		c.HCEps = 1e-6
	}
	if c.NumKeyValueHeads <= 0 {
		c.NumKeyValueHeads = 1
	}
	if c.HeadDim <= 0 && c.NumAttentionHeads > 0 {
		c.HeadDim = 512
	}
	if c.QKRopeHeadDim <= 0 {
		c.QKRopeHeadDim = 64
	}
	c.QKNopeHeadDim = c.HeadDim - c.QKRopeHeadDim
	if c.OGroups <= 0 {
		c.OGroups = 8
	}
	if c.Scale == 0 && c.HeadDim > 0 {
		c.Scale = float32(1.0 / math.Sqrt(float64(c.HeadDim)))
	}
	if c.SlidingWindow <= 0 {
		c.SlidingWindow = 128
	}
	if c.IndexHeadDim <= 0 {
		c.IndexHeadDim = 128
	}
	if c.IndexNHeads <= 0 {
		c.IndexNHeads = 64
	}
	if c.IndexTopK <= 0 {
		c.IndexTopK = 512
	}
}

func (c *Config) applyQuantFromConfigJSON(configData []byte) {
	gs, bits, mode := c.QuantGroupSize, c.QuantBits, c.QuantMode
	fields := model.QuantConfigFields{
		QuantGroupSize: &gs,
		QuantBits:      &bits,
		QuantMode:      &mode,
		TensorQuant:    c.TensorQuant,
	}
	model.ApplyQuantizationFromConfig(configData, &fields)
	c.QuantGroupSize, c.QuantBits, c.QuantMode = gs, bits, mode
	c.TensorQuant = fields.TensorQuant
}

func (c *Config) quantFor(path string) (groupSize, bits int, mode string) {
	groupSize, bits, mode, _ = model.TensorQuantParams(
		c.QuantGroupSize, c.QuantBits, c.QuantMode, c.TensorQuant, path+".weight",
	)
	return
}

func supportsGatherQMM(mode string, bits int) bool {
	return mode == "affine" && (bits == 2 || bits == 4 || bits == 8)
}

func compressRatio(cfg *Config, i int) int32 {
	if i >= 0 && i < len(cfg.CompressRatios) {
		return cfg.CompressRatios[i]
	}
	return 0
}
