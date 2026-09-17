// Package qwen3 provides a shared Qwen3 text encoder used by multiple image generation models.
package qwen3

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Config holds Qwen3 text encoder configuration
type Config struct {
	HiddenSize        int32   `json:"hidden_size"`
	NumHiddenLayers   int32   `json:"num_hidden_layers"`
	IntermediateSize  int32   `json:"intermediate_size"`
	NumAttentionHeads int32   `json:"num_attention_heads"`
	NumKeyValueHeads  int32   `json:"num_key_value_heads"`
	VocabSize         int32   `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	HeadDim           int32   `json:"head_dim"`
}

// Attention implements Qwen3 attention with QK norms
type Attention struct {
	QProj nn.LinearLayer `weight:"q_proj"`
	KProj nn.LinearLayer `weight:"k_proj"`
	VProj nn.LinearLayer `weight:"v_proj"`
	OProj nn.LinearLayer `weight:"o_proj"`
	QNorm *nn.RMSNorm    `weight:"q_norm"`
	KNorm *nn.RMSNorm    `weight:"k_norm"`
	// Computed fields
	NHeads    int32
	NKVHeads  int32
	HeadDim   int32
	Scale     float32
	RopeTheta float32
}

func linearWeightShape(layer nn.LinearLayer) []int32 {
	switch l := layer.(type) {
	case *nn.Linear:
		if l.Weight != nil {
			return l.Weight.Shape()
		}
	case *nn.QuantizedLinear:
		if l.Weight != nil {
			return l.Weight.Shape()
		}
	}
	return nil
}

// manualScaledDotProductAttention is a CUDA-safe fallback when fast SDPA returns
// empty arrays. q/k/v are [B, H, L, D]; mask is optional additive [1,1,L,L].
func manualScaledDotProductAttention(q, k, v *mlx.Array, scale float32, mask *mlx.Array) *mlx.Array {
	kt := mlx.Transpose(k, 0, 1, 3, 2) // [B,H,D,L]
	scores := mlx.Matmul(q, kt) // [B,H,L,L]
	scores = mlx.Mul(scores, mlx.NewScalarArray(scale))
	if mask != nil {
		scores = mlx.Add(scores, mask)
	} else {
		scores = nn.ApplyCausalMask(scores)
	}
	weights := mlx.Softmax(scores, -1)
	return mlx.Matmul(weights, v)
}

// applyRoPEQwen3 applies the custom RoPE for Qwen3 text encoder
func applyRoPEQwen3(x *mlx.Array, seqLen int32, theta float32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	H := shape[2]
	D := shape[3]
	half := D / 2

	freqsArr := make([]float32, half)
	logTheta := float32(math.Log(float64(theta)))
	for i := int32(0); i < half; i++ {
		freqsArr[i] = float32(math.Exp(float64(-logTheta * float32(i) / float32(half))))
	}
	freqs := mlx.NewArray(freqsArr, []int32{half})

	posArr := make([]float32, seqLen)
	for i := int32(0); i < seqLen; i++ {
		posArr[i] = float32(i)
	}
	pos := mlx.NewArray(posArr, []int32{seqLen})

	posExpanded := mlx.Reshape(pos, seqLen, 1)
	freqsExpanded := mlx.Reshape(freqs, 1, half)
	args := mlx.Mul(posExpanded, freqsExpanded)

	cosVals := mlx.Cos(args)
	sinVals := mlx.Sin(args)
	cosVals = mlx.Reshape(cosVals, seqLen, 1, half)
	sinVals = mlx.Reshape(sinVals, seqLen, 1, half)

	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, H, half})
	x2 := mlx.Slice(x, []int32{0, 0, 0, half}, []int32{B, L, H, D})

	part1 := mlx.Sub(mlx.Mul(x1, cosVals), mlx.Mul(x2, sinVals))
	part2 := mlx.Add(mlx.Mul(x1, sinVals), mlx.Mul(x2, cosVals))

	return mlx.Concatenate([]*mlx.Array{part1, part2}, 3)
}

// Forward computes attention with causal masking and optional padding mask
func (attn *Attention) Forward(x *mlx.Array, mask *mlx.Array, maskMode string) *mlx.Array {
	shape := x.Shape()
	if len(shape) < 2 {
		panic(fmt.Sprintf("qwen3 attention: expected rank>=2 activations, got shape=%v valid=%v qproj=%v",
			shape, x != nil && x.IsValid(), attn.QProj != nil))
	}
	B := shape[0]
	L := shape[1]

	q := attn.QProj.Forward(x)
	k := attn.KProj.Forward(x)
	v := attn.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NKVHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NKVHeads, attn.HeadDim)

	// QK norm uses 1e-6 hardcoded (Qwen3 specific)
	q = attn.QNorm.Forward(q, 1e-6)
	k = attn.KNorm.Forward(k, 1e-6)

	q = applyRoPEQwen3(q, L, attn.RopeTheta)
	k = applyRoPEQwen3(k, L, attn.RopeTheta)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	if attn.NKVHeads < attn.NHeads {
		repeats := attn.NHeads / attn.NKVHeads
		k = repeatKV(k, repeats)
		v = repeatKV(v, repeats)
	}

	// Materialize without cleanup — ResumeCleanup mid-block frees residual `x`
	// (and other live activations) on CUDA Contiguous/mmap graphs.
	mlx.SuppressCleanup()
	mlx.EvalSync(q, k, v)
	if mask != nil {
		mlx.EvalSync(mask)
	}

	// Prefer manual SDPA on CUDA: fast SDPA returns empty arrays on sm_120.
	var out *mlx.Array
	if mlx.MetalIsAvailable() {
		out = mlx.ScaledDotProductAttentionWithSinks(q, k, v, attn.Scale, maskMode, mask, nil)
	}
	if out == nil || !out.IsValid() || len(out.Shape()) == 0 {
		out = manualScaledDotProductAttention(q, k, v, attn.Scale, mask)
		mlx.EvalSync(out)
	}

	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, L, attn.NHeads*attn.HeadDim)

	out = attn.OProj.Forward(out)

	return out
}

// repeatKV repeats key/value heads for GQA
func repeatKV(x *mlx.Array, repeats int32) *mlx.Array {
	if repeats == 1 {
		return x
	}
	shape := x.Shape()
	x = mlx.ExpandDims(x, 2)
	x = mlx.Tile(x, []int32{1, 1, repeats, 1, 1})
	return mlx.Reshape(x, shape[0], shape[1]*repeats, shape[2], shape[3])
}

// MLP implements Qwen3 SwiGLU MLP
type MLP struct {
	GateProj nn.LinearLayer `weight:"gate_proj"`
	UpProj   nn.LinearLayer `weight:"up_proj"`
	DownProj nn.LinearLayer `weight:"down_proj"`
}

// Forward applies the MLP
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := m.GateProj.Forward(x)
	gate = mlx.SiLU(gate)
	up := m.UpProj.Forward(x)
	h := mlx.Mul(gate, up)
	return m.DownProj.Forward(h)
}

// Block represents a single Qwen3 transformer block
type Block struct {
	Attention         *Attention  `weight:"self_attn"`
	MLP               *MLP        `weight:"mlp"`
	InputLayerNorm    *nn.RMSNorm `weight:"input_layernorm"`
	PostAttnLayerNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
}

// Forward applies the Qwen3 block
func (qb *Block) Forward(x *mlx.Array, eps float32, mask *mlx.Array, maskMode string) *mlx.Array {
	h := qb.InputLayerNorm.Forward(x, eps)
	attnOut := qb.Attention.Forward(h, mask, maskMode)
	x = mlx.Add(x, attnOut)

	h = qb.PostAttnLayerNorm.Forward(x, eps)
	mlpOut := qb.MLP.Forward(h)
	x = mlx.Add(x, mlpOut)

	return x
}

// TextEncoder is the full Qwen3 encoder
type TextEncoder struct {
	EmbedTokens *nn.Embedding `weight:"model.embed_tokens"`
	Layers      []*Block      `weight:"model.layers"`
	FinalNorm   *nn.RMSNorm   `weight:"model.norm"`
	*Config
}

// Load loads the Qwen3 text encoder from ollama blob storage.
func (m *TextEncoder) Load(modelManifest *manifest.ModelManifest, configPath string) error {
	fmt.Print("  Loading text encoder... ")

	// Load config from blob
	var cfg Config
	if err := modelManifest.ReadConfigJSON(configPath, &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Config = &cfg
	m.Layers = make([]*Block, cfg.NumHiddenLayers)

	// Load weights from tensor blobs
	weights, err := manifest.LoadWeightsFromManifest(modelManifest, "text_encoder")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	if err := m.loadWeights(weights); err != nil {
		weights.ReleaseAll()
		return err
	}
	// WHY skip ReleaseAll on CUDA: Contiguous(mmap) aliases blob storage; freeing
	// native handles empties TE weight shapes (Forward panics ndim=0). Metal eval
	// fully copies, so ReleaseAll is safe there. Runner/subprocess exit frees maps.
	if !mlx.GPUIsAvailable() || mlx.MetalIsAvailable() {
		weights.ReleaseAll()
	}
	return nil
}

// loadWeights loads weights from any WeightSource into the model
func (m *TextEncoder) loadWeights(weights safetensors.WeightSource) error {
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	m.initComputedFields()
	fmt.Println("✓")
	return nil
}

// initComputedFields initializes computed fields after loading weights
func (m *TextEncoder) initComputedFields() {
	cfg := m.Config
	m.FinalNorm.Eps = cfg.RMSNormEps
	for _, block := range m.Layers {
		// Attention
		block.Attention.NHeads = cfg.NumAttentionHeads
		block.Attention.NKVHeads = cfg.NumKeyValueHeads
		block.Attention.HeadDim = cfg.HeadDim
		block.Attention.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
		block.Attention.RopeTheta = cfg.RopeTheta
		block.Attention.QNorm.Eps = cfg.RMSNormEps
		block.Attention.KNorm.Eps = cfg.RMSNormEps
		// Block norms
		block.InputLayerNorm.Eps = cfg.RMSNormEps
		block.PostAttnLayerNorm.Eps = cfg.RMSNormEps
	}
}

// Forward encodes text tokens with provided attention mask (LxL) and mask mode.
func (te *TextEncoder) Forward(tokens *mlx.Array, attnMask *mlx.Array, maskMode string) *mlx.Array {
	if te.EmbedTokens == nil || te.EmbedTokens.Weight == nil || !te.EmbedTokens.Weight.IsValid() {
		panic(fmt.Sprintf("qwen3 TE: embed weight missing valid=%v", te.EmbedTokens != nil && te.EmbedTokens.Weight != nil && te.EmbedTokens.Weight.IsValid()))
	}
	mlx.SuppressCleanup()
	defer func() { mlx.ResumeCleanup() }()

	h := te.EmbedTokens.Forward(tokens)
	eps := te.RMSNormEps

	for _, layer := range te.Layers {
		h = layer.Forward(h, eps, attnMask, maskMode)
		// Reclaim per-layer SDPA/dequant temps without freeing the residual.
		mlx.Keep(h)
		if attnMask != nil {
			mlx.Keep(attnMask)
		}
		mlx.EvalSync(h)
		_ = mlx.ResumeCleanup()
		mlx.SuppressCleanup()
	}

	// Apply final RMS norm
	h = te.FinalNorm.Forward(h, eps)
	mlx.Keep(h)
	mlx.EvalSync(h)
	return h
}

// ForwardWithLayerOutputs encodes text tokens and returns hidden states from specified layers.
// This is used by Flux2 which needs embeddings from specific intermediate layers.
func (te *TextEncoder) ForwardWithLayerOutputs(tokens *mlx.Array, layerIndices []int, attnMask *mlx.Array, maskMode string) []*mlx.Array {
	h := te.EmbedTokens.Forward(tokens)
	eps := te.RMSNormEps

	outputs := make([]*mlx.Array, len(layerIndices))
	layerSet := make(map[int]int)
	for i, idx := range layerIndices {
		layerSet[idx] = i
	}

	for i, layer := range te.Layers {
		h = layer.Forward(h, eps, attnMask, maskMode)
		if outIdx, ok := layerSet[i]; ok {
			outputs[outIdx] = h
		}
	}

	return outputs
}

// ApplyChatTemplate wraps prompt in Qwen3 chat format.
// If think is true, adds the <think></think> block after the assistant tag
// (matches tokenizer.apply_chat_template with enable_thinking=False in Python).
func ApplyChatTemplate(prompt string, think bool) string {
	base := "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	if think {
		return base + "<think>\n\n</think>\n\n"
	}
	return base
}

// EncodePrompt encodes a text prompt using the tokenizer and encoder.
// If think is true, includes the <think></think> block in the chat template.
func (te *TextEncoder) EncodePrompt(tok *tokenizer.Tokenizer, prompt string, maxLen int, think bool) (*mlx.Array, *mlx.Array) {
	formattedPrompt := ApplyChatTemplate(prompt, think)

	tokens := tok.Encode(formattedPrompt, false)

	if len(tokens) > maxLen {
		tokens = tokens[:maxLen]
	}

	// Pad only to next multiple of 32 (not always maxLen) so DiT attention
	// stays small on 16GB CUDA cards.
	seqLen := len(tokens)
	if seqLen < 1 {
		seqLen = 1
	}
	if pad := (32 - (seqLen % 32)) % 32; pad > 0 {
		seqLen += pad
	}
	if seqLen > maxLen {
		seqLen = maxLen
	}

	maskData := make([]float32, seqLen)
	for i := 0; i < len(tokens) && i < seqLen; i++ {
		maskData[i] = 1.0
	}

	// Get PAD token (different from EOS for Qwen3)
	padToken := tok.PAD()
	if padToken < 0 {
		padToken = tok.EOS() // fallback
	}

	paddedTokens := make([]int32, seqLen)
	copy(paddedTokens, tokens)
	for i := len(tokens); i < seqLen; i++ {
		paddedTokens[i] = padToken
	}

	tokensArr := mlx.NewArrayInt32(paddedTokens, []int32{1, int32(seqLen)})
	maskArr := mlx.NewArray(maskData, []int32{1, int32(seqLen)})

	// Build combined causal + PAD mask [L, L]
	// mask[i,j] = 0 if (j <= i AND valid[j]) else -inf
	L := int32(seqLen)
	validLen := int32(len(tokens))
	combinedMaskData := make([]float32, L*L)
	negInf := float32(-1e9)
	for i := int32(0); i < L; i++ {
		for j := int32(0); j < L; j++ {
			idx := i*L + j
			if j <= i && j < validLen {
				combinedMaskData[idx] = 0
			} else {
				combinedMaskData[idx] = negInf
			}
		}
	}
	maskMat := mlx.NewArray(combinedMaskData, []int32{1, 1, L, L})

	// WHY "array": MLX fast SDPA requires maskMode="array" when passing an
	// additive mask tensor; empty mode + non-nil mask yields empty outputs on CUDA.
	// WHY [1,1,L,L]: fast SDPA expects a broadcastable 4D additive mask.
	embeddings := te.Forward(tokensArr, maskMat, "array")

	return embeddings, maskArr
}

// EncodePromptWithLayers encodes a text prompt and returns embeddings from specified layers.
// Used by Flux2 which concatenates embeddings from multiple intermediate layers.
// If think is true, includes the <think></think> block in the chat template.
// Returns embeddings and padded sequence length.
func (te *TextEncoder) EncodePromptWithLayers(tok *tokenizer.Tokenizer, prompt string, maxLen int, layerIndices []int, think bool) (*mlx.Array, int32) {
	formattedPrompt := ApplyChatTemplate(prompt, think)
	tokens := tok.Encode(formattedPrompt, false)

	if len(tokens) > maxLen {
		tokens = tokens[:maxLen]
	}

	// Pad to maxLen
	padToken := tok.PAD()
	if padToken < 0 {
		padToken = tok.EOS() // fallback
	}
	padded := make([]int32, maxLen)
	copy(padded, tokens)
	for i := len(tokens); i < maxLen; i++ {
		padded[i] = padToken
	}
	tokensArr := mlx.NewArrayInt32(padded, []int32{1, int32(maxLen)})

	// Build combined causal + PAD mask [L, L]
	// mask[i,j] = 0 if (j <= i AND valid[j]) else -inf
	// This combines causal masking with PAD token masking
	L := int32(maxLen)
	validLen := int32(len(tokens))
	maskData := make([]float32, L*L)
	negInf := float32(-1e9)
	for i := int32(0); i < L; i++ {
		for j := int32(0); j < L; j++ {
			idx := i*L + j
			if j <= i && j < validLen {
				maskData[idx] = 0 // allowed: causal OK and not PAD
			} else {
				maskData[idx] = negInf // blocked: future or PAD
			}
		}
	}
	maskMat := mlx.NewArray(maskData, []int32{1, 1, L, L})

	layerOutputs := te.ForwardWithLayerOutputs(tokensArr, layerIndices, maskMat, "array")

	// Concatenate layer outputs along the hidden dimension
	// Each output is [B, L, hidden_dim], result is [B, L, num_layers * hidden_dim]
	embeddings := mlx.Concatenate(layerOutputs, 2)

	// Return embeddings and padded length
	return embeddings, int32(maxLen)
}
