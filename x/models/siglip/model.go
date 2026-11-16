// Package siglip loads SigLIP and SigLIP 2 as MLX embedding models.
//
// An image in the request returns the vision tower's pooled vector. Text
// alone returns the text tower's pooled vector. The embed runner mean-pools
// the forward result and L2-normalizes it; both towers therefore return a
// single row, [1, 1, D], so that mean is the vector itself and the extra
// scale cancels under L2.
package siglip

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
	for _, arch := range []string{
		"SiglipModel",
		"SiglipVisionModel",
		"SiglipTextModel",
		"Siglip2Model",
		"Siglip2VisionModel",
		"Siglip2TextModel",
	} {
		base.Register(arch, newModel)
	}
}

var _ base.Model = (*Model)(nil)

// Model is either tower, or both. Image requests use the vision tower even
// when the caller also sent text; SigLIP scores the two embeddings apart,
// and this API returns one vector.
type Model struct {
	Config Config
	Vision *visionTower
	Text   *textTower

	tok         *tokenizer.Tokenizer
	fixedH      int
	fixedW      int
	maxPatches  int
	bilinear    bool
	padID       int32
	quantGroup  int
	quantBits   int
	quantMode   string
	tensorQuant map[string]*model.TensorQuantInfo
}

func newModel(root *model.Root) (base.Model, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	cfg, err := parseConfig(data)
	if err != nil {
		return nil, err
	}
	group, bits, mode := model.QuantizationParams(root.QuantType())
	if root.GroupSize() > 0 {
		group = root.GroupSize()
	}
	m := &Model{
		Config:      cfg,
		padID:       1,
		quantGroup:  group,
		quantBits:   bits,
		quantMode:   mode,
		tensorQuant: root.AllTensorQuant(),
	}
	if cfg.Text != nil {
		m.padID = cfg.Text.PadTokenID
	}
	m.applyPreprocessor(root)
	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tokCfg := &tokenizer.TokenizerConfig{ConfigJSON: data}
	if gen, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokCfg.TokenizerConfigJSON = gen
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokCfg)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	m.tok = tok
	if tok.PAD() >= 0 {
		m.padID = tok.PAD()
	}
	return m, nil
}

type preprocessor struct {
	Resample      int     `json:"resample"`
	MaxNumPatches int     `json:"max_num_patches"`
	PatchSize     int     `json:"patch_size"`
	Size          sizeBox `json:"size"`
}

type sizeBox struct {
	Height int `json:"height"`
	Width  int `json:"width"`
}

func (m *Model) applyPreprocessor(root *model.Root) {
	if m.Config.Vision == nil {
		return
	}
	v := m.Config.Vision
	m.bilinear = v.NaFlex
	if !v.NaFlex && v.ImageSize > 0 {
		m.fixedH, m.fixedW = int(v.ImageSize), int(v.ImageSize)
	}
	m.maxPatches = int(v.NumPatches)
	raw, err := root.Manifest.ReadConfig("preprocessor_config.json")
	if err != nil {
		return
	}
	var pre preprocessor
	if json.Unmarshal(raw, &pre) != nil {
		return
	}
	if pre.Size.Height > 0 && pre.Size.Width > 0 {
		m.fixedH, m.fixedW = pre.Size.Height, pre.Size.Width
	}
	if pre.MaxNumPatches > 0 {
		m.maxPatches = pre.MaxNumPatches
	}
	// PIL resample: 2 bilinear, 3 bicubic. A published size wins over NaFlex.
	switch pre.Resample {
	case 2:
		m.bilinear = true
	case 3:
		m.bilinear = false
	}
	if m.fixedH > 0 && pre.MaxNumPatches == 0 {
		v.NaFlex = false
	}
}

func (m *Model) Forward(b *batch.Batch, _ []cache.Cache) *mlx.Array {
	if len(b.Media) > 0 {
		return m.poolMedia(b.Media)
	}
	if m.Text == nil {
		panic("siglip: text requested but this checkpoint has no text tower")
	}
	return m.Text.forward(b.InputIDs, m.padID)
}

func (m *Model) poolMedia(items []batch.MediaItem) *mlx.Array {
	vecs := make([]*mlx.Array, 0, len(items))
	for _, item := range items {
		if item.Features != nil {
			vecs = append(vecs, item.Features)
		}
	}
	if len(vecs) == 0 {
		panic("siglip: image request produced no features")
	}
	if len(vecs) == 1 {
		return asRow(vecs[0])
	}
	stacked := mlx.Concatenate(vecs, 0)
	mean := mlx.Mean(stacked, 0, false)
	return mlx.Reshape(mean, 1, 1, int32(mean.Dim(mean.NumDims()-1)))
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return x }

func (m *Model) NumLayers() int {
	if m.Config.Text != nil {
		return int(m.Config.Text.NumHiddenLayers)
	}
	if m.Config.Vision != nil {
		return int(m.Config.Vision.NumHiddenLayers)
	}
	return 0
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) MaxContextLength() int {
	if m.Config.Text == nil {
		return 0
	}
	return int(m.Config.Text.MaxPositionEmbeddings)
}

func (m *Model) HiddenSize() int32 {
	if m.Config.Text != nil {
		return m.Config.Text.ProjectionSize
	}
	if m.Config.Vision != nil {
		return m.Config.Vision.HiddenSize
	}
	return 0
}

func (m *Model) SupportsImages() bool { return m.Vision != nil }
func (m *Model) SupportsAudio() bool  { return false }

func (m *Model) MediaTokenStrings() (string, string, string, string, string, string) {
	return "", "", "", "", "", ""
}

func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	if m.Vision == nil {
		return nil, fmt.Errorf("this model does not support images")
	}
	prepared := &base.PreparedRequest{}
	for i, seg := range segments {
		if seg.Kind == "" {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		if seg.Kind != "image" {
			return nil, fmt.Errorf("siglip does not support %s media", seg.Kind)
		}
		plan, err := prepareImage(seg.Data, m.Config.Vision, m.fixedH, m.fixedW, m.maxPatches, m.bilinear)
		if err != nil {
			return nil, err
		}
		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, m.padID)
		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, start + 1},
			Source:    i,
			MediaData: plan.payload(),
			Dims:      plan.dims(),
			Opaque:    plan,
		})
	}
	if len(prepared.Tokens) == 0 {
		prepared.Tokens = []int32{m.padID}
	}
	return prepared, nil
}

func (m *Model) EncodeMedia(prepared *base.PreparedRequest) ([]batch.MediaItem, error) {
	if m.Vision == nil {
		return nil, fmt.Errorf("this model does not support images")
	}
	items := make([]batch.MediaItem, len(prepared.Items))
	for i := range prepared.Items {
		plan, ok := prepared.Items[i].Opaque.(imagePlan)
		if !ok {
			return nil, fmt.Errorf("siglip media item %d is missing its image plan", i)
		}
		items[i] = batch.MediaItem{Features: m.Vision.encode(plan)}
	}
	return items, nil
}

func (p imagePlan) payload() []float32 {
	if len(p.Patches) > 0 {
		return p.Patches
	}
	return p.Pixels
}

func (p imagePlan) dims() []int {
	if len(p.Patches) > 0 {
		return []int{1, p.GridH * p.GridW, len(p.Patches) / (p.GridH * p.GridW)}
	}
	return []int{1, p.Height, p.Width, 3}
}

type encoderBlock struct {
	Norm1 *nn.LayerNorm
	Norm2 *nn.LayerNorm
	Q     nn.LinearLayer
	K     nn.LinearLayer
	V     nn.LinearLayer
	Out   nn.LinearLayer
	FC1   nn.LinearLayer
	FC2   nn.LinearLayer
}

func (b *encoderBlock) forward(x *mlx.Array, heads int32, act func(*mlx.Array) *mlx.Array) *mlx.Array {
	residual := x
	h := b.Norm1.Forward(x)
	h = attention(h, b.Q, b.K, b.V, b.Out, heads)
	x = mlx.Add(residual, h)
	residual = x
	h = b.Norm2.Forward(x)
	h = b.FC2.Forward(act(b.FC1.Forward(h)))
	return mlx.Add(residual, h)
}

func attention(x *mlx.Array, qProj, kProj, vProj, oProj nn.LinearLayer, heads int32) *mlx.Array {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	hidden := int32(x.Dim(2))
	headDim := hidden / heads
	shape := func(y *mlx.Array) *mlx.Array {
		return mlx.Transpose(mlx.Reshape(y, B, L, heads, headDim), 0, 2, 1, 3)
	}
	q, k, v := shape(qProj.Forward(x)), shape(kProj.Forward(x)), shape(vProj.Forward(x))
	scale := float32(1 / math.Sqrt(float64(headDim)))
	out := mlx.FastScaledDotProductAttention(q, k, v, scale, "", nil)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, hidden)
	return oProj.Forward(out)
}

func activation(name string) func(*mlx.Array) *mlx.Array {
	switch name {
	case "gelu":
		return mlx.GELU
	default:
		return mlx.GELUApprox
	}
}

func asRow(x *mlx.Array) *mlx.Array {
	d := int32(x.Dim(x.NumDims() - 1))
	return mlx.Reshape(x, 1, 1, d)
}
