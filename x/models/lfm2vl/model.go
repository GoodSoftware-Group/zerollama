// Package lfm2vl implements LFM2-VL: the LFM2 decoder, a SigLIP2 NaFlex tower,
// and the pixel-shuffle projector. Images past the token budget are split into
// 512px tiles plus a thumbnail, with <|img_row_R_col_C|> and <|img_thumbnail|>
// markers around the image-token runs.
package lfm2vl

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/lfm2"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/models/siglip"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() { base.Register("Lfm2VlForConditionalGeneration", newModel) }

type Model struct {
	*lfm2.Model
	vision       *siglip.Tower
	proj         *projector
	visionCfg    *siglip.VisionConfig
	lim          imageLimits
	imageToken   int32
	projectorAct string
	special      bool
	channels     int
}

type projector struct {
	norm   *nn.LayerNorm
	fc1    nn.LinearLayer
	fc2    nn.LinearLayer
	factor int
	act    func(*mlx.Array) *mlx.Array
}

type envelope struct {
	ImageTokenID          int32           `json:"image_token_id"`
	DownsampleFactor      int             `json:"downsample_factor"`
	EncoderPatchSize      int             `json:"encoder_patch_size"`
	MaxImageTokens        int             `json:"max_image_tokens"`
	MinImageTokens        int             `json:"min_image_tokens"`
	MaxPixelsTolerance    float64         `json:"max_pixels_tolerance"`
	MaxTiles              int             `json:"max_tiles"`
	MinTiles              int             `json:"min_tiles"`
	TileSize              int             `json:"tile_size"`
	UseThumbnail          *bool           `json:"use_thumbnail"`
	UseImageSpecialTokens *bool           `json:"use_image_special_tokens"`
	DoImageSplitting      *bool           `json:"do_image_splitting"`
	ProjectorBias         bool            `json:"projector_bias"`
	ProjectorHiddenAct    string          `json:"projector_hidden_act"`
	TextConfig            json.RawMessage `json:"text_config"`
	VisionConfig          json.RawMessage `json:"vision_config"`
}

type preJSON struct {
	MaxPixelsTolerance *float64 `json:"max_pixels_tolerance"`
	Resample           *int     `json:"resample"`
	DoImageSplitting   *bool    `json:"do_image_splitting"`
	UseThumbnail       *bool    `json:"use_thumbnail"`
	MinTiles           *int     `json:"min_tiles"`
	MaxTiles           *int     `json:"max_tiles"`
	TileSize           *int     `json:"tile_size"`
	MinImageTokens     *int     `json:"min_image_tokens"`
	MaxImageTokens     *int     `json:"max_image_tokens"`
	EncoderPatchSize   *int     `json:"encoder_patch_size"`
	DownsampleFactor   *int     `json:"downsample_factor"`
}

func newModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	var env envelope
	if err := json.Unmarshal(configData, &env); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if len(env.TextConfig) == 0 || len(env.VisionConfig) == 0 {
		return nil, fmt.Errorf("lfm2-vl config is missing text_config or vision_config")
	}
	if env.ImageTokenID <= 0 {
		return nil, fmt.Errorf("lfm2-vl config is missing image_token_id")
	}
	var text lfm2.Config
	if err := json.Unmarshal(env.TextConfig, &text); err != nil {
		return nil, fmt.Errorf("parse text_config: %w", err)
	}
	quantBlob := env.TextConfig
	if !hasQuant(quantBlob) {
		quantBlob = configData
	}
	if err := lfm2.Prepare(&text, root, quantBlob); err != nil {
		return nil, err
	}
	vision, err := siglip.ParseVisionConfig(env.VisionConfig)
	if err != nil {
		return nil, err
	}
	tok, err := lfm2.LoadTokenizer(root, configData)
	if err != nil {
		return nil, err
	}
	var pre preJSON
	if raw, err := root.Manifest.ReadConfig("preprocessor_config.json"); err == nil {
		_ = json.Unmarshal(raw, &pre)
	}
	return &Model{
		Model:        lfm2.NewPrepared(&text, tok),
		visionCfg:    vision,
		lim:          limitsFrom(env, &pre),
		imageToken:   env.ImageTokenID,
		projectorAct: env.ProjectorHiddenAct,
		special:      env.UseImageSpecialTokens == nil || *env.UseImageSpecialTokens,
		channels:     int(vision.NumChannels),
	}, nil
}

func hasQuant(data []byte) bool {
	var q struct {
		Quantization struct {
			Bits int `json:"bits"`
		} `json:"quantization"`
	}
	_ = json.Unmarshal(data, &q)
	return q.Quantization.Bits > 0
}

func limitsFrom(env envelope, pre *preJSON) imageLimits {
	lim := imageLimits{
		downsample: orInt(env.DownsampleFactor, 2),
		patch:      orInt(env.EncoderPatchSize, 16),
		minTokens:  orInt(env.MinImageTokens, 64),
		maxTokens:  orInt(env.MaxImageTokens, 256),
		tile:       orInt(env.TileSize, 512),
		minTiles:   orInt(env.MinTiles, 2),
		maxTiles:   orInt(env.MaxTiles, 10),
		tolerance:  env.MaxPixelsTolerance,
		split:      env.DoImageSplitting == nil || *env.DoImageSplitting,
		thumbnail:  env.UseThumbnail == nil || *env.UseThumbnail,
		bicubic:    true,
	}
	if lim.tolerance <= 0 {
		lim.tolerance = 2
	}
	if pre == nil {
		return lim
	}
	if pre.DownsampleFactor != nil && *pre.DownsampleFactor > 0 {
		lim.downsample = *pre.DownsampleFactor
	}
	if pre.EncoderPatchSize != nil && *pre.EncoderPatchSize > 0 {
		lim.patch = *pre.EncoderPatchSize
	}
	if pre.MinImageTokens != nil && *pre.MinImageTokens > 0 {
		lim.minTokens = *pre.MinImageTokens
	}
	if pre.MaxImageTokens != nil && *pre.MaxImageTokens > 0 {
		lim.maxTokens = *pre.MaxImageTokens
	}
	if pre.TileSize != nil && *pre.TileSize > 0 {
		lim.tile = *pre.TileSize
	}
	if pre.MinTiles != nil && *pre.MinTiles > 0 {
		lim.minTiles = *pre.MinTiles
	}
	if pre.MaxTiles != nil && *pre.MaxTiles > 0 {
		lim.maxTiles = *pre.MaxTiles
	}
	if pre.MaxPixelsTolerance != nil && *pre.MaxPixelsTolerance > 0 {
		lim.tolerance = *pre.MaxPixelsTolerance
	}
	if pre.DoImageSplitting != nil {
		lim.split = *pre.DoImageSplitting
	}
	if pre.UseThumbnail != nil {
		lim.thumbnail = *pre.UseThumbnail
	}
	if pre.Resample != nil {
		lim.bicubic = *pre.Resample != 2
	}
	if !lim.split {
		lim.minTiles, lim.maxTiles = 1, 1
	}
	return lim
}

func orInt(v, fallback int) int {
	if v > 0 {
		return v
	}
	return fallback
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	tower, err := siglip.LoadTower(linears, tensors, m.visionCfg,
		"model.vision_tower.vision_model.",
		"model.vision_tower.",
	)
	if err != nil {
		return err
	}
	if !tower.NaFlex() {
		return fmt.Errorf("lfm2-vl expects a SigLIP2 NaFlex vision tower")
	}
	proj, err := loadProjector(linears, tensors, m.lim.downsample, m.projectorAct)
	if err != nil {
		return err
	}
	m.vision = tower
	m.proj = proj
	return nil
}

func loadProjector(linears model.LinearFactory, tensors map[string]*mlx.Array, factor int, actName string) (*projector, error) {
	const prefix = "model.multi_modal_projector."
	fc1 := linears.Make(prefix + "linear_1")
	fc2 := linears.Make(prefix + "linear_2")
	if fc1 == nil || fc2 == nil {
		return nil, fmt.Errorf("missing %slinear_1 or linear_2", prefix)
	}
	p := &projector{fc1: fc1, fc2: fc2, factor: factor, act: projectorAct(actName)}
	if w := tensors[prefix+"layer_norm.weight"]; w != nil {
		p.norm = &nn.LayerNorm{Weight: w, Bias: tensors[prefix+"layer_norm.bias"], Eps: 1e-5}
	}
	return p, nil
}

func projectorAct(name string) func(*mlx.Array) *mlx.Array {
	switch name {
	case "gelu_pytorch_tanh", "gelu_tanh":
		return mlx.GELUApprox
	case "relu":
		return mlx.ReLU
	case "silu":
		return mlx.SiLU
	default:
		return mlx.GELU
	}
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	h := m.EmbedTokens.Forward(b.InputIDs)
	if len(b.Media) > 0 {
		h = scatterMedia(h, b, int(m.HiddenSize))
	}
	return m.ForwardEmbeddings(h, b, caches)
}

type mediaJob struct {
	tiles []tileRef
	runs  [][2]int
}

type tileRef struct {
	offset, count, gridH, gridW int
}

func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	if m.vision == nil {
		return nil, fmt.Errorf("lfm2-vl vision tower is not loaded")
	}
	prepared := &base.PreparedRequest{}
	for s, seg := range segments {
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		if seg.Kind != "image" {
			return nil, fmt.Errorf("lfm2-vl does not support %s input", seg.Kind)
		}
		job, expansion, err := m.expandImage(seg.Data, len(prepared.Tokens))
		if err != nil {
			return nil, err
		}
		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, expansion...)
		patchDim := m.lim.patch * m.lim.patch * m.channels
		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    s,
			MediaData: job.patches,
			Dims:      []int{len(job.patches) / patchDim, patchDim},
			Opaque:    job.mediaJob,
			Causal:    true,
		})
	}
	return prepared, nil
}

type builtImage struct {
	patches  []float32
	mediaJob mediaJob
}

func (m *Model) expandImage(data []byte, base int) (builtImage, []int32, error) {
	img, err := decodeSize(data)
	if err != nil {
		return builtImage{}, nil, err
	}
	lay, err := layoutImage(img.w, img.h, m.lim)
	if err != nil {
		return builtImage{}, nil, err
	}
	patch := m.lim.patch
	var built builtImage
	var expansion []int32
	appendRun := func(n int) error {
		if n < 1 {
			return errImage("image produced no tokens")
		}
		built.mediaJob.runs = append(built.mediaJob.runs, [2]int{base + len(expansion), n})
		for range n {
			expansion = append(expansion, m.imageToken)
		}
		return nil
	}
	if m.special {
		id, err := special(m.Tokenizer(), "<|image_start|>")
		if err != nil {
			return builtImage{}, nil, err
		}
		expansion = append(expansion, id)
	}
	for i, tile := range lay.tiles {
		if tile.thumbnail {
			id, err := special(m.Tokenizer(), "<|img_thumbnail|>")
			if err != nil {
				return builtImage{}, nil, err
			}
			expansion = append(expansion, id)
		} else if lay.gridW > 1 || lay.gridH > 1 {
			row := i/lay.gridW + 1
			col := i%lay.gridW + 1
			id, err := special(m.Tokenizer(), fmt.Sprintf("<|img_row_%d_col_%d|>", row, col))
			if err != nil {
				return builtImage{}, nil, err
			}
			expansion = append(expansion, id)
		}
		n, err := tokensForTile(tile, m.lim)
		if err != nil {
			return builtImage{}, nil, err
		}
		if err := appendRun(n); err != nil {
			return builtImage{}, nil, err
		}
		pixels, err := siglip.NormalizeImage(data, tile.width, tile.height, m.channels, m.lim.bicubic)
		if err != nil {
			return builtImage{}, nil, err
		}
		packed := siglip.Patchify(pixels, tile.height, tile.width, patch, m.channels)
		patchDim := patch * patch * m.channels
		built.mediaJob.tiles = append(built.mediaJob.tiles, tileRef{
			offset: len(built.patches) / patchDim,
			count:  len(packed) / patchDim,
			gridH:  tile.height / patch,
			gridW:  tile.width / patch,
		})
		built.patches = append(built.patches, packed...)
	}
	if m.special {
		id, err := special(m.Tokenizer(), "<|image_end|>")
		if err != nil {
			return builtImage{}, nil, err
		}
		expansion = append(expansion, id)
	}
	return built, expansion, nil
}

func special(tok *tokenizer.Tokenizer, name string) (int32, error) {
	id, ok := tok.GetSpecialToken(name)
	if !ok {
		return 0, fmt.Errorf("lfm2-vl tokenizer is missing %s", name)
	}
	return id, nil
}

type decodedSize struct{ w, h int }

func decodeSize(data []byte) (decodedSize, error) {
	// NormalizeImage decodes; a size-only pass avoids resizing twice for the
	// layout decision. image.Decode is enough and the pixels are read later.
	cfg, err := imageConfig(data)
	if err != nil {
		return decodedSize{}, err
	}
	return decodedSize{w: cfg.w, h: cfg.h}, nil
}

func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	job := item.Opaque.(mediaJob)
	outs := make([]*mlx.Array, 0, len(job.tiles))
	for _, tile := range job.tiles {
		row := data.Slice(mlx.Slice(tile.offset, tile.offset+tile.count), mlx.Slice())
		tokens, err := m.vision.PatchTokens(row, tile.gridH, tile.gridW)
		if err != nil {
			panic(err)
		}
		outs = append(outs, m.proj.project(tokens, tile.gridH, tile.gridW))
	}
	if len(outs) == 1 {
		return outs[0]
	}
	return mlx.Concatenate(outs, 0)
}

func (p *projector) project(tokens *mlx.Array, gridH, gridW int) *mlx.Array {
	hidden := tokens.Dim(tokens.NumDims() - 1)
	flat := tokens
	if tokens.NumDims() == 3 {
		flat = mlx.Reshape(tokens, int32(gridH*gridW), int32(hidden))
	}
	idx := pixelShuffleIndex(gridH, gridW, p.factor)
	gathered := mlx.Take(flat, mlx.FromValues(idx, len(idx)), 0)
	group := p.factor * p.factor
	unshuffled := mlx.Reshape(gathered, int32(len(idx)/group), int32(group*hidden))
	if p.norm != nil {
		unshuffled = p.norm.Forward(unshuffled)
	}
	return p.fc2.Forward(p.act(p.fc1.Forward(unshuffled)))
}

func pixelShuffleIndex(gridH, gridW, factor int) []int32 {
	hOut, wOut := gridH/factor, gridW/factor
	idx := make([]int32, 0, hOut*wOut*factor*factor)
	for rr := 0; rr < hOut; rr++ {
		for cc := 0; cc < wOut; cc++ {
			for dh := 0; dh < factor; dh++ {
				for dw := 0; dw < factor; dw++ {
					idx = append(idx, int32((rr*factor+dh)*gridW+(cc*factor+dw)))
				}
			}
		}
	}
	return idx
}

func scatterMedia(h *mlx.Array, b *batch.Batch, hidden int) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		job := item.Opaque.(mediaJob)
		base := int(b.SeqOffsets[item.Seq])
		qEnd := base + int(b.SeqQueryLens[item.Seq])
		featAt := 0
		for _, run := range job.runs {
			start, n := run[0], run[1]
			lo := max(start, base)
			hi := min(start+n, qEnd)
			if hi > lo {
				feat := item.Features.Slice(mlx.Slice(featAt+(lo-start), featAt+(hi-start)), mlx.Slice())
				feat = mlx.Reshape(feat.AsType(h.DType()), 1, int32(hi-lo), int32(hidden))
				h = h.SliceUpdate(feat, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(lo-base, hi-base), mlx.Slice())
			}
			featAt += n
		}
	}
	return h
}
