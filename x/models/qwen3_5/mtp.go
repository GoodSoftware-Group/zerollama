package qwen3_5

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

// mtpHead is Qwen 3.5/3.6/Next in-checkpoint multi-token prediction
// (mlx-serve mtp.zig / vLLM qwen3_next_mtp): enorm+hnorm, fc on concat,
// one or more full-attention decoder layers, shared lm_head.
type mtpHead struct {
	target *Model
	prefix string // companion DRAFT prefix (e.g. "draft."); empty when in-checkpoint

	// Exported so mlx.Collect can pin weights before Sweep (gemma assistant pattern).
	FC     nn.LinearLayer
	ENorm  *nn.RMSNorm
	HNorm  *nn.RMSNorm
	Norm   *nn.RMSNorm
	Embed  nn.EmbeddingLayer
	Head   nn.LinearLayer
	Layers []*Layer
}

var _ base.DraftModel = (*mtpHead)(nil)

func mtpTensorRoot(tensors map[string]*mlx.Array) string {
	for _, root := range []string{"mtp.", "language_model.mtp."} {
		if tensors[root+"fc.weight"] != nil || tensors[root+"pre_fc_norm_hidden.weight"] != nil {
			return root
		}
	}
	for name := range tensors {
		if i := strings.Index(name, "mtp.layers."); i >= 0 {
			return name[:i] + "mtp."
		}
	}
	return ""
}

func normalizeMTPRoot(prefix string) string {
	if prefix == "" {
		return ""
	}
	if !strings.HasSuffix(prefix, ".") {
		return prefix + "."
	}
	return prefix
}

// newQwen35MTPDraft loads mlx-community Qwen3.6-*-MTP packs (model_type
// qwen3_5_mtp) attached via Modelfile DRAFT. Tensors land as draft.fc.weight
// etc. Do not +1-shift those norms — the HF MLX pack is already in mlx layout.
func newQwen35MTPDraft(root *model.Root, target base.Model) (base.DraftModel, error) {
	m, ok := target.(*Model)
	if !ok || m == nil {
		return nil, fmt.Errorf("qwen3_5_mtp requires a Qwen 3.5/3.6 target")
	}
	prefix := "draft."
	if root != nil && root.Draft != nil && root.Draft.TensorPrefix != "" {
		prefix = normalizeMTPRoot(root.Draft.TensorPrefix)
	}
	return &mtpHead{target: m, prefix: prefix}, nil
}

func countMTPLayers(tensors map[string]*mlx.Array, root string) int {
	needle := root + "layers."
	max := -1
	for name := range tensors {
		rest, ok := strings.CutPrefix(name, needle)
		if !ok {
			continue
		}
		dot := strings.IndexByte(rest, '.')
		if dot <= 0 {
			continue
		}
		n, err := strconv.Atoi(rest[:dot])
		if err != nil {
			continue
		}
		if n > max {
			max = n
		}
	}
	return max + 1
}

func (m *Model) loadMTPHead(tensors map[string]*mlx.Array, linears model.LinearFactory, shouldShift, useQuantizedExperts bool) error {
	root := mtpTensorRoot(tensors)
	n := countMTPLayers(tensors, root)
	if root == "" || n == 0 {
		return nil
	}
	head, err := buildMTPHead(m, tensors, root, linears, shouldShift, useQuantizedExperts)
	if err != nil {
		return err
	}
	m.mtp = head
	return nil
}

func buildMTPHead(m *Model, tensors map[string]*mlx.Array, root string, linears model.LinearFactory, shouldShift, useQuantizedExperts bool) (*mtpHead, error) {
	n := countMTPLayers(tensors, root)
	if n == 0 {
		return nil, fmt.Errorf("no MTP layers under %q", root)
	}
	cfg := m.Config
	fcName := strings.TrimSuffix(root, ".") + ".fc"
	if root == "" {
		fcName = "fc"
	}
	fc := linears.Make(fcName)
	if fc == nil {
		return nil, fmt.Errorf("missing %sfc", root)
	}
	eW := maybeShiftNormWeight(root+"pre_fc_norm_embedding.weight", tensors[root+"pre_fc_norm_embedding.weight"], shouldShift)
	hW := maybeShiftNormWeight(root+"pre_fc_norm_hidden.weight", tensors[root+"pre_fc_norm_hidden.weight"], shouldShift)
	if eW == nil || hW == nil {
		return nil, fmt.Errorf("missing %spre_fc_norm_{embedding,hidden}", root)
	}
	head := &mtpHead{
		target: m,
		prefix: root,
		FC:     fc,
		ENorm:  nn.NewRMSNorm(eW, cfg.RMSNormEps),
		HNorm:  nn.NewRMSNorm(hW, cfg.RMSNormEps),
		Embed:  m.EmbedTokens,
		Head:   m.LMHead,
		Layers: make([]*Layer, n),
	}
	if w := tensors[root+"embed_tokens.weight"]; w != nil {
		if emb := model.MakeEmbeddingLayer(tensors, root+"embed_tokens", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant); emb != nil {
			head.Embed = emb
		}
	}
	if lm := linears.Make(root + "shared_head.head"); lm != nil {
		head.Head = lm
	}
	normW := maybeShiftNormWeight(root+"norm.weight", tensors[root+"norm.weight"], shouldShift)
	if normW == nil {
		normW = maybeShiftNormWeight(root+"shared_head.norm.weight", tensors[root+"shared_head.norm.weight"], shouldShift)
	}
	if normW != nil {
		head.Norm = nn.NewRMSNorm(normW, cfg.RMSNormEps)
	} else {
		head.Norm = m.Norm
	}

	for i := range n {
		layerPrefix := fmt.Sprintf("%slayers.%d", root, i)
		layer, err := loadMTPLayer(tensors, linears, cfg, layerPrefix, shouldShift, useQuantizedExperts)
		if err != nil {
			return nil, fmt.Errorf("mtp layer %d: %w", i, err)
		}
		head.Layers[i] = layer
	}
	return head, nil
}

func (h *mtpHead) LoadWeights(tensors map[string]*mlx.Array) error {
	if h == nil || h.target == nil {
		return fmt.Errorf("qwen3_5_mtp: missing target")
	}
	if len(h.Layers) > 0 {
		return nil
	}
	root := h.prefix
	if root == "" {
		root = "draft."
	}
	cfg := h.target.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	useQuantizedExperts := supportsGatherQMM(cfg.QuantMode, cfg.QuantBits)
	built, err := buildMTPHead(h.target, tensors, root, linears, false, useQuantizedExperts)
	if err != nil {
		return err
	}
	*h = *built
	h.target.mtp = h
	return nil
}

func loadMTPLayer(tensors map[string]*mlx.Array, linears model.LinearFactory, cfg *Config, layerPrefix string, shouldShift, useQuantizedExperts bool) (*Layer, error) {
	layer := &Layer{IsLinear: false}
	if w := maybeShiftNormWeight(layerPrefix+".input_layernorm.weight", tensors[layerPrefix+".input_layernorm.weight"], shouldShift); w != nil {
		layer.InputNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}
	if w := maybeShiftNormWeight(layerPrefix+".post_attention_layernorm.weight", tensors[layerPrefix+".post_attention_layernorm.weight"], shouldShift); w != nil {
		layer.PostAttentionNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}
	if layer.InputNorm == nil || layer.PostAttentionNorm == nil {
		return nil, fmt.Errorf("missing layer norms")
	}

	attn := &FullAttention{}
	attn.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
	attn.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
	attn.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
	attn.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")
	if w := maybeShiftNormWeight(layerPrefix+".self_attn.q_norm.weight", tensors[layerPrefix+".self_attn.q_norm.weight"], shouldShift); w != nil {
		attn.QNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}
	if w := maybeShiftNormWeight(layerPrefix+".self_attn.k_norm.weight", tensors[layerPrefix+".self_attn.k_norm.weight"], shouldShift); w != nil {
		attn.KNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}
	if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil {
		return nil, fmt.Errorf("missing full attention projections")
	}
	if attn.QNorm == nil || attn.KNorm == nil {
		return nil, fmt.Errorf("missing full attention q/k norms")
	}
	layer.FullAttn = attn

	if linears.Make(layerPrefix+".mlp.gate_proj") != nil {
		mlp := &DenseMLP{
			GateProj: linears.Make(layerPrefix + ".mlp.gate_proj"),
			UpProj:   linears.Make(layerPrefix + ".mlp.up_proj"),
			DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
		}
		if mlp.GateProj == nil || mlp.UpProj == nil || mlp.DownProj == nil {
			return nil, fmt.Errorf("missing dense mlp projections")
		}
		layer.MLP = mlp
		return layer, nil
	}

	moe := &SparseMoE{}
	moe.Gate = linears.Make(layerPrefix + ".mlp.gate")
	if moe.Gate == nil {
		return nil, fmt.Errorf("missing moe gate")
	}
	switchMLP, err := loadSwitchMLP(tensors, cfg, useQuantizedExperts, layerPrefix)
	if err != nil {
		return nil, err
	}
	moe.SwitchMLP = switchMLP
	sharedGateProj := linears.Make(layerPrefix + ".mlp.shared_expert.gate_proj")
	sharedUpProj := linears.Make(layerPrefix + ".mlp.shared_expert.up_proj")
	sharedDownProj := linears.Make(layerPrefix + ".mlp.shared_expert.down_proj")
	if sharedGateProj != nil && sharedUpProj != nil && sharedDownProj != nil {
		moe.SharedExpert = &DenseMLP{
			GateProj: sharedGateProj,
			UpProj:   sharedUpProj,
			DownProj: sharedDownProj,
		}
		moe.SharedExpertGate = linears.Make(layerPrefix + ".mlp.shared_expert_gate")
	}
	layer.MLP = moe
	return layer, nil
}

func (h *mtpHead) DraftCaches(caches []cache.Cache) []cache.Cache {
	n := len(h.Layers)
	if n == 0 || len(caches) < n {
		return nil
	}
	return caches[len(caches)-n:]
}

func (h *mtpHead) Unembed(x *mlx.Array) *mlx.Array {
	return h.Head.Forward(x)
}

func (h *mtpHead) Draft(b *batch.Batch, caches []cache.Cache) (hidden, projected *mlx.Array) {
	cfg := h.target.Config
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	emb := h.ENorm.Forward(h.Embed.Forward(b.InputIDs), cfg.RMSNormEps)
	hs := h.HNorm.Forward(b.Hidden, cfg.RMSNormEps)
	hState := h.FC.Forward(emb.Concatenate(-1, hs))
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	draftKV := h.DraftCaches(caches)
	for i, layer := range h.Layers {
		var c cache.Cache
		if i < len(draftKV) {
			c = draftKV[i]
		}
		hState = layer.Forward(hState, b, c, positions, B, L, cfg)
	}
	if h.Norm != nil {
		hState = h.Norm.Forward(hState, cfg.RMSNormEps)
	}
	return hState, hState
}
