package qwen3next

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// mtpHead is in-checkpoint multi-token prediction (Ollama GGUF `mtp.*`).
// llama.cpp graph_mtp / mlx-serve mtp: enorm+hnorm, fc on concat, full-attn layers, shared lm_head.
type mtpHead struct {
	FC    *nn.Linear  `gguf:"fc"`
	ENorm *nn.RMSNorm `gguf:"pre_fc_norm_embedding"`
	HNorm *nn.RMSNorm `gguf:"pre_fc_norm_hidden"`
	Norm  *nn.RMSNorm `gguf:"norm"`

	Layers []Layer `gguf:"layers"`
}

var _ model.MTPSpec = (*Model)(nil)

const maxMTPLayers = 4

func newMTPHead(numLayers int, moe bool) *mtpHead {
	if numLayers <= 0 {
		return nil
	}
	if numLayers > maxMTPLayers {
		numLayers = maxMTPLayers
	}
	h := &mtpHead{Layers: make([]Layer, numLayers)}
	for i := range h.Layers {
		h.Layers[i].Operator = &FullAttention{}
		if moe {
			h.Layers[i].MLP = &sparse{}
		} else {
			h.Layers[i].MLP = &dense{}
		}
	}
	return h
}

func (h *mtpHead) loaded() bool {
	if h == nil || h.FC == nil || h.FC.Weight == nil || h.ENorm == nil || h.ENorm.Weight == nil || h.HNorm == nil || h.HNorm.Weight == nil {
		return false
	}
	if len(h.Layers) == 0 {
		return false
	}
	attn, ok := h.Layers[0].Operator.(*FullAttention)
	return ok && attn != nil && attn.Query != nil && attn.Query.Weight != nil
}

func (h *mtpHead) trim() {
	if h == nil {
		return
	}
	n := 0
	for i := range h.Layers {
		attn, ok := h.Layers[i].Operator.(*FullAttention)
		if !ok || attn == nil || attn.Query == nil || attn.Query.Weight == nil {
			break
		}
		n = i + 1
	}
	h.Layers = h.Layers[:n]
}

// DraftForward runs the MTP head. tokens are the draft-step token ids;
// hidden is the target hidden (pre-output-norm) aligned with those tokens.
// KV for MTP layers is stored after the trunk layers on the same HybridCache.
func (m *Model) DraftForward(ctx ml.Context, batch input.Batch, hidden ml.Tensor) (ml.Tensor, error) {
	if m == nil || !m.MTP.loaded() {
		return nil, fmt.Errorf("qwen3next: no MTP head")
	}
	if hidden == nil {
		return nil, fmt.Errorf("qwen3next: MTP draft needs target hidden")
	}
	if m.Cache == nil {
		return nil, fmt.Errorf("qwen3next: MTP draft needs cache")
	}
	cache, ok := m.Cache.(*HybridCache)
	if !ok {
		return nil, fmt.Errorf("qwen3next: MTP draft needs hybrid cache")
	}

	positions := m.buildPositions(ctx, batch)
	emb := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	e := m.MTP.ENorm.Forward(ctx, emb, m.eps)
	h := m.MTP.HNorm.Forward(ctx, hidden, m.eps)
	cur := m.MTP.FC.Forward(ctx, e.Concat(ctx, h, 0))

	base := len(m.Layers)
	for i, layer := range m.MTP.Layers {
		cache.SetLayer(base + i)
		var outputs ml.Tensor
		if i == len(m.MTP.Layers)-1 {
			outputs = batch.Outputs
		}
		var err error
		cur, err = layer.Forward(ctx, base+i, cur, positions, outputs, cache, m.Options)
		if err != nil {
			return nil, err
		}
	}
	if m.MTP.Norm != nil && m.MTP.Norm.Weight != nil {
		cur = m.MTP.Norm.Forward(ctx, cur, m.eps)
	} else {
		cur = m.OutputNorm.Forward(ctx, cur, m.eps)
	}
	return m.Output.Forward(ctx, cur), nil
}
