package qwen3_5

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestCountMTPLayers(t *testing.T) {
	tensors := map[string]*mlx.Array{
		"mtp.layers.0.self_attn.q_proj.weight": nil,
		"mtp.layers.0.mlp.gate_proj.weight":    nil,
		"mtp.fc.weight":                        nil,
	}
	if got := countMTPLayers(tensors, "mtp."); got != 1 {
		t.Fatalf("count=%d want 1", got)
	}
	if mtpTensorRoot(tensors) != "mtp." {
		t.Fatalf("root %q", mtpTensorRoot(tensors))
	}
}

func TestSelfDraftNilWithoutMTP(t *testing.T) {
	m := &Model{}
	if m.SelfDraft() != nil {
		t.Fatal("expected nil")
	}
}

func TestLoadWeightsMTPHead(t *testing.T) {
	skipIfNoMLX(t)

	cfg := &Config{
		HiddenSize:          4,
		IntermediateSize:    8,
		NumHiddenLayers:     1,
		NumAttentionHeads:   1,
		NumKeyValueHeads:    1,
		HeadDim:             4,
		RMSNormEps:          1e-6,
		TieWordEmbeddings:   true,
		LayerTypes:          []string{"full"},
		LinearNumValueHeads: 1,
		LinearNumKeyHeads:   1,
		LinearKeyHeadDim:    2,
		LinearValueHeadDim:  2,
		LinearConvKernelDim: 4,
		Scale:               0.5,
		RopeDim:             2,
		RopeTheta:           10000,
	}
	m := &Model{Config: cfg, Layers: make([]*Layer, 1)}

	ones4 := mlx.FromValues([]float32{1, 1, 1, 1}, 4)
	q := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 1, 0, 0,
		0, 1, 1, 0,
		0, 0, 1, 1,
		1, 0, 0, 1,
	}, 8, 4)
	k := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}, 4, 4)
	o := k
	gate := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 1, 0, 0,
		0, 1, 1, 0,
		0, 0, 1, 1,
		1, 0, 0, 1,
	}, 8, 4)
	down := mlx.FromValues([]float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
	}, 4, 8)

	layerTensors := func(prefix string) map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + "input_layernorm.weight":          ones4,
			prefix + "post_attention_layernorm.weight": ones4,
			prefix + "self_attn.q_proj.weight":         q,
			prefix + "self_attn.k_proj.weight":         k,
			prefix + "self_attn.v_proj.weight":         k,
			prefix + "self_attn.o_proj.weight":         o,
			prefix + "self_attn.q_norm.weight":         ones4,
			prefix + "self_attn.k_norm.weight":         ones4,
			prefix + "mlp.gate_proj.weight":            gate,
			prefix + "mlp.up_proj.weight":              gate,
			prefix + "mlp.down_proj.weight":            down,
		}
	}

	tensors := map[string]*mlx.Array{
		"model.embed_tokens.weight":        mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4),
		"model.norm.weight":                ones4,
		"mtp.pre_fc_norm_embedding.weight": ones4,
		"mtp.pre_fc_norm_hidden.weight":    ones4,
		"mtp.fc.weight":                    mlx.FromValues(make([]float32, 32), 4, 8),
		"mtp.norm.weight":                  ones4,
	}
	for k, v := range layerTensors("model.layers.0.") {
		tensors[k] = v
	}
	for k, v := range layerTensors("mtp.layers.0.") {
		tensors[k] = v
	}

	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	d := m.SelfDraft()
	if d == nil {
		t.Fatal("SelfDraft")
	}
	caches := m.NewCaches()
	if len(caches) != 2 {
		t.Fatalf("caches=%d want trunk+mtp", len(caches))
	}
	if got := d.DraftCaches(caches); len(got) != 1 || got[0] != caches[1] {
		t.Fatalf("DraftCaches %v", got)
	}

	ids := mlx.FromValues([]int32{0}, 1, 1)
	hidden := mlx.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 1, 1, 4)
	out, proj := d.Draft(&batch.Batch{
		InputIDs:     ids,
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
		Hidden:       hidden,
	}, caches)
	mlx.Eval(out, proj)
	if out.Dim(0) != 1 || out.Dim(out.NumDims()-1) != 4 {
		t.Fatalf("draft shape %v", out.Dims())
	}
	var _ base.DraftModel = d
}

func TestLoadWeightsCompanionMTPDraft(t *testing.T) {
	skipIfNoMLX(t)

	cfg := &Config{
		HiddenSize:          4,
		IntermediateSize:    8,
		NumHiddenLayers:     1,
		NumAttentionHeads:   1,
		NumKeyValueHeads:    1,
		HeadDim:             4,
		RMSNormEps:          1e-6,
		TieWordEmbeddings:   true,
		LayerTypes:          []string{"full"},
		LinearNumValueHeads: 1,
		LinearNumKeyHeads:   1,
		LinearKeyHeadDim:    2,
		LinearValueHeadDim:  2,
		LinearConvKernelDim: 4,
		Scale:               0.5,
		RopeDim:             2,
		RopeTheta:           10000,
	}
	m := &Model{Config: cfg, Layers: make([]*Layer, 1)}

	ones4 := mlx.FromValues([]float32{1, 1, 1, 1}, 4)
	q := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 1, 0, 0,
		0, 1, 1, 0,
		0, 0, 1, 1,
		1, 0, 0, 1,
	}, 8, 4)
	k := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}, 4, 4)
	o := k
	gate := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 1, 0, 0,
		0, 1, 1, 0,
		0, 0, 1, 1,
		1, 0, 0, 1,
	}, 8, 4)
	down := mlx.FromValues([]float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
	}, 4, 8)

	layerTensors := func(prefix string) map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + "input_layernorm.weight":          ones4,
			prefix + "post_attention_layernorm.weight": ones4,
			prefix + "self_attn.q_proj.weight":         q,
			prefix + "self_attn.k_proj.weight":         k,
			prefix + "self_attn.v_proj.weight":         k,
			prefix + "self_attn.o_proj.weight":         o,
			prefix + "self_attn.q_norm.weight":         ones4,
			prefix + "self_attn.k_norm.weight":         ones4,
			prefix + "mlp.gate_proj.weight":            gate,
			prefix + "mlp.up_proj.weight":              gate,
			prefix + "mlp.down_proj.weight":            down,
		}
	}

	tensors := map[string]*mlx.Array{
		"model.embed_tokens.weight":          mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4),
		"model.norm.weight":                  ones4,
		"draft.pre_fc_norm_embedding.weight": ones4,
		"draft.pre_fc_norm_hidden.weight":    ones4,
		"draft.fc.weight":                    mlx.FromValues(make([]float32, 32), 4, 8),
		"draft.norm.weight":                  ones4,
	}
	for k, v := range layerTensors("model.layers.0.") {
		tensors[k] = v
	}
	for k, v := range layerTensors("draft.layers.0.") {
		tensors[k] = v
	}

	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	if m.SelfDraft() != nil {
		t.Fatal("in-checkpoint MTP should be absent")
	}

	d, err := newQwen35MTPDraft(nil, m)
	if err != nil {
		t.Fatalf("newQwen35MTPDraft: %v", err)
	}
	if err := d.LoadWeights(tensors); err != nil {
		t.Fatalf("draft LoadWeights: %v", err)
	}
	if m.SelfDraft() == nil {
		t.Fatal("SelfDraft after companion LoadWeights")
	}
	if n := len(mlx.Collect(d)); n == 0 {
		t.Fatal("mlx.Collect(draft) empty — Sweep would free MTP weights")
	}
	caches := m.NewCaches()
	if len(caches) != 2 {
		t.Fatalf("caches=%d want trunk+mtp", len(caches))
	}
	if got := d.DraftCaches(caches); len(got) != 1 || got[0] != caches[1] {
		t.Fatalf("DraftCaches %v", got)
	}

	ids := mlx.FromValues([]int32{0}, 1, 1)
	hidden := mlx.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 1, 1, 4)
	out, proj := d.Draft(&batch.Batch{
		InputIDs:     ids,
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
		Hidden:       hidden,
	}, caches)
	mlx.Eval(out, proj)
	if out.Dim(0) != 1 || out.Dim(out.NumDims()-1) != 4 {
		t.Fatalf("draft shape %v", out.Dims())
	}
}
