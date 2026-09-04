// Package deepseekv4 implements DeepseekV4ForCausalLM for mlxrunner (Flash q2 DQ).
//
// Why this package: the mlx-lm 2-bit pack is already on disk (~90 GiB) and UMA
// can run it; ggml has no Flash graph. Stubbing CSA/HCA as full MHA is invalid
// (most layers are compress_ratio 4 or 128).
//
// Why --link create: copying 90 GiB blobs is optional; source_dir loads shards in place.
// Why GatherQMM 2-bit: dequanting 256 experts does not fit 128 GiB UMA.
// Why layer-local compressor buffers: greedy decode first; cache.Cache
// snapshots do not yet own CSA/HCA state (speculate rewind can desync).
//
// Docs: docs/mlx-deepseek-v4-flash.md · findings: docs/mlx-deepseek-v4-flash-findings.md
package deepseekv4

import (
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("DeepseekV4ForCausalLM", newModel)
	base.Register("DeepSeekV4ForCausalLM", newModel)
}

type Layer struct {
	AttnNorm *nn.RMSNorm
	FFNNorm  *nn.RMSNorm
	AttnHC   *HyperConn
	FFNHC    *HyperConn
	Attn     *Attention
	MoE      *MoE
}

type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer
	HCHead      *HyperConn
	tok         *tokenizer.Tokenizer
	*Config
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
	cfg.finish()
	cfg.applyQuantFromConfigJSON(configData)
	if qt := root.QuantType(); qt != "" {
		gs, bits, mode := model.QuantizationParams(qt)
		if cfg.QuantBits == 0 {
			cfg.QuantBits = bits
		}
		if cfg.QuantGroupSize == 0 {
			cfg.QuantGroupSize = gs
		}
		if cfg.QuantMode == "" {
			cfg.QuantMode = mode
		}
		if g := root.GroupSize(); g > 0 && cfg.QuantGroupSize == 0 {
			cfg.QuantGroupSize = g
		}
	}
	if cfg.QuantMode == "" {
		cfg.QuantMode = "affine"
	}
	if cfg.QuantBits == 0 {
		cfg.QuantBits = 4
	}
	if cfg.QuantGroupSize == 0 {
		cfg.QuantGroupSize = 64
	}

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tokCfg := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if g, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokCfg.GenerationConfigJSON = g
	}
	if t, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokCfg.TokenizerConfigJSON = t
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokCfg)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	return &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}, nil
}

func loadHC(tensors map[string]*mlx.Array, prefix string) *HyperConn {
	fn := tensors[prefix+".fn"]
	base := tensors[prefix+".base"]
	scale := tensors[prefix+".scale"]
	if scale == nil {
		scale = tensors[prefix+"_scale"]
	}
	if fn == nil || base == nil || scale == nil {
		return nil
	}
	return &HyperConn{Fn: fn, Base: base, Scale: scale}
}

func loadStacked(tensors map[string]*mlx.Array, path string, cfg *Config) (w, scales, biases *mlx.Array, bits, gs int, ok bool) {
	w = tensors[path+".weight"]
	if w == nil {
		return
	}
	scales = tensors[path+".scales"]
	if scales == nil {
		scales = tensors[path+".weight_scale"]
	}
	biases = tensors[path+".biases"]
	if biases == nil {
		biases = tensors[path+".weight_qbias"]
	}
	gs, bits, mode := cfg.quantFor(path)
	if scales != nil {
		gs, bits, mode = model.ResolveLinearQuantParams(gs, bits, mode, cfg.TensorQuant, path+".weight", w, scales)
		_ = mode
	}
	ok = true
	return
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	cfg := m.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "model.embed_tokens", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing model.embed_tokens")
	}
	if w := tensors["model.norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	} else {
		return fmt.Errorf("missing model.norm.weight")
	}
	m.LMHead = linears.Make("lm_head")
	if m.LMHead == nil {
		return fmt.Errorf("missing lm_head")
	}
	m.HCHead = loadHC(tensors, "model.hc_head")
	if m.HCHead == nil {
		return fmt.Errorf("missing model.hc_head")
	}

	useQ := supportsGatherQMM(cfg.QuantMode, 2) || supportsGatherQMM(cfg.QuantMode, cfg.QuantBits)
	for i := range cfg.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d", i)
		layer := &Layer{
			AttnHC: loadHC(tensors, p+".attn_hc"),
			FFNHC:  loadHC(tensors, p+".ffn_hc"),
		}
		if w := tensors[p+".attn_norm.weight"]; w != nil {
			layer.AttnNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[p+".ffn_norm.weight"]; w != nil {
			layer.FFNNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if layer.AttnNorm == nil || layer.FFNNorm == nil || layer.AttnHC == nil || layer.FFNHC == nil {
			return fmt.Errorf("layer %d: missing norms or hyper-connections", i)
		}
		attn := &Attention{
			WQA:   linears.Make(p + ".attn.wq_a"),
			WQB:   linears.Make(p + ".attn.wq_b"),
			WKV:   linears.Make(p + ".attn.wkv"),
			WoA:   linears.Make(p + ".attn.wo_a"),
			WoB:   linears.Make(p + ".attn.wo_b"),
			Ratio: compressRatio(cfg, int(i)),
		}
		if w := tensors[p+".attn.q_norm.weight"]; w != nil {
			attn.QNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[p+".attn.kv_norm.weight"]; w != nil {
			attn.KVNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if s := tensors[p+".attn.attn_sink"]; s != nil {
			attn.Sinks = s
		}
		switch attn.Ratio {
		case 4:
			attn.Comp = loadCompressor(tensors, linears, p+".attn.compressor", 4, cfg.HeadDim, true)
			attn.IdxComp = loadCompressor(tensors, linears, p+".attn.indexer.compressor", 4, cfg.IndexHeadDim, true)
			attn.IdxWQB = linears.Make(p + ".attn.indexer.wq_b")
			attn.IdxProj = linears.Make(p + ".attn.indexer.weights_proj")
			if attn.Comp == nil || attn.IdxComp == nil || attn.IdxWQB == nil || attn.IdxProj == nil {
				return fmt.Errorf("layer %d: missing CSA compressor/indexer", i)
			}
		case 128:
			attn.Comp = loadCompressor(tensors, linears, p+".attn.compressor", 128, cfg.HeadDim, false)
			if attn.Comp == nil {
				return fmt.Errorf("layer %d: missing HCA compressor", i)
			}
		}
		if attn.WQA == nil || attn.WQB == nil || attn.WKV == nil || attn.WoA == nil || attn.WoB == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		layer.Attn = attn

		sw := &SwitchMLP{UseQuantized: useQ}
		gw, gs, gb, gbits, ggs, gok := loadStacked(tensors, p+".ffn.switch_mlp.gate_proj", cfg)
		uw, us, ub, ubits, ugs, uok := loadStacked(tensors, p+".ffn.switch_mlp.up_proj", cfg)
		dw, ds, db, dbits, dgs, dok := loadStacked(tensors, p+".ffn.switch_mlp.down_proj", cfg)
		if !gok || !uok || !dok {
			return fmt.Errorf("layer %d: missing switch_mlp", i)
		}
		if useQ && gs != nil {
			sw.GateWeightQ, sw.GateScales, sw.GateBiases = gw, gs, gb
			sw.UpWeightQ, sw.UpScales, sw.UpBiases = uw, us, ub
			sw.DownWeightQ, sw.DownScales, sw.DownBiases = dw, ds, db
			sw.GateBits, sw.UpBits, sw.DownBits = gbits, ubits, dbits
			sw.GateGroupSize, sw.UpGroupSize, sw.DownGroupSize = ggs, ugs, dgs
		} else {
			sw.UseQuantized = false
			sw.GateWeight, sw.UpWeight, sw.DownWeight = gw, uw, dw
		}
		gate := &MoEGate{Gate: linears.Make(p + ".ffn.gate")}
		if b := tensors[p+".ffn.gate.e_score_correction_bias"]; b != nil {
			gate.Bias = b
		}
		moe := &MoE{
			Gate:      gate,
			SwitchMLP: sw,
			Hash:      int32(i) < cfg.NumHashLayers,
			Tid2eid:   tensors[p+".ffn.gate.tid2eid"],
		}
		if cfg.NSharedExperts > 0 {
			moe.SharedExperts = &SharedExperts{
				GateProj: linears.Make(p + ".ffn.shared_experts.gate_proj"),
				UpProj:   linears.Make(p + ".ffn.shared_experts.up_proj"),
				DownProj: linears.Make(p + ".ffn.shared_experts.down_proj"),
			}
		}
		layer.MoE = moe
		m.Layers[i] = layer
	}
	if n := compressRatio(cfg, 2); n != 0 {
		slog.Info("deepseekv4: loaded Flash graph", "layers", cfg.NumHiddenLayers, "csa", true, "hca", true)
	}
	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	ropeRaw := buildRope(m.Config, false)
	ropeComp := buildRope(m.Config, true)
	h := broadcastHC(m.EmbedTokens.Forward(b.InputIDs), m.HCMult)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		pack := ropeRaw
		if layer.Attn.Ratio != 0 {
			pack = ropeComp
		}
		residual := h
		x, mix := hcPre(h, layer.AttnHC, m.Config)
		x = layer.AttnNorm.Forward(x, m.RMSNormEps)
		x = layer.Attn.Forward(x, b, c, positions, B, L, m.Config, pack)
		h = hcPost(x, residual, mix, m.Config)

		residual = h
		x, mix = hcPre(h, layer.FFNHC, m.Config)
		x = layer.FFNNorm.Forward(x, m.RMSNormEps)
		x = layer.MoE.Forward(x, b.InputIDs, m.Config)
		h = hcPost(x, residual, mix, m.Config)
	}
	h = hcHead(h, m.HCHead, m.Config)
	if m.Norm != nil {
		h = m.Norm.Forward(h, m.RMSNormEps)
	}
	return h
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

func (m *Model) NumLayers() int { return len(m.Layers) }

func (m *Model) MaxContextLength() int { return int(m.MaxPositionEmbeddings) }

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// ParkSpeculation turns off PLD/MTP. Compressor and HC mix live on the
// layer, not in cache.Cache snapshots, so a speculative rewind (SWA
// window 128, restore to ~prefill+draft) panics or desyncs CSA/HCA.
func (m *Model) ParkSpeculation() string {
	return "deepseekv4: compressor/HC state is not in cache.Cache; PLD rewind desyncs SWA"
}

func (m *Model) NewCaches() []cache.Cache {
	out := make([]cache.Cache, len(m.Layers))
	win := int(m.SlidingWindow)
	for i := range out {
		if win > 0 {
			out[i] = cache.NewRotatingKVCache(win)
		} else {
			out[i] = cache.NewKVCache()
		}
	}
	return out
}
