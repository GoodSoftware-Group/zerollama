package deepseekv4

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

func TestParseQuantizationPerTensor(t *testing.T) {
	raw := []byte(`{
		"hidden_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"head_dim": 8,
		"hc_mult": 4,
		"quantization": {
			"group_size": 64,
			"bits": 4,
			"mode": "affine",
			"model.layers.0.ffn.switch_mlp.gate_proj": {"group_size": 32, "bits": 2, "mode": "affine"}
		}
	}`)
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.finish()
	cfg.applyQuantFromConfigJSON(raw)
	if cfg.QuantBits != 4 || cfg.QuantGroupSize != 64 {
		t.Fatalf("defaults bits=%d gs=%d", cfg.QuantBits, cfg.QuantGroupSize)
	}
	gs, bits, mode := cfg.quantFor("model.layers.0.ffn.switch_mlp.gate_proj")
	if gs != 32 || bits != 2 || mode != "affine" {
		t.Fatalf("expert quant gs=%d bits=%d mode=%s", gs, bits, mode)
	}
}

func TestArchitectureRegistered(t *testing.T) {
	if !base.Registered("DeepseekV4ForCausalLM") {
		t.Fatal("DeepseekV4ForCausalLM not registered")
	}
}

func TestSupportsGatherQMMIncludes2Bit(t *testing.T) {
	if !supportsGatherQMM("affine", 2) {
		t.Fatal("2-bit affine must use GatherQMM (do not dequant 256 experts)")
	}
	if supportsGatherQMM("mxfp4", 4) {
		t.Fatal("mxfp4 is not the Flash expert path")
	}
}

func TestDsv4YarnAttnFactorIsOne(t *testing.T) {
	cfg := &Config{
		HeadDim:           8,
		QKRopeHeadDim:     4,
		RopeTheta:         10000,
		CompressRopeTheta: 160000,
		RopeScaling:       &nn.RopeParameters{Factor: 16, Type: "yarn"},
	}
	cfg.finish()
	if pack := buildRope(cfg, true); pack.mscale != 1 {
		t.Fatalf("compress YaRN mscale=%v want 1 (mlx-lm / HF V4)", pack.mscale)
	}
	if pack := buildRope(cfg, false); pack.mscale != 1 {
		t.Fatalf("ratio-0 mscale=%v want 1", pack.mscale)
	}
}

func TestSinkhornRowColSums(t *testing.T) {
	x := mlx.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	out := sinkhorn(x, 4, 1e-6)
	mlx.Eval(out)
	d := out.Dims()
	if len(d) != 4 || d[2] != 2 || d[3] != 2 {
		t.Fatalf("dims %v", d)
	}
}

func TestCompressedWritePosStride(t *testing.T) {
	got := compressedWritePos(3, 4)
	want := []int32{0, 4, 8}
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("got %v want %v", got, want)
	}
	hca := compressedWritePos(2, 128)
	if hca[0] != 0 || hca[1] != 128 {
		t.Fatalf("hca write pos %v", hca)
	}
}

func TestRollTimeToOldest(t *testing.T) {
	// Ring write index 1, K=4: slots [1,2,3,0] become oldest-first.
	x := mlx.FromValues([]float32{
		0, 1, 2, 3,
	}, 1, 1, 4, 1)
	got := rollTimeToOldest(x, 1)
	mlx.Eval(got)
	vals := got.Floats()
	want := []float32{1, 2, 3, 0}
	if len(vals) != 4 || vals[0] != want[0] || vals[1] != want[1] || vals[2] != want[2] || vals[3] != want[3] {
		t.Fatalf("got %v want %v", vals, want)
	}
	id := rollTimeToOldest(x, 4)
	mlx.Eval(id)
	if id.Dim(2) != 4 {
		t.Fatalf("identity dim %d", id.Dim(2))
	}
}

func TestInverseRoPELastDims(t *testing.T) {
	cfg := &Config{HeadDim: 8, QKRopeHeadDim: 4, RopeTheta: 10000}
	cfg.finish()
	x := mlx.FromValues([]float32{
		0, 0, 0, 0, 1, 0, 0, 1,
	}, 1, 1, 1, 8)
	off := mlx.FromValues([]int32{3}, 1)
	pack := buildRope(cfg, false)
	y := applyRoPE(x, off, cfg, pack, false)
	z := applyRoPE(y, off, cfg, pack, true)
	mlx.Eval(z)
	if len(z.Dims()) != 4 {
		t.Fatalf("dims %v", z.Dims())
	}
}

func TestGroupedWoADense(t *testing.T) {
	// G=2, in=4, oLora=3 → weight [6, 4]
	w := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 1, 0, 0,
		0, 0, 1, 1,
	}, 6, 4)
	layer := nn.NewLinear(w, nil)
	x := mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}, 1, 1, 2, 4)
	out := groupedWoA(layer, x, 2, 3)
	mlx.Eval(out)
	d := out.Dims()
	if len(d) != 3 || d[2] != 6 {
		t.Fatalf("want [1,1,6] got %v", d)
	}
}

func TestBroadcastHCRank(t *testing.T) {
	h := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 1, 4)
	out := broadcastHC(h, 4)
	mlx.Eval(out)
	d := out.Dims()
	if len(d) != 4 || d[2] != 4 || d[3] != 4 {
		t.Fatalf("want [1,1,4,4] got %v", d)
	}
}
