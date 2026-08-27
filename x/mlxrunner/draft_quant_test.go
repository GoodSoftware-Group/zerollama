package mlxrunner

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

type stubDraft struct {
	Gate nn.LinearLayer
	Keep nn.LinearLayer
}

func (s *stubDraft) Draft(*batch.Batch, []cache.Cache) (hidden, projected *mlx.Array) {
	return nil, nil
}
func (s *stubDraft) Unembed(*mlx.Array) *mlx.Array { return nil }
func (s *stubDraft) DraftCaches(caches []cache.Cache) []cache.Cache {
	return caches
}
func (s *stubDraft) LoadWeights(map[string]*mlx.Array) error { return nil }

func TestQuantizeDraftCompanionReplacesDense(t *testing.T) {
	skipIfNoMLX(t)
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}
	t.Setenv("ZEROLLAMA_MLX_DRAFT_QUANT", "")
	w := mlx.Zeros(mlx.DTypeFloat32, 64, 64)
	mlx.Eval(w)
	d := &stubDraft{Gate: nn.NewLinear(w, nil), Keep: nn.WrapDecodeQuant(nn.NewLinear(w, nil))}
	if n := quantizeDraftCompanion(d); n != 2 {
		t.Fatalf("quantized %d layers, want 2", n)
	}
	if _, ok := d.Gate.(*nn.QuantizedLinear); !ok {
		t.Fatalf("Gate type %T", d.Gate)
	}
	if _, ok := d.Keep.(*nn.QuantizedLinear); !ok {
		t.Fatalf("Keep type %T", d.Keep)
	}
}

func TestQuantizeDraftCompanionOff(t *testing.T) {
	skipIfNoMLX(t)
	t.Setenv("ZEROLLAMA_MLX_DRAFT_QUANT", "off")
	w := mlx.Zeros(mlx.DTypeFloat32, 64, 64)
	mlx.Eval(w)
	d := &stubDraft{Gate: nn.NewLinear(w, nil)}
	if n := quantizeDraftCompanion(d); n != 0 {
		t.Fatalf("quantized %d, want 0", n)
	}
	if _, ok := d.Gate.(*nn.Linear); !ok {
		t.Fatalf("Gate type %T", d.Gate)
	}
}
