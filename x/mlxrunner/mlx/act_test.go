package mlx

import (
	"testing"
)

func TestSwiGLUScaledMatchesSeparateScaling(t *testing.T) {
	for _, tt := range []struct {
		name               string
		gateScale, upScale []float32
	}{
		{name: "no scales"},
		{name: "scalar scales", gateScale: []float32{0.75}, upScale: []float32{0.75}},
		{name: "gate scale only", gateScale: []float32{0.75}},
		{name: "up scale only", upScale: []float32{0.75}},
		{name: "per-output scales", gateScale: []float32{0.5, 0.75, 1.25, 1.5}, upScale: []float32{1.5, 1.25, 0.75, 0.5}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			withMLXThread(t, func() {
				EnableCompile()
				gate := FromValues([]float32{-3.25, -1.5, -0.25, 0.5, 1.75, 3, 4.5, 6}, 2, 4).AsType(DTypeBFloat16)
				up := FromValues([]float32{2.5, -2, 1.25, -0.75, 0.125, 1.5, -3.5, 5}, 2, 4).AsType(DTypeBFloat16)
				storedScale := func(factors []float32) *Array {
					if factors == nil {
						return nil
					}
					if len(factors) == 1 {
						return FromValue(factors[0])
					}
					return FromValues(factors, len(factors))
				}
				gateScale, upScale := storedScale(tt.gateScale), storedScale(tt.upScale)

				wantGate, wantUp := gate, up
				if gateScale != nil {
					wantGate = Mul(wantGate, gateScale).AsType(wantGate.DType())
				}
				if upScale != nil {
					wantUp = Mul(wantUp, upScale).AsType(wantUp.DType())
				}
				want := SwiGLU(wantGate, wantUp)
				got := SwiGLUScaled(gate, gateScale, up, upScale)
				wantF32, gotF32 := want.AsType(DTypeFloat32), got.AsType(DTypeFloat32)
				Eval(wantF32, gotF32)

				wantValues, gotValues := wantF32.Floats(), gotF32.Floats()
				for i := range wantValues {
					if gotValues[i] != wantValues[i] {
						t.Fatalf("SwiGLUScaled()[%d] = %v, want %v", i, gotValues[i], wantValues[i])
					}
				}
			})
		})
	}
}
