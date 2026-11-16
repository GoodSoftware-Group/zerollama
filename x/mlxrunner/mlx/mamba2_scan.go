package mlx

import "math"

// Mamba2Scan runs the Mamba2 recurrent scan as a graph. Inputs are float32
// with shapes hidden [B, T, H, D], bState/cState [B, T, G, S] where H%G == 0,
// dt [B, T, H], state [B, H, D, S], and a/d/dtBias [H]. captureAll also
// returns the state after every token but the last.
//
// mask, when non-nil, is a [B, T] bool marking real (true) versus padded
// positions. Padded positions are identity steps.
func Mamba2Scan(hidden, bState, cState, dt, state, a, d, dtBias, mask *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	if mask != nil {
		B, T := int32(mask.Dim(0)), int32(mask.Dim(1))
		zero := FromValue(float32(0)).AsType(hidden.DType())
		mask4 := Reshape(mask, B, T, 1, 1)
		hidden = Where(mask4, hidden, zero)
		cState = Where(mask4, cState, zero)
		negInf := FromValue(float32(math.Inf(-1))).AsType(dt.DType())
		dt = Where(Reshape(mask, B, T, 1), dt, negInf)
	}
	return mamba2ScanGraph(hidden, bState, cState, dt, state, a, d, dtBias, captureAll)
}

func repeatMambaGroups(x *Array, repeats int32) *Array {
	if repeats <= 1 {
		return x
	}
	dims := x.Dims()
	x = ExpandDims(x, 3)
	x = Tile(x, []int32{1, 1, 1, repeats, 1})
	return Reshape(x, int32(dims[0]), int32(dims[1]), int32(dims[2])*repeats, int32(dims[3]))
}

func sliceMambaTime(x *Array, t int32) *Array {
	dims := x.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[1], stop[1] = t, t+1
	return Squeeze(SliceStartStop(x, start, stop), 1)
}

func mamba2ScanGraph(hidden, bState, cState, dt, state, a, d, dtBias *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	B := int32(hidden.Dim(0))
	T := int32(hidden.Dim(1))
	H := int32(hidden.Dim(2))
	G := int32(bState.Dim(2))

	if G > 0 && H%G == 0 {
		bState = repeatMambaGroups(bState, H/G)
		cState = repeatMambaGroups(cState, H/G)
	}

	a4 := Reshape(a, 1, H, 1, 1)
	d3 := Reshape(d, 1, H, 1)
	bias2 := Reshape(dtBias, 1, H)

	outs := make([]*Array, 0, T)
	for t := range T {
		xt := sliceMambaTime(hidden, t)
		bt := sliceMambaTime(bState, t)
		ct := sliceMambaTime(cState, t)

		dtt := Add(sliceMambaTime(dt, t), bias2)
		dtt = Softplus(dtt)
		dA := Exp(Mul(Reshape(dtt, B, H, 1, 1), a4))
		dB := Mul(Reshape(dtt, B, H, 1), bt)

		state = Add(Mul(state, dA), Mul(ExpandDims(xt, -1), ExpandDims(dB, 2)))
		yt := Sum(Mul(state, ExpandDims(ct, 2)), 3, false)
		outs = append(outs, Add(yt, Mul(xt, d3)))

		if captureAll && t+1 < T {
			interior = append(interior, state)
		}
	}
	return Stack(outs, 1), state, interior
}
