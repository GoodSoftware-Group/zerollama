package freetokenlab

// PrefillOverlap is §3.1 full-layer double buffering: stream layer l+1 while
// GPU computes layer l. Exposed time is max(compute, transfer) per layer
// after the first fill, versus serialize(compute)+transfer without overlap.
type PrefillOverlap struct {
	Layers         int
	ComputeSec     float64 // GPU time per layer (routed experts)
	TransferSec    float64 // time to stream one full expert pool layer
	SerialSec      float64
	PipelinedSec   float64
	ThroughputGain float64
}

// OverlapPrefill models nLayers of prefill with optional second buffer.
func OverlapPrefill(nLayers int, computePerLayer, transferPerLayer float64) PrefillOverlap {
	if nLayers < 1 {
		nLayers = 1
	}
	serial := float64(nLayers) * (computePerLayer + transferPerLayer)
	// First layer must wait for its transfer; remaining layers hide compute
	// behind the next transfer (or vice versa).
	pipe := transferPerLayer
	if nLayers > 0 {
		hidden := computePerLayer
		if transferPerLayer > hidden {
			hidden = transferPerLayer
		}
		pipe = transferPerLayer + float64(nLayers-1)*hidden
	}
	g := 1.0
	if pipe > 0 {
		g = serial / pipe
	}
	return PrefillOverlap{
		Layers:         nLayers,
		ComputeSec:     computePerLayer,
		TransferSec:    transferPerLayer,
		SerialSec:      serial,
		PipelinedSec:   pipe,
		ThroughputGain: g,
	}
}
