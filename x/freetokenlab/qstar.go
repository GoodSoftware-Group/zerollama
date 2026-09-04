// Package freetokenlab is an offline simulator of FreeToken policies
// (arXiv:2608.16157) for zerollama MoE serving research.
//
// It does not spawn inference or touch production ports. Formulas match
// §3.2 bandwidth-adaptive miss split; cache/prefill/semantic models are
// the paper's policies on synthetic traces until llama-server dumps real
// expert IDs.
package freetokenlab

import "math"

// Bandwidth is measured host-to-device expert transfer (BP) and host-side
// expert kernel (BH) in GB/s, as in FreeToken Table 1.
type Bandwidth struct {
	BP float64 // PCIe / DMA expert-transfer GB/s
	BH float64 // CPU expert-processing GB/s
}

// QStarFillCount is q* = m * BP / BH, rounded, with at least one fill when
// there is a miss so the GPU cache keeps warming (paper §3.2).
func QStarFillCount(misses int, bw Bandwidth) int {
	if misses <= 0 {
		return 0
	}
	if bw.BH <= 0 {
		return misses
	}
	q := int(math.Round(float64(misses) * bw.BP / bw.BH))
	if q < 1 {
		q = 1
	}
	if q > misses {
		q = misses
	}
	// When host cannot outrun the link, residual CPU bandwidth is ~0 and
	// the policy degenerates to pure cache fill.
	if bw.BH <= bw.BP {
		return misses
	}
	return q
}

// MissSplit is one decode layer's residual misses divided between PCIe fill
// and in-place CPU execution.
type MissSplit struct {
	Misses     int
	Fill       int     // |F|
	CPU        int     // |C|
	FillSec    float64 // q S / BP
	CPUSec     float64 // (m-q) S / (BH-BP)  (0 if no residual)
	LayerSec   float64 // max of concurrent branches
	AllFillSec float64 // baseline: every miss is a PCIe fill
}

// SplitMisses applies q* and concurrent branch times for expert size S bytes.
func SplitMisses(misses int, expertBytes int64, bw Bandwidth) MissSplit {
	q := QStarFillCount(misses, bw)
	s := MissSplit{Misses: misses, Fill: q, CPU: misses - q}
	if misses <= 0 || expertBytes <= 0 {
		return s
	}
	gb := float64(expertBytes) / 1e9
	if bw.BP > 0 {
		s.FillSec = float64(q) * gb / bw.BP
		s.AllFillSec = float64(misses) * gb / bw.BP
	}
	resid := bw.BH - bw.BP
	if s.CPU > 0 && resid > 0 {
		s.CPUSec = float64(s.CPU) * gb / resid
	} else if s.CPU > 0 && bw.BH > 0 {
		// No residual bandwidth: CPU path would contend with DMA; treat as
		// serialized host cost (pessimistic, matches "don't use CPU").
		s.CPUSec = float64(s.CPU) * gb / bw.BH
		s.FillSec += s.CPUSec
		s.CPUSec = 0
	}
	s.LayerSec = s.FillSec
	if s.CPUSec > s.LayerSec {
		s.LayerSec = s.CPUSec
	}
	return s
}

// SpeedupVsAllFill is AllFillSec / LayerSec (1 = no win).
func (s MissSplit) SpeedupVsAllFill() float64 {
	if s.LayerSec <= 0 {
		return 1
	}
	return s.AllFillSec / s.LayerSec
}
