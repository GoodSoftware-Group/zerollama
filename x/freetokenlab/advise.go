package freetokenlab

import (
	"fmt"
	"math"
)

// Advice turns lab numbers into Flash-MoE / chat settings we can actually set.
// anemll has no q* CPU-expert flag; slot-bank is already LRU-style.
type Advice struct {
	Profile           string
	CPUSplit          bool
	PrefetchTemporal  bool
	SemanticChatMode  string // placeholder | summary (opt-in)
	SlotsSticky15     int    // LRU slots for ~15% miss on sticky Zipf
	SlotsSticky15Frac float64
	Experts           int
	TopK              int
	Notes             []string
}

// AdviseProfile maps a Profiles() key to operator knobs (256 experts, top-k=6).
func AdviseProfile(name string) Advice {
	return AdviseProfileFor(name, 256, 6)
}

// AdviseProfileFor uses GGUF expert_count / expert_used_count.
func AdviseProfileFor(name string, nExp, k int) Advice {
	bw, ok := Profiles()[name]
	if !ok {
		name = "mac-uma"
		bw = MacUMA
	}
	if nExp < 2 {
		nExp = 256
	}
	if k < 1 {
		k = 6
	}
	if k > nExp {
		k = nExp
	}
	a := Advice{
		Profile:          name,
		CPUSplit:         QStarFillCount(4, bw) < 4,
		PrefetchTemporal: false,
		SemanticChatMode: "placeholder",
		Experts:          nExp,
		TopK:             k,
	}

	const layers = 8
	pre := DensePrefill(layers, nExp, 32, k)
	sticky := ZipfStickyDecode(layers, 120, nExp, k, 1.15, 3, 4, 1)
	slots := SlotBankForExpertsK(nExp, k)
	a.SlotsSticky15 = slots
	a.SlotsSticky15Frac = float64(slots) / float64(nExp)

	iid := ZipfDecode(layers, 80, nExp, k, 1.15, 1)
	lruIID := SimulateCache(PolicyLRU, nExp, slots, pre, iid)
	pfIID := SimulateCache(PolicyLRUPrefetch, nExp, slots, pre, iid)
	lruSt := SimulateCache(PolicyLRU, nExp, slots, pre, sticky)
	pfSt := SimulateCache(PolicyLRUPrefetch, nExp, slots, pre, sticky)
	if pfIID.MissRate+0.01 < lruIID.MissRate {
		a.PrefetchTemporal = true
	}

	if a.CPUSplit {
		a.Notes = append(a.Notes,
			"q* wants some CPU experts on this PCIe/host split — anemll has no flag; measure BP/BH before inventing one")
	} else {
		a.Notes = append(a.Notes, "unified or BP≥BH: fill every miss in the slot-bank; do not run CPU experts")
	}
	a.Notes = append(a.Notes, fmt.Sprintf(
		"sticky LRU miss@%d slots (%.0f%% of %d k=%d) ≈ %.3f; prefetch sticky %.3f iid lru %.3f prefetch %.3f",
		slots, a.SlotsSticky15Frac*100, nExp, k, lruSt.MissRate, pfSt.MissRate, lruIID.MissRate, pfIID.MissRate))
	a.Notes = append(a.Notes,
		"agent KV: chat_compress placeholder (default) keeps exact prefix; summary is opt-in and invalidates tail KV")
	if !a.PrefetchTemporal {
		a.Notes = append(a.Notes,
			"leave ZEROLLAMA_FLASH_MOE_PREFETCH unset; --moe-prefetch-temporal does not help when LRU holds the stay-set")
	} else {
		a.Notes = append(a.Notes, "ZEROLLAMA_FLASH_MOE_PREFETCH=1 may help i.i.d. Zipf routing")
	}
	return a
}

// RecommendSlots is the smallest per-layer slot count whose LRU miss rate is
// ≤ target (or nExperts if the target is unreachable).
func RecommendSlots(policy CachePolicy, nExperts int, prefill, decode []TraceStep, targetMiss float64) (slots int, rate float64) {
	if nExperts < 1 {
		return 1, 1
	}
	full := SimulateCache(policy, nExperts, nExperts, prefill, decode)
	if full.MissRate > targetMiss {
		return nExperts, full.MissRate
	}
	lo, hi := 1, nExperts
	best, bestRate := nExperts, full.MissRate
	for lo <= hi {
		mid := (lo + hi) / 2
		r := SimulateCache(policy, nExperts, mid, prefill, decode)
		if r.MissRate <= targetMiss {
			best, bestRate = mid, r.MissRate
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	return best, bestRate
}

// DoctorLine is one-line status for `zerollama doctor`.
func (a Advice) DoctorLine() string {
	cpu := "no-cpu-split"
	if a.CPUSplit {
		cpu = "q*-cpu (no anemll flag)"
	}
	pf := "prefetch=off"
	if a.PrefetchTemporal {
		pf = "prefetch=on"
	}
	return fmt.Sprintf("%s %s %s slots~%d k=%d (sticky≤15%% miss) chat=%s",
		a.Profile, cpu, pf, a.SlotsSticky15, a.TopK, a.SemanticChatMode)
}

// SlotBankForExperts is --moe-slot-bank for a GGUF expert_count (top-k=6).
func SlotBankForExperts(nExperts int) int {
	return SlotBankForExpertsK(nExperts, 6)
}

// SlotBankForExpertsK uses GGUF expert_used_count when k>0 (else 6).
func SlotBankForExpertsK(nExperts, k int) int {
	return AdviseSlotBankK(nExperts, k, 1e9, 0).Routing
}

func stickyZipfTraces(nExperts, k int) (n, kUse int, pre, dec []TraceStep) {
	n = nExperts
	if n < 2 {
		return 0, k, nil, nil
	}
	kUse = k
	if kUse < 1 {
		kUse = 6
	}
	layers, preTok, decSteps := 8, 32, 120
	if n < 64 {
		layers, preTok, decSteps = 4, 16, 60
	}
	if kUse > n {
		kUse = n
	}
	pre = DensePrefill(layers, n, preTok, kUse)
	dec = ZipfStickyDecode(layers, decSteps, n, kUse, 1.15, 3, 4, 1)
	return n, kUse, pre, dec
}

// SlotBankAdvice is routing-sized LRU vs anemll RAM table. Serve still omits
// --moe-slot-bank unless the operator copies Recommend into env/options.
type SlotBankAdvice struct {
	Routing   int
	RamCap    int
	Recommend int
	BankBytes int64
	MissRate  float64 // sticky Zipf LRU miss at Recommend
}

// SlotBankBytes is packed expert tensors × slots / expert_count (all MoE layers).
func SlotBankBytes(slots int, expertTensorBytes int64, nExperts int) int64 {
	if slots < 1 || expertTensorBytes < 1 || nExperts < 1 {
		return 0
	}
	return int64(slots) * expertTensorBytes / int64(nExperts)
}

// AdviseSlotBank is min(routing sticky-Zipf bank, RAM table) at top-k=6.
func AdviseSlotBank(nExperts int, ramGiB float64) SlotBankAdvice {
	return AdviseSlotBankK(nExperts, 6, ramGiB, 0)
}

// AdviseSlotBankK uses GGUF expert_used_count and optional packed expert bytes.
func AdviseSlotBankK(nExperts, k int, ramGiB float64, expertTensorBytes int64) SlotBankAdvice {
	n, _, pre, dec := stickyZipfTraces(nExperts, k)
	if n < 2 {
		return SlotBankAdvice{Routing: 1, RamCap: RamCapSlots(ramGiB), Recommend: 1, MissRate: 1}
	}
	r, rRate := RecommendSlots(PolicyLRU, n, pre, dec, 0.15)
	if r < 1 {
		r = 1
	}
	cap := RamCapSlots(ramGiB)
	rec := r
	if cap < rec {
		rec = cap
	}
	if rec < 1 {
		rec = 1
	}
	miss := rRate
	if rec != r {
		miss = SimulateCache(PolicyLRU, n, rec, pre, dec).MissRate
	}
	return SlotBankAdvice{
		Routing:   r,
		RamCap:    cap,
		Recommend: rec,
		BankBytes: SlotBankBytes(rec, expertTensorBytes, n),
		MissRate:  miss,
	}
}

// RamCapSlots matches docs/flash-moe.md (8→16, 16→32, 36→64, 128→128).
// Values between rows interpolate; above 128 GiB stays 128.
func RamCapSlots(ramGiB float64) int {
	if ramGiB <= 0 {
		return 16
	}
	type row struct {
		ram   float64
		slots int
	}
	table := []row{{8, 16}, {16, 32}, {36, 64}, {128, 128}}
	if ramGiB <= table[0].ram {
		return table[0].slots
	}
	last := table[len(table)-1]
	if ramGiB >= last.ram {
		return last.slots
	}
	for i := 1; i < len(table); i++ {
		if ramGiB <= table[i].ram {
			lo, hi := table[i-1], table[i]
			t := (ramGiB - lo.ram) / (hi.ram - lo.ram)
			return int(math.Round(float64(lo.slots) + t*float64(hi.slots-lo.slots)))
		}
	}
	return last.slots
}
