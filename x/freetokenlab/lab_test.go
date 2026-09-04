package freetokenlab

import (
	"math"
	"strings"
	"testing"
)

func TestQStar5090MatchesPaperRatio(t *testing.T) {
	// q* ≈ m BP/BH ; 4 * 52.7/77.3 ≈ 2.73 → 3 fills, 1 CPU.
	q := QStarFillCount(4, RTX5090Server)
	if q != 3 {
		t.Fatalf("q*=%d want 3", q)
	}
}

func TestQStarMacUMAFillsAll(t *testing.T) {
	if got := QStarFillCount(8, MacUMA); got != 8 {
		t.Fatalf("mac uma q*=%d want 8 (no residual host BW)", got)
	}
}

func TestQStar5080EstSplits(t *testing.T) {
	q := QStarFillCount(4, RTX5080Est)
	if q < 1 || q >= 4 {
		t.Fatalf("5080-est q*=%d want 1..3 (PCIe residual)", q)
	}
}

func TestQStarAlwaysWarmsCache(t *testing.T) {
	// Tiny BP still fills at least one miss.
	q := QStarFillCount(5, Bandwidth{BP: 1, BH: 100})
	if q < 1 {
		t.Fatalf("q*=%d want >=1", q)
	}
}

func TestSplitConcurrentBeatsAllFillOn5090(t *testing.T) {
	const expert = 80 << 20 // 80 MiB expert
	s := SplitMisses(4, expert, RTX5090Server)
	if s.Fill != 3 || s.CPU != 1 {
		t.Fatalf("split fill=%d cpu=%d", s.Fill, s.CPU)
	}
	if s.LayerSec >= s.AllFillSec {
		t.Fatalf("q* layer %.4f should beat all-fill %.4f", s.LayerSec, s.AllFillSec)
	}
	if s.SpeedupVsAllFill() < 1.05 {
		t.Fatalf("speedup %.3f too small", s.SpeedupVsAllFill())
	}
}

func TestSplit4090MoreCPUThan5090(t *testing.T) {
	// 4090: BP/BH smaller → fewer fills, more CPU.
	q5090 := QStarFillCount(10, RTX5090Server)
	q4090 := QStarFillCount(10, RTX4090)
	if q4090 >= q5090 {
		t.Fatalf("4090 q*=%d should be < 5090 q*=%d", q4090, q5090)
	}
}

func TestOverlapHidesCompute(t *testing.T) {
	o := OverlapPrefill(40, 0.02, 0.05)
	if o.PipelinedSec >= o.SerialSec {
		t.Fatalf("pipe %.3f should beat serial %.3f", o.PipelinedSec, o.SerialSec)
	}
	if math.Abs(o.ThroughputGain-o.SerialSec/o.PipelinedSec) > 1e-9 {
		t.Fatalf("gain inconsistent")
	}
}

func TestLRUBeatsStaticOnStickyDecode(t *testing.T) {
	const nExp, slots, layers = 256, 32, 8
	pre := DensePrefill(layers, nExp, 64, 6)
	dec := StickyDecode(layers, 200, nExp, 6, 1, 20, 1)
	st := SimulateCache(PolicyStatic, nExp, slots, pre, dec)
	hot := SimulateCache(PolicyPrefillHot, nExp, slots, pre, dec)
	lru := SimulateCache(PolicyLRU, nExp, slots, pre, dec)
	if lru.MissRate >= st.MissRate {
		t.Fatalf("lru miss %.3f should beat static %.3f", lru.MissRate, st.MissRate)
	}
	if lru.MissRate >= hot.MissRate {
		t.Fatalf("lru miss %.3f should beat prefill-hot %.3f", lru.MissRate, hot.MissRate)
	}
}

func TestKeepTailCompressReplayInvalidatesTail(t *testing.T) {
	reuse, recompute := KeepTailCompressReplay(100, 50, 2000)
	if reuse != 100 || recompute != 2050 {
		t.Fatalf("reuse=%d recompute=%d", reuse, recompute)
	}
	// Naive "checkpoint at tail start" would recompute only the summary (50).
	if recompute == 50 {
		t.Fatal("tail must be in recompute")
	}
	strip := SuffixStripEdit(24000, 400)
	if PrefillTokensWithSemanticAnchor(strip) != 400 {
		t.Fatal(strip)
	}
}

func TestLoadAnemllMiniTrace(t *testing.T) {
	tr, err := LoadAnemllFile("testdata/anemll_mini.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if tr.Records != 6 || tr.NExperts != 6 {
		t.Fatalf("records=%d n_exp=%d", tr.Records, tr.NExperts)
	}
	if len(tr.Prefill) != 2 || len(tr.Decode) != 4 {
		t.Fatalf("prefill=%d decode=%d", len(tr.Prefill), len(tr.Decode))
	}
	lru := SimulateCache(PolicyLRU, tr.NExperts, 2, tr.Prefill, tr.Decode)
	if lru.Accesses == 0 {
		t.Fatal("no decode accesses")
	}
}

func TestPrefetchBeatsOrMatchesLRUOnSticky(t *testing.T) {
	const nExp, slots, layers = 256, 32, 8
	pre := DensePrefill(layers, nExp, 32, 6)
	dec := StickyDecode(layers, 80, nExp, 6, 1, 20, 2)
	lru := SimulateCache(PolicyLRU, nExp, slots, pre, dec)
	pf := SimulateCache(PolicyLRUPrefetch, nExp, slots, pre, dec)
	if pf.MissRate > lru.MissRate+1e-9 {
		t.Fatalf("prefetch miss %.3f worse than lru %.3f", pf.MissRate, lru.MissRate)
	}
}

func TestAdviseMacUMANoCPUSplit(t *testing.T) {
	a := AdviseProfile("mac-uma")
	if a.CPUSplit {
		t.Fatal("mac-uma should not recommend CPU experts")
	}
	if a.PrefetchTemporal {
		t.Fatal("sticky LRU should not turn prefetch on")
	}
	if a.SemanticChatMode != "placeholder" {
		t.Fatalf("chat=%s", a.SemanticChatMode)
	}
	if a.SlotsSticky15 < 1 || a.SlotsSticky15 > 256 {
		t.Fatalf("slots=%d", a.SlotsSticky15)
	}
	if !strings.Contains(a.DoctorLine(), "mac-uma") || !strings.Contains(a.DoctorLine(), "k=6") {
		t.Fatalf("doctor line %q", a.DoctorLine())
	}
}

func TestAdviseProfileForTopK(t *testing.T) {
	a6 := AdviseProfileFor("mac-uma", 256, 6)
	a8 := AdviseProfileFor("mac-uma", 256, 8)
	if a8.TopK != 8 || a8.SlotsSticky15 < a6.SlotsSticky15 {
		t.Fatalf("k6 slots=%d k8 slots=%d", a6.SlotsSticky15, a8.SlotsSticky15)
	}
}

func TestAdvise5080EstCPUSplit(t *testing.T) {
	a := AdviseProfile("5080-est")
	if !a.CPUSplit {
		t.Fatal("5080-est BH>BP should split some misses to CPU in the paper model")
	}
}

func TestRecommendSlotsMonotone(t *testing.T) {
	const nExp, layers = 64, 4
	pre := DensePrefill(layers, nExp, 16, 4)
	dec := ZipfStickyDecode(layers, 40, nExp, 4, 1.15, 2, 3, 1)
	s, rate := RecommendSlots(PolicyLRU, nExp, pre, dec, 0.2)
	if rate > 0.2+1e-9 {
		t.Fatalf("rate %.3f at slots=%d", rate, s)
	}
	lo := SimulateCache(PolicyLRU, nExp, s, pre, dec)
	if lo.MissRate > 0.2+1e-9 {
		t.Fatalf("verify %.3f", lo.MissRate)
	}
	if s > 1 {
		hi := SimulateCache(PolicyLRU, nExp, s-1, pre, dec)
		if hi.MissRate <= 0.2 {
			t.Fatalf("slots-1=%d still %.3f ≤ target — search not tight", s-1, hi.MissRate)
		}
	}
}

func TestSlotBankForExpertsScales(t *testing.T) {
	s64 := SlotBankForExperts(64)
	s256 := SlotBankForExperts(256)
	if s64 < 1 || s256 < s64 {
		t.Fatalf("slots 64=%d 256=%d", s64, s256)
	}
	if s256 > 256 {
		t.Fatalf("256-expert bank %d", s256)
	}
}

func TestRamCapSlotsTable(t *testing.T) {
	want := map[float64]int{8: 16, 16: 32, 36: 64, 128: 128, 256: 128}
	for ram, slots := range want {
		if got := RamCapSlots(ram); got != slots {
			t.Fatalf("ram=%.0f got %d want %d", ram, got, slots)
		}
	}
}

func TestAdviseSlotBankRAMWinsOnSmallHost(t *testing.T) {
	r := SlotBankForExperts(256)
	a := AdviseSlotBank(256, 8)
	if a.Routing != r || a.RamCap != 16 {
		t.Fatalf("%+v routing=%d", a, r)
	}
	if a.Recommend != min(r, 16) {
		t.Fatalf("recommend %d", a.Recommend)
	}
	wide := AdviseSlotBank(256, 128)
	if wide.Recommend != r {
		t.Fatalf("128GiB should keep routing %d got %+v", r, wide)
	}
	if wide.MissRate > 0.15+1e-9 {
		t.Fatalf("128GiB miss %.3f", wide.MissRate)
	}
	tight := AdviseSlotBank(256, 8)
	if tight.Recommend == tight.Routing && tight.Routing > 16 {
		t.Fatalf("8GiB should RAM-cap %+v", tight)
	}
}

func TestSlotBankBytes(t *testing.T) {
	if SlotBankBytes(16, 256<<20, 256) != 16<<20 {
		t.Fatal(SlotBankBytes(16, 256<<20, 256))
	}
	if SlotBankBytes(0, 100, 8) != 0 {
		t.Fatal("zero slots")
	}
}

func TestIIDZipfStaticBeatsLRU(t *testing.T) {
	const nExp, slots, layers = 256, 95, 8 // ~37%
	pre := DensePrefill(layers, nExp, 32, 6)
	dec := ZipfDecode(layers, 80, nExp, 6, 1.15, 1)
	st := SimulateCache(PolicyStatic, nExp, slots, pre, dec)
	lru := SimulateCache(PolicyLRU, nExp, slots, pre, dec)
	if st.MissRate >= lru.MissRate {
		t.Fatalf("i.i.d. Zipf: static %.3f should beat LRU %.3f (no locality)", st.MissRate, lru.MissRate)
	}
}
