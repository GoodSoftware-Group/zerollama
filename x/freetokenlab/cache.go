package freetokenlab

// CachePolicy names FreeToken Fig. 4b placement styles.
type CachePolicy int

const (
	// PolicyStatic pins experts 0..slots-1 (llama.cpp load-time split).
	PolicyStatic CachePolicy = iota
	// PolicyPrefillHot pins the union of the first prefill's experts, frozen
	// for decode (KTransformers-style prefill-updated placement).
	PolicyPrefillHot
	// PolicyLRU is a shared LRU over (layer, expert) like FreeToken.
	PolicyLRU
	// PolicyLRUPrefetch is LRU plus one-step temporal prefetch (Flash-MoE
	// --moe-prefetch-temporal): last step's experts are installed before the
	// current lookup and do not count as this step's misses.
	PolicyLRUPrefetch
)

func (p CachePolicy) String() string {
	switch p {
	case PolicyStatic:
		return "static"
	case PolicyPrefillHot:
		return "prefill-hot"
	case PolicyLRU:
		return "lru"
	case PolicyLRUPrefetch:
		return "lru+prefetch1"
	default:
		return "unknown"
	}
}

// TraceStep is routed expert IDs for one token at one MoE layer (deduped).
type TraceStep struct {
	Layer   int
	Experts []int
}

// CacheSimResult is miss counts over a decode trace at fixed slot capacity.
type CacheSimResult struct {
	Policy   CachePolicy
	Slots    int
	Accesses int
	Misses   int
	MissRate float64
}

// SimulateCache replays steps against a per-layer slot budget (same capacity
// for all policies). PrefillHot uses prefill to choose the pinned set.
func SimulateCache(policy CachePolicy, nExperts, slotsPerLayer int, prefill, decode []TraceStep) CacheSimResult {
	if slotsPerLayer < 1 {
		slotsPerLayer = 1
	}
	pinned := map[int]map[int]bool{} // layer → expert → resident (static/hot)
	lru := map[int]*lruSet{}

	initLayer := func(layer int) {
		if _, ok := pinned[layer]; ok {
			return
		}
		pinned[layer] = map[int]bool{}
		lru[layer] = newLRU(slotsPerLayer)
		switch policy {
		case PolicyStatic:
			n := slotsPerLayer
			if n > nExperts {
				n = nExperts
			}
			for e := 0; e < n; e++ {
				pinned[layer][e] = true
			}
		case PolicyPrefillHot:
			seen := []int{}
			used := map[int]bool{}
			for _, st := range prefill {
				if st.Layer != layer {
					continue
				}
				for _, e := range st.Experts {
					if used[e] {
						continue
					}
					used[e] = true
					seen = append(seen, e)
				}
			}
			for i := 0; i < slotsPerLayer && i < len(seen); i++ {
				pinned[layer][seen[i]] = true
			}
			// If prefill did not fill the bank, pad with low IDs.
			for e := 0; len(pinned[layer]) < slotsPerLayer && e < nExperts; e++ {
				pinned[layer][e] = true
			}
		}
	}

	last := map[int][]int{}
	var accesses, misses int
	for _, st := range decode {
		initLayer(st.Layer)
		cur := uniq(st.Experts)
		if policy == PolicyLRUPrefetch {
			for _, e := range last[st.Layer] {
				_ = lru[st.Layer].touch(e)
			}
		}
		for _, e := range cur {
			accesses++
			switch policy {
			case PolicyLRU, PolicyLRUPrefetch:
				if !lru[st.Layer].touch(e) {
					misses++
				}
			default:
				if !pinned[st.Layer][e] {
					misses++
				}
			}
		}
		last[st.Layer] = cur
	}
	r := CacheSimResult{Policy: policy, Slots: slotsPerLayer, Accesses: accesses, Misses: misses}
	if accesses > 0 {
		r.MissRate = float64(misses) / float64(accesses)
	}
	return r
}

func uniq(in []int) []int {
	seen := map[int]bool{}
	var out []int
	for _, v := range in {
		if seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out
}

type lruSet struct {
	cap int
	ord []int
	has map[int]bool
}

func newLRU(cap int) *lruSet {
	return &lruSet{cap: cap, has: map[int]bool{}}
}

func (l *lruSet) touch(e int) (hit bool) {
	if l.has[e] {
		for i, x := range l.ord {
			if x == e {
				l.ord = append(l.ord[:i], l.ord[i+1:]...)
				break
			}
		}
		l.ord = append(l.ord, e)
		return true
	}
	if len(l.ord) >= l.cap {
		vic := l.ord[0]
		l.ord = l.ord[1:]
		delete(l.has, vic)
	}
	l.ord = append(l.ord, e)
	l.has[e] = true
	return false
}

// SweepMissRates replays one trace at several slot fractions of nExperts.
func SweepMissRates(policy CachePolicy, nExperts int, fractions []float64, prefill, decode []TraceStep) []CacheSimResult {
	out := make([]CacheSimResult, 0, len(fractions))
	for _, f := range fractions {
		slots := int(float64(nExperts)*f + 0.5)
		if slots < 1 {
			slots = 1
		}
		if slots > nExperts {
			slots = nExperts
		}
		out = append(out, SimulateCache(policy, nExperts, slots, prefill, decode))
	}
	return out
}
