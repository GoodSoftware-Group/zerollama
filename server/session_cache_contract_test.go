package server

import (
	"strings"
	"testing"
	"time"
)

func TestMLXQoSCacheLevelAndReset(t *testing.T) {
	q := mlxQoSFromOptions(map[string]any{
		"zerollama": map[string]any{
			"cache_level": "dram",
			"cache_reset": true,
		},
	})
	if q.CacheLevel != qosCacheLevelDRAM || !q.CacheReset {
		t.Fatalf("got level=%q reset=%v", q.CacheLevel, q.CacheReset)
	}
	if !q.ForbidsDiskPersist() || q.WantsDiskPersist() {
		t.Fatal("dram should forbid disk")
	}

	q = mlxQoSFromOptions(map[string]any{
		"zerollama": map[string]any{"cache_level": "ssd"},
	})
	if q.CacheLevel != qosCacheLevelDisk || !q.WantsDiskPersist() {
		t.Fatalf("ssd alias -> %q", q.CacheLevel)
	}

	q = mlxQoSFromOptions(nil)
	if q.CacheLevel != qosCacheLevelAuto || q.CacheReset {
		t.Fatalf("defaults level=%q reset=%v", q.CacheLevel, q.CacheReset)
	}
}

func TestWaitParentMultiplex(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:multi"
	parent := "hermes:agent:parent:1"
	other := "hermes:agent:other:2"
	child := "hermes:spawn:child:3"

	// Parent claimed, then primary switches to another agent — parent stays in key hot-map.
	releaseParent := g.begin(modelKey, parent, mlxClassInteractive, mlxQoS{})
	releaseOther := g.begin(modelKey, other, mlxClassInteractive, mlxQoS{})
	releaseParent() // parent in cooldown, other is primary holder

	now := time.Now()
	defer_, policy, blocked := g.shouldDefer(modelKey, child, parent, mlxClassAuxiliary, mlxQoS{}, now)
	if !defer_ || policy != "wait_parent" {
		t.Fatalf("defer=%v policy=%q want wait_parent", defer_, policy)
	}
	if blocked != parent {
		t.Fatalf("blocked_by=%q want %q", blocked, parent)
	}

	// Unrelated child without parent should use normal primary defer (other), not wait_parent.
	defer2, policy2, _ := g.shouldDefer(modelKey, "hermes:spawn:x", "", mlxClassAuxiliary, mlxQoS{}, now)
	if !defer2 || policy2 == "wait_parent" {
		t.Fatalf("unrelated spawn policy=%q", policy2)
	}

	releaseOther()

	// Expire parent key cooldown deterministically.
	g.mu.Lock()
	if e := g.keyHot[modelKey][parent]; e != nil {
		e.hotUntil = now.Add(-time.Second)
		e.inflight = 0
	}
	if e := g.keyHot[modelKey][other]; e != nil {
		e.hotUntil = now.Add(-time.Second)
		e.inflight = 0
	}
	g.refreshPrimaryFromKeyHotLocked(modelKey, now)
	g.mu.Unlock()

	if defer_, policy, _ := g.shouldDefer(modelKey, child, parent, mlxClassAuxiliary, mlxQoS{}, now); defer_ {
		t.Fatalf("should not wait after parent cooled, policy=%q", policy)
	}
}

func TestBeginEndOverwriteNoInflightLeak(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:race"

	releaseA := g.begin(modelKey, "key:a", mlxClassInteractive, mlxQoS{})
	releaseB := g.begin(modelKey, "key:b", mlxClassInteractive, mlxQoS{})
	releaseA() // previously leaked when primary was overwritten to key:b
	releaseB()

	now := time.Now()
	g.mu.Lock()
	slot := g.slots[modelKey]
	a := g.keyHot[modelKey]["key:a"]
	b := g.keyHot[modelKey]["key:b"]
	primaryInflight := 0
	if slot != nil {
		primaryInflight = slot.inflight
	}
	aInflight, bInflight := -1, -1
	if a != nil {
		aInflight = a.inflight
	}
	if b != nil {
		bInflight = b.inflight
	}
	g.mu.Unlock()

	if aInflight != 0 || bInflight != 0 {
		t.Fatalf("keyHot inflight leak a=%d b=%d", aInflight, bInflight)
	}
	if primaryInflight != 0 {
		t.Fatalf("primary inflight leak %d", primaryInflight)
	}
	// Both keys should still be hot (cooldown), so fairness still sees activity.
	if defer_, _, _ := g.shouldDefer(modelKey, "key:c", "", mlxClassAuxiliary, mlxQoS{}, now); !defer_ {
		t.Fatal("auxiliary should still defer during cooldown after clean end")
	}
}

func TestWaitParentNormalizedAuxBranch(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:m"
	rawParent := "hermes:20260721_120000_spawn"
	gateParent := injectMLXSessionKey(modelKey, rawParent, mlxClassAuxiliary, mlxQoS{})
	if gateParent == rawParent || !strings.HasPrefix(gateParent, "aux:") {
		t.Fatalf("expected aux rewrite, got %q from %q", gateParent, rawParent)
	}

	release := g.begin(modelKey, gateParent, mlxClassAuxiliary, mlxQoS{})
	defer release()

	now := time.Now()
	// Child sends the raw parent id; wait_parent must resolve to the rewritten gate key.
	defer_, policy, blocked := g.shouldDefer(modelKey, "hermes:spawn:child", rawParent, mlxClassAuxiliary, mlxQoS{}, now)
	if !defer_ || policy != "wait_parent" {
		t.Fatalf("defer=%v policy=%q want wait_parent", defer_, policy)
	}
	if blocked != gateParent {
		t.Fatalf("blocked_by=%q want gate key %q", blocked, gateParent)
	}
}

func TestActiveSessionsIncludesKeyHotMap(t *testing.T) {
	g := newMLXAgentGate()
	modelKey := "digest:ps"
	releaseA := g.begin(modelKey, "key:a", mlxClassInteractive, mlxQoS{
		ParentKey:  "parent:x",
		CacheLevel: qosCacheLevelDisk,
		ProjectID:  "harness",
	})
	releaseB := g.begin(modelKey, "key:b", mlxClassAuxiliary, mlxQoS{SessionGroup: "g1"})
	defer releaseA()
	defer releaseB()

	sessions := g.activeSessionsForModel(modelKey, time.Now())
	if len(sessions) < 2 {
		t.Fatalf("want >=2 sessions, got %d %+v", len(sessions), sessions)
	}
	foundParent := false
	for _, s := range sessions {
		if s.SessionKey == "key:a" && s.SessionParent == "parent:x" && s.CacheLevel == qosCacheLevelDisk {
			foundParent = true
		}
	}
	if !foundParent {
		t.Fatalf("missing enriched session: %+v", sessions)
	}
}

func TestZerollamaVersionAdvertisesCacheContract(t *testing.T) {
	caps := zerollamaVersionCapabilities()
	if caps["session_parent_defer"] != true || caps["cache_reset"] != true {
		t.Fatalf("caps=%v", caps)
	}
	levels, ok := caps["cache_level"].([]string)
	if !ok || len(levels) != 4 {
		t.Fatalf("cache_level=%v", caps["cache_level"])
	}
	qos := zerollamaVersionQoS()
	fields := qos["options"].(map[string]any)["fields"].(map[string]string)
	if fields["cache_reset"] == "" || fields["cache_level"] == "" {
		t.Fatalf("fields=%v", fields)
	}
}
