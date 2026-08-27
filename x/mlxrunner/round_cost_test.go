package mlxrunner

import (
	"path/filepath"
	"testing"
)

func TestRoundCostSaveRestore(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", filepath.Join(dir, "models"))

	c := newDepthController()
	c.scheduled = 4
	c.probeInterval = 16
	c.cost.observe(0, 10_000_000)
	c.cost.observe(3, 18_000_000)
	c.acc.observe(3, 2)
	for range 12 {
		c.acc.observe(3, 2)
	}
	c.saveRoundCost("library/gemma4:e4b")

	got := newDepthController()
	got.restoreRoundCost("library/gemma4:e4b")
	if got.scheduled != 4 {
		t.Fatalf("scheduled=%d want 4", got.scheduled)
	}
	if got.probeInterval != 16 {
		t.Fatalf("probeInterval=%d want 16", got.probeInterval)
	}
	if !got.cost.sampled(0) || !got.cost.sampled(3) {
		t.Fatalf("cost depths %v missing 0 and 3", got.cost.depths)
	}
	if got.acc.frontier() < 1 {
		t.Fatalf("acceptance frontier=%d, want restored samples", got.acc.frontier())
	}
}

func TestRoundCostPathSanitizes(t *testing.T) {
	p := roundCostPath("host/ns/name:tag")
	if p == "" || filepath.Ext(p) != ".json" {
		t.Fatalf("path %q", p)
	}
	if roundCostPath("   ") != "" {
		t.Fatal("empty name")
	}
}

func TestRoundCostCtxBucketsIsolated(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", filepath.Join(dir, "models"))

	c := newDepthController()
	c.applyCtxBucket(100)
	c.scheduled = 5
	c.cost.observe(0, 10_000_000)
	c.cost.observe(4, 16_000_000)
	c.saveRoundCost("m")

	c.applyCtxBucket(9000)
	if c.cost.sampled(4) {
		t.Fatal("8k bucket must not inherit 2k cost samples")
	}
	c.scheduled = 2
	c.cost.observe(0, 20_000_000)
	c.cost.observe(2, 40_000_000)
	c.saveRoundCost("m")

	got := newDepthController()
	got.restoreRoundCost("m")
	got.applyCtxBucket(50)
	if got.scheduled != 5 {
		t.Fatalf("2k scheduled=%d want 5", got.scheduled)
	}
	got.applyCtxBucket(9000)
	if got.scheduled != 2 {
		t.Fatalf("8k scheduled=%d want 2", got.scheduled)
	}
}

func TestCtxBucket(t *testing.T) {
	if ctxBucket(1) != 2048 || ctxBucket(4097) != 8192 || ctxBucket(20000) != 32768 {
		t.Fatalf("%d %d %d", ctxBucket(1), ctxBucket(4097), ctxBucket(20000))
	}
}
