package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestParseFulfillmentModes(t *testing.T) {
	cases := map[string]fulfillmentMode{
		"complete":  fulfillmentComplete,
		"guarantee": fulfillmentComplete,
		"reliable":  fulfillmentComplete,
		"benchmark": fulfillmentBenchmark,
		"bench":     fulfillmentBenchmark,
		"speed":     fulfillmentBenchmark,
		"exclusive": fulfillmentBenchmark,
	}
	for name, want := range cases {
		got, ok := parseFulfillmentName(name)
		if !ok || got != want {
			t.Fatalf("%q -> %v ok=%v want %v", name, got, ok, want)
		}
	}
}

func TestMLXQoSFulfillmentForcesInteractive(t *testing.T) {
	q := mlxQoSFromOptions(map[string]any{
		"zerollama": map[string]any{
			"qos_class":   "background",
			"fulfillment": "benchmark",
		},
	})
	if q.Fulfillment != fulfillmentBenchmark {
		t.Fatalf("fulfillment=%v", q.Fulfillment)
	}
	if q.Class != mlxClassInteractive || !q.Explicit {
		t.Fatalf("class=%s explicit=%v", q.Class, q.Explicit)
	}
}

func TestEnsureFulfillmentSessionKey(t *testing.T) {
	opts := map[string]any{
		"zerollama": map[string]any{
			"fulfillment": "complete",
			"project_id":  "bench-suite",
		},
	}
	qos := mlxQoSFromOptions(opts)
	key := ensureFulfillmentSessionKey(opts, qos)
	want := "fulfill:complete:bench-suite"
	if key != want {
		t.Fatalf("key=%q want %q", key, want)
	}
	if opts["prompt_cache_key"] != want {
		t.Fatalf("opts not injected: %v", opts["prompt_cache_key"])
	}
}

func TestFulfillmentKeepAliveFloor(t *testing.T) {
	qos := mlxQoS{Fulfillment: fulfillmentBenchmark}
	got := fulfillmentKeepAliveFloor(qos, nil)
	if got == nil || got.Duration < fulfillBenchmarkKeepAliveFloor {
		t.Fatalf("benchmark floor = %v", got)
	}
	explicit := &api.Duration{Duration: time.Minute}
	if fulfillmentKeepAliveFloor(qos, explicit) != explicit {
		t.Fatal("explicit keep_alive should win")
	}
}

func TestFulfillmentExclusiveDefer(t *testing.T) {
	g := newMLXAgentGate()
	release := g.beginFulfillment("model:a", "fulfill:benchmark:x", fulfillmentBenchmark)
	defer release()

	deferNow, policy := g.shouldDeferFulfillment("model:b", "other", mlxClassInteractive)
	if !deferNow || policy != "fulfillment_exclusive" {
		t.Fatalf("defer=%v policy=%s", deferNow, policy)
	}
	deferNow, policy = g.shouldDeferFulfillment("model:a", "fulfill:benchmark:x", mlxClassInteractive)
	if deferNow {
		t.Fatalf("same hold should not defer, policy=%s", policy)
	}
}

func TestFulfillmentCompleteSameModelOnly(t *testing.T) {
	g := newMLXAgentGate()
	release := g.beginFulfillment("model:a", "fulfill:complete:x", fulfillmentComplete)
	defer release()

	deferNow, policy := g.shouldDeferFulfillment("model:a", "other", mlxClassInteractive)
	if !deferNow || policy != "fulfillment_complete_same_model" {
		t.Fatalf("same model: defer=%v policy=%s", deferNow, policy)
	}
	deferNow, _ = g.shouldDeferFulfillment("model:b", "peer", mlxClassInteractive)
	if deferNow {
		t.Fatal("complete should allow interactive on other models")
	}
	deferNow, policy = g.shouldDeferFulfillment("model:b", "bg", mlxClassBackground)
	if !deferNow || policy != "fulfillment_complete_lower" {
		t.Fatalf("lower global: defer=%v policy=%s", deferNow, policy)
	}
}

func TestWaitForFulfillmentExclusiveIdle(t *testing.T) {
	g := newMLXAgentGate()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := g.waitForFulfillment(ctx, "m1", "fulfill:benchmark:t", fulfillmentBenchmark); err != nil {
		t.Fatal(err)
	}
}

func TestProtectedModelKeys(t *testing.T) {
	g := newMLXAgentGate()
	release := g.beginFulfillment("digest:protected", "fulfill:complete:1", fulfillmentComplete)
	defer release()
	keys := g.protectedModelKeys()
	if _, ok := keys["digest:protected"]; !ok {
		t.Fatalf("expected protected key, got %#v", keys)
	}
}
