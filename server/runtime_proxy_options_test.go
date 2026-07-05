package server

import "testing"

func TestNumPredictFromOptions(t *testing.T) {
	t.Parallel()

	if _, ok := numPredictFromOptions(nil); ok {
		t.Fatal("expected no limit when options nil")
	}

	n, ok := numPredictFromOptions(map[string]any{"num_predict": 256})
	if !ok || n != 256 {
		t.Fatalf("got (%d, %v), want (256, true)", n, ok)
	}

	if _, ok := numPredictFromOptions(map[string]any{"num_predict": -1}); ok {
		t.Fatal("expected no limit for num_predict -1")
	}

	if _, ok := numPredictFromOptions(map[string]any{}); ok {
		t.Fatal("expected no limit for empty options")
	}
}

func TestRuntimeOptionsWithNumPredict(t *testing.T) {
	t.Parallel()

	if got := runtimeOptionsWithNumPredict(0, false); len(got) != 0 {
		t.Fatalf("unlimited: got %v", got)
	}
	if got := runtimeOptionsWithNumPredict(128, true); got["num_predict"] != 128 {
		t.Fatalf("limited: got %v", got)
	}
}

func TestProxyOptsFromV1Body(t *testing.T) {
	t.Setenv("ZEROLLAMA_AGENT_CACHE_RUNTIME", "1")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")

	if got := proxyOptsFromV1Body(nil); got != nil {
		t.Fatalf("nil body: got %v", got)
	}

	top := map[string]any{"prompt_cache_key": "hermes:sess-1"}
	if got := proxyOptsFromV1Body(top); got["prompt_cache_key"] != "hermes:sess-1" {
		t.Fatalf("top-level only: got %v", got)
	}

	nested := map[string]any{
		"prompt_cache_key": "hermes:sess-2",
		"options": map[string]any{
			"num_ctx":          131072,
			"prompt_cache_key": "hermes:sess-2",
			"eliza": map[string]any{
				"conversationId": "sess-2",
			},
		},
	}
	got := proxyOptsFromV1Body(nested)
	if got["num_ctx"] != 131072 {
		t.Fatalf("options map: num_ctx = %v", got["num_ctx"])
	}
	if got["prompt_cache_key"] != "hermes:sess-2" {
		t.Fatalf("options map: prompt_cache_key = %v", got["prompt_cache_key"])
	}
	if !agentCachePrefersRuntime(got) {
		t.Fatal("expected agent runtime route for Hermes-shaped v1 body")
	}

	mergeTop := map[string]any{
		"prompt_cache_key": "hermes:sess-3",
		"options":          map[string]any{"num_ctx": 131072},
	}
	merged := proxyOptsFromV1Body(mergeTop)
	if merged["prompt_cache_key"] != "hermes:sess-3" {
		t.Fatalf("merge top-level key: got %v", merged)
	}
	if merged["num_ctx"] != 131072 {
		t.Fatalf("merge preserved num_ctx: got %v", merged)
	}
}
