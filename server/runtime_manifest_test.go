package server

import "testing"

func TestRuntimeProxyOptionsMergesGGUF(t *testing.T) {
	opts := runtimeProxyOptions("nonexistent-model-xyz", 64, true, nil)
	if _, ok := opts["gguf"]; ok {
		t.Fatal("unexpected gguf for missing model")
	}
	if opts["num_predict"] != 64 {
		t.Fatalf("num_predict=%v", opts["num_predict"])
	}

	opts = runtimeOptionsWithNumPredict(0, false)
	opts["gguf"] = "/tmp/test.gguf"
	if v, ok := opts["gguf"]; !ok || v != "/tmp/test.gguf" {
		t.Fatalf("gguf=%v", opts["gguf"])
	}
}

func TestRuntimeProxyOptionsPreservesClientGguf(t *testing.T) {
	client := map[string]any{"gguf": "/override/model.gguf"}
	opts := runtimeProxyOptions("missing-model", 64, true, client)
	if opts["gguf"] != "/override/model.gguf" {
		t.Fatalf("gguf=%v", opts["gguf"])
	}
}

func TestRuntimeV1ProxyOptionsMaxTokensOnly(t *testing.T) {
	opts := runtimeV1ProxyOptions("missing-model", map[string]any{
		"max_tokens": float64(32),
		"messages":   []any{},
	})
	if opts["num_predict"] != 32 {
		t.Fatalf("num_predict=%v", opts["num_predict"])
	}
}

func TestRuntimeV1ProxyOptionsTopLevelNumCtx(t *testing.T) {
	opts := runtimeV1ProxyOptions("missing-model", map[string]any{
		"num_ctx": float64(6144),
		"messages": []any{},
	})
	if opts["num_ctx"] != float64(6144) {
		t.Fatalf("num_ctx=%v", opts["num_ctx"])
	}
}

func TestRuntimeV1ProxyOptionsManifestGGUF(t *testing.T) {
	for _, name := range []string{"llama3.2", "llama3.2:latest", "llama3:latest"} {
		path, ok := runtimeGGUFPath(name)
		if !ok || path == "" {
			continue
		}
		opts := runtimeV1ProxyOptions(name, map[string]any{"max_tokens": float64(16)})
		g, ok := opts["gguf"].(string)
		if !ok || g != path {
			t.Fatalf("model %q gguf=%v want %q", name, opts["gguf"], path)
		}
		return
	}
	t.Skip("no pulled GGUF model in test environment")
}

func TestRuntimeProxyOptionsPreservesClientNumCtx(t *testing.T) {
	client := map[string]any{"num_ctx": 8192, "temperature": 0.2}
	opts := runtimeProxyOptions("missing-model", 128, true, client)
	if opts["num_ctx"] != 8192 {
		t.Fatalf("num_ctx=%v", opts["num_ctx"])
	}
	if opts["temperature"] != 0.2 {
		t.Fatalf("temperature=%v", opts["temperature"])
	}
	if opts["num_predict"] != 128 {
		t.Fatalf("num_predict=%v", opts["num_predict"])
	}
}
