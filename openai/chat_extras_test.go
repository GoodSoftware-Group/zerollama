package openai

import (
	"testing"
)

func TestBindChatCompletionRequest_extraBody(t *testing.T) {
	body := []byte(`{
		"model": "gemma4:26b-optiq",
		"messages": [{"role":"user","content":"hi"}],
		"stream": true,
		"extra_body": {
			"prompt_cache_key": "hermes:agent:main:cli:1",
			"keep_alive": "30m",
			"enable_prefix_mm_cache": true,
			"options": {
				"num_ctx": 131072,
				"prompt_cache_key": "hermes:agent:main:cli:1"
			}
		}
	}`)
	req, err := BindChatCompletionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	if req.PromptCacheKey == nil || *req.PromptCacheKey != "hermes:agent:main:cli:1" {
		t.Fatalf("PromptCacheKey=%v", req.PromptCacheKey)
	}
	if req.KeepAlive == nil {
		t.Fatal("KeepAlive nil")
	}
	if req.EnablePrefixMMCache == nil || !*req.EnablePrefixMMCache {
		t.Fatalf("EnablePrefixMMCache=%v", req.EnablePrefixMMCache)
	}
	if req.Options["num_ctx"] != float64(131072) {
		t.Fatalf("num_ctx=%v", req.Options["num_ctx"])
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Options["prompt_cache_key"] != "hermes:agent:main:cli:1" {
		t.Fatalf("options prompt_cache_key=%v", out.Options["prompt_cache_key"])
	}
}

func TestBindChatCompletionRequest_extraBodyZerollamaQoS(t *testing.T) {
	body := []byte(`{
		"model": "gemma4:26b-optiq",
		"messages": [{"role":"user","content":"hi"}],
		"extra_body": {
			"zerollama": {
				"qos_class": "auxiliary",
				"session_group": "my-harness",
				"cache_scope": "shared"
			}
		}
	}`)
	req, err := BindChatCompletionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	z, ok := req.Options["zerollama"].(map[string]any)
	if !ok {
		t.Fatalf("options=%v", req.Options)
	}
	if z["qos_class"] != "auxiliary" || z["session_group"] != "my-harness" {
		t.Fatalf("zerollama=%v", z)
	}
}
