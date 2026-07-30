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

func TestBindChatCompletionRequest_TopLevelFlatQoS(t *testing.T) {
	// OpenAI SDK flattens extra_body.{qos_class,project_*} onto the HTTP root.
	body := []byte(`{
		"model": "gemma4:26b-optiq",
		"messages": [{"role":"user","content":"hi"}],
		"qos_class": "auxiliary",
		"project_id": "hermes-lean",
		"project_name": "discord:dm:123"
	}`)
	req, err := BindChatCompletionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	z, ok := req.Options["zerollama"].(map[string]any)
	if !ok {
		t.Fatalf("options=%v", req.Options)
	}
	if z["qos_class"] != "auxiliary" || z["project_id"] != "hermes-lean" || z["project_name"] != "discord:dm:123" {
		t.Fatalf("zerollama=%v", z)
	}
}

func TestBindChatCompletionRequest_TopLevelZerollamaObject(t *testing.T) {
	// SDK flattens extra_body.zerollama → top-level "zerollama".
	body := []byte(`{
		"model": "gemma4:26b-optiq",
		"messages": [{"role":"user","content":"hi"}],
		"zerollama": {
			"qos_class": "interactive",
			"project_id": "hermes-lean"
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
	if z["qos_class"] != "interactive" || z["project_id"] != "hermes-lean" {
		t.Fatalf("zerollama=%v", z)
	}
}

func TestBindChatCompletionRequest_FlatExtraBodyQoS(t *testing.T) {
	// Nested extra_body without SDK flatten — flat keys must still lift.
	body := []byte(`{
		"model": "gemma4:26b-optiq",
		"messages": [{"role":"user","content":"hi"}],
		"extra_body": {
			"qos_class": "background",
			"project_name": "batch"
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
	if z["qos_class"] != "background" || z["project_name"] != "batch" {
		t.Fatalf("zerollama=%v", z)
	}
}

func TestBindChatCompletionRequest_NestedZerollamaWinsOverFlat(t *testing.T) {
	body := []byte(`{
		"model": "gemma4:26b-optiq",
		"messages": [{"role":"user","content":"hi"}],
		"qos_class": "background",
		"options": {
			"zerollama": {"qos_class": "interactive", "project_id": "main"}
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
	if z["qos_class"] != "interactive" || z["project_id"] != "main" {
		t.Fatalf("nested must win over flat: zerollama=%v", z)
	}
}

func TestBindChatCompletionRequest_NestedZerollamaWinsOverTopLevelObject(t *testing.T) {
	body := []byte(`{
		"model": "gemma4:26b-optiq",
		"messages": [{"role":"user","content":"hi"}],
		"zerollama": {"qos_class": "background", "project_name": "from-top"},
		"options": {
			"zerollama": {"qos_class": "interactive", "project_id": "main"}
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
	if z["qos_class"] != "interactive" {
		t.Fatalf("options.zerollama must win over top-level zerollama: %v", z)
	}
	if z["project_id"] != "main" {
		t.Fatalf("project_id=%v", z["project_id"])
	}
	if z["project_name"] != "from-top" {
		t.Fatalf("top-level-only keys should underlay: %v", z)
	}
}
