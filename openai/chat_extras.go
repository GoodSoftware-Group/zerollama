package openai

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// BindChatCompletionRequest unmarshals an OpenAI chat body and merges nested
// extra_body fields. Some SDKs nest zerollama keys under extra_body instead of
// promoting them to top-level JSON.
func BindChatCompletionRequest(body []byte) (ChatCompletionRequest, error) {
	var req ChatCompletionRequest
	if len(body) == 0 {
		return req, nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return req, err
	}
	// Trap 77: reject invented top-level keys so HTTP 200 is not silent fail-open.
	if err := rejectUnknownChatCompletionFields(raw); err != nil {
		return req, err
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return req, err
	}
	if eb, ok := raw["extra_body"]; ok {
		mergeChatExtraBody(&req, eb)
	}
	mergeOptionsPromptCacheKey(&req)
	return req, nil
}

func mergeChatExtraBody(req *ChatCompletionRequest, extra json.RawMessage) {
	var flat map[string]json.RawMessage
	if err := json.Unmarshal(extra, &flat); err != nil {
		return
	}
	if req.PromptCacheKey == nil {
		if s := rawString(flat["prompt_cache_key"]); s != "" {
			req.PromptCacheKey = &s
		}
	}
	if req.EnablePrefixMMCache == nil {
		if b, ok := rawBool(flat["enable_prefix_mm_cache"]); ok && b {
			req.EnablePrefixMMCache = &b
		}
	}
	if req.KeepAlive == nil {
		if ka := rawKeepAlive(flat["keep_alive"]); ka != nil {
			req.KeepAlive = ka
		}
	}
	if optsRaw, ok := flat["options"]; ok {
		var opts map[string]any
		if json.Unmarshal(optsRaw, &opts) == nil {
			req.Options = mergeOptionsMaps(req.Options, opts)
		}
	}
	if zRaw, ok := flat["zerollama"]; ok {
		var z map[string]any
		if json.Unmarshal(zRaw, &z) == nil {
			req.Options = mergeZerollamaOptions(req.Options, z)
		}
	}
}

func mergeZerollamaOptions(opts map[string]any, z map[string]any) map[string]any {
	if opts == nil {
		opts = map[string]any{}
	}
	existing, _ := opts["zerollama"].(map[string]any)
	opts["zerollama"] = mergeOptionsMaps(existing, z)
	return opts
}

func mergeOptionsPromptCacheKey(req *ChatCompletionRequest) {
	if req.Options == nil {
		return
	}
	if req.PromptCacheKey == nil {
		if s, ok := req.Options["prompt_cache_key"].(string); ok && strings.TrimSpace(s) != "" {
			key := strings.TrimSpace(s)
			req.PromptCacheKey = &key
		}
	}
}

func mergeOptionsMaps(base, overlay map[string]any) map[string]any {
	if base == nil {
		out := make(map[string]any, len(overlay))
		for k, v := range overlay {
			out[k] = v
		}
		return out
	}
	out := make(map[string]any, len(base)+len(overlay))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overlay {
		if k == "eliza" {
			if existing, ok := out[k].(map[string]any); ok {
				if eliza, ok := v.(map[string]any); ok {
					out[k] = mergeOptionsMaps(existing, eliza)
					continue
				}
			}
		}
		if k == "zerollama" {
			if existing, ok := out[k].(map[string]any); ok {
				if z, ok := v.(map[string]any); ok {
					out[k] = mergeOptionsMaps(existing, z)
					continue
				}
			}
		}
		out[k] = v
	}
	return out
}

func rawString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return ""
	}
	return strings.TrimSpace(s)
}

func rawBool(raw json.RawMessage) (bool, bool) {
	if len(raw) == 0 {
		return false, false
	}
	var b bool
	if json.Unmarshal(raw, &b) != nil {
		return false, false
	}
	return b, true
}

func rawKeepAlive(raw json.RawMessage) *api.Duration {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil && strings.TrimSpace(s) != "" {
		d, err := time.ParseDuration(strings.TrimSpace(s))
		if err == nil {
			return &api.Duration{Duration: d}
		}
	}
	var d api.Duration
	if json.Unmarshal(raw, &d) == nil && d.Duration > 0 {
		return &d
	}
	return nil
}
