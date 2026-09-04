package openai

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// BindChatCompletionRequest unmarshals an OpenAI chat body and merges nested
// extra_body fields.
//
// WHY multiple wire shapes: some SDKs nest zerollama under extra_body; the
// OpenAI Python SDK flattens extra_body onto the HTTP root (flat qos_class /
// project_name / zerollama). All must fold into options.zerollama rather than 400.
// Precedence (strongest → weakest): options.zerollama → top-level zerollama
// object → flat aliases. See docs/openai-harness-qos-wire-shapes.md.
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
	// Precedence (strongest → weakest): options.zerollama → top-level zerollama
	// object → flat aliases. Fold flat first; underlay top-level zerollama so
	// nested options.zerollama already on req wins on conflict.
	req.Options = foldFlatZerollamaRaw(req.Options, raw)
	if zRaw, ok := raw["zerollama"]; ok {
		var z map[string]any
		if json.Unmarshal(zRaw, &z) == nil && len(z) > 0 {
			req.Options = underlayZerollamaOptions(req.Options, z)
		}
	}
	if eb, ok := raw["extra_body"]; ok {
		mergeChatExtraBody(&req, eb)
	}
	mergeOptionsPromptCacheKey(&req)
	if err := validateChatTemplateKwargs(req.ChatTemplateKwargs); err != nil {
		return req, err
	}
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
		} else if s := rawString(flat["session_id"]); s != "" {
			req.SessionID = &s
		}
	}
	if req.SessionID == nil {
		if s := rawString(flat["session_id"]); s != "" {
			req.SessionID = &s
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
	if req.Timeout == nil {
		if t := rawKeepAlive(flat["timeout"]); t != nil {
			req.Timeout = t
		}
	}
	if len(req.Format) == 0 {
		if f, ok := flat["format"]; ok && len(f) > 0 {
			req.Format = append(json.RawMessage(nil), f...)
		}
	}
	if req.Compression == nil {
		if raw, ok := flat["compression"]; ok && len(raw) > 0 {
			var cfg api.ChatCompressionConfig
			if json.Unmarshal(raw, &cfg) == nil {
				req.Compression = &cfg
			}
		}
	}
	if req.StreamOptions == nil {
		if raw, ok := flat["stream_options"]; ok && len(raw) > 0 {
			var so StreamOptions
			if json.Unmarshal(raw, &so) == nil {
				req.StreamOptions = &so
			}
		}
	}
	overlaySamplingFromRaw(&samplingOverlay{
		Temperature:         &req.Temperature,
		TopP:                &req.TopP,
		MinP:                &req.MinP,
		TypicalP:            &req.TypicalP,
		FrequencyPenalty:    &req.FrequencyPenalty,
		PresencePenalty:     &req.PresencePenalty,
		RepetitionPenalty:   &req.RepetitionPenalty,
		RepeatPenalty:       &req.RepeatPenalty,
		TopK:                &req.TopK,
		Seed:                &req.Seed,
		MaxTokens:           &req.MaxTokens,
		MaxCompletionTokens: &req.MaxCompletionTokens,
		EnablePLD:           &req.EnablePLD,
		EnableMTP:           &req.EnableMTP,
		EnableDrafter:       &req.EnableDrafter,
	}, flat)
	overlayInt(&req.N, flat["n"])
	overlayInt(&req.ReasoningBudgetTokens, flat["reasoning_budget_tokens"])
	if req.ServiceTier == "" {
		if s := rawString(flat["service_tier"]); s != "" {
			req.ServiceTier = s
		}
	}
	overlayBool(&req.EnableThinking, flat["enable_thinking"])
	overlayBool(&req.ParallelToolCalls, flat["parallel_tool_calls"])
	overlayBool(&req.Store, flat["store"])
	overlayLogitBias(&req.LogitBias, flat["logit_bias"])
	if len(req.Tools) == 0 {
		if raw, ok := flat["tools"]; ok && len(raw) > 0 {
			var tools []api.Tool
			if json.Unmarshal(raw, &tools) == nil {
				req.Tools = tools
			}
		}
	}
	if len(req.Functions) == 0 {
		if raw, ok := flat["functions"]; ok && len(raw) > 0 {
			var fns []api.ToolFunction
			if json.Unmarshal(raw, &fns) == nil {
				req.Functions = fns
			}
		}
	}
	if req.ToolChoice == nil {
		if raw, ok := flat["tool_choice"]; ok && len(raw) > 0 {
			var v any
			if json.Unmarshal(raw, &v) == nil {
				req.ToolChoice = v
			}
		}
	}
	if req.FunctionCall == nil {
		if raw, ok := flat["function_call"]; ok && len(raw) > 0 {
			var v any
			if json.Unmarshal(raw, &v) == nil {
				req.FunctionCall = v
			}
		}
	}
	if !req.ContinueFinalMessage {
		if b, ok := rawBool(flat["continue_final_message"]); ok {
			req.ContinueFinalMessage = b
		}
	}
	// Flat harness keys under nested extra_body (SDK did not flatten).
	// Weaker than extra_body.options / extra_body.zerollama below.
	req.Options = foldFlatZerollamaRaw(req.Options, flat)
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

// foldFlatZerollamaRaw lifts allowlisted flat harness keys into options.zerollama.
// WHY existing options.zerollama wins: nested/explicit is the Ollama contract;
// flat keys are SDK-flattened aliases and must not overwrite careful clients.
func foldFlatZerollamaRaw(opts map[string]any, raw map[string]json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return opts
	}
	flat := make(map[string]any, len(chatCompletionZerollamaFlatFields))
	for _, name := range chatCompletionZerollamaFlatFields {
		v, ok := raw[name]
		if !ok || len(v) == 0 {
			continue
		}
		var anyVal any
		if json.Unmarshal(v, &anyVal) != nil {
			continue
		}
		flat[name] = anyVal
	}
	return foldFlatZerollamaMap(opts, flat)
}

// FoldFlatZerollamaMap lifts flat harness keys from a decoded JSON object into
// options.zerollama. Exported for the runtime v1 proxy path.
// WHY exported: proxyOptsFromV1Body / runtimeV1ProxyOptions must apply the same
// fold the OpenAI bind path uses, or Python never sees qos_class after a 400 fix.
func FoldFlatZerollamaMap(opts map[string]any, src map[string]any) map[string]any {
	return foldFlatZerollamaMap(opts, src)
}

func foldFlatZerollamaMap(opts map[string]any, src map[string]any) map[string]any {
	if len(src) == 0 {
		return opts
	}
	flat := make(map[string]any, len(chatCompletionZerollamaFlatFields))
	for _, name := range chatCompletionZerollamaFlatFields {
		v, ok := src[name]
		if !ok || v == nil {
			continue
		}
		flat[name] = v
	}
	if len(flat) == 0 {
		return opts
	}
	if opts == nil {
		opts = map[string]any{}
	}
	existing, _ := opts["zerollama"].(map[string]any)
	// existing (nested) overlays flat so options.zerollama wins.
	opts["zerollama"] = mergeOptionsMaps(flat, existing)
	return opts
}

func mergeZerollamaOptions(opts map[string]any, z map[string]any) map[string]any {
	if opts == nil {
		opts = map[string]any{}
	}
	existing, _ := opts["zerollama"].(map[string]any)
	opts["zerollama"] = mergeOptionsMaps(existing, z)
	return opts
}

// underlayZerollamaOptions merges z beneath existing options.zerollama (nested wins).
// WHY underlay (not overlay): top-level "zerollama" is usually SDK-flattened
// extra_body.zerollama; options.zerollama is the explicit nested contract.
func underlayZerollamaOptions(opts map[string]any, z map[string]any) map[string]any {
	if opts == nil {
		opts = map[string]any{}
	}
	existing, _ := opts["zerollama"].(map[string]any)
	opts["zerollama"] = mergeOptionsMaps(z, existing)
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
		} else if s, ok := req.Options["session_id"].(string); ok && strings.TrimSpace(s) != "" {
			key := strings.TrimSpace(s)
			req.SessionID = &key
		}
	}
	if req.SessionID == nil {
		if s, ok := req.Options["session_id"].(string); ok && strings.TrimSpace(s) != "" {
			key := strings.TrimSpace(s)
			req.SessionID = &key
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

func BindCompletionRequest(body []byte) (CompletionRequest, error) {
	var req CompletionRequest
	if len(body) == 0 {
		return req, nil
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return req, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return req, err
	}
	if eb, ok := raw["extra_body"]; ok {
		mergeCompletionExtraBody(&req, eb)
	}
	return req, nil
}

func mergeCompletionExtraBody(req *CompletionRequest, extra json.RawMessage) {
	var flat map[string]json.RawMessage
	if json.Unmarshal(extra, &flat) != nil {
		return
	}
	overlayF32(&req.Temperature, flat["temperature"])
	overlayF32(&req.TopP, flat["top_p"])
	overlayF32(&req.MinP, flat["min_p"])
	overlayF32(&req.TypicalP, flat["typical_p"])
	overlayF32(&req.FrequencyPenalty, flat["frequency_penalty"])
	overlayF32(&req.PresencePenalty, flat["presence_penalty"])
	overlayF32(&req.RepetitionPenalty, flat["repetition_penalty"])
	overlayF32(&req.RepeatPenalty, flat["repeat_penalty"])
	overlayInt(&req.TopK, flat["top_k"])
	overlayInt(&req.Seed, flat["seed"])
	overlayInt(&req.MaxTokens, flat["max_tokens"])
	overlayInt(&req.N, flat["n"])
	overlayInt(&req.BestOf, flat["best_of"])
	if req.ServiceTier == "" {
		if s := rawString(flat["service_tier"]); s != "" {
			req.ServiceTier = s
		}
	}
	overlayBool(&req.EnablePLD, flat["enable_pld"])
	overlayBool(&req.EnableMTP, flat["enable_mtp"])
	overlayBool(&req.EnableDrafter, flat["enable_drafter"])
	overlayLogitBias(&req.LogitBias, flat["logit_bias"])
	if !req.Echo {
		if b, ok := rawBool(flat["echo"]); ok {
			req.Echo = b
		}
	}
}
