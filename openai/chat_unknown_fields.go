package openai

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
)

// chatCompletionPassthroughFields are recognized OpenAI / Ollama request keys that
// may not appear on ChatCompletionRequest but must not 400 (SDKs send them).
// enable_thinking / chat_template_kwargs live on ChatCompletionRequest and are
// validated/mapped in chat_thinking_aliases.go (unknown nested kwargs → 400).
//
// WHY harness QoS keys (qos_class, project_*, zerollama, …) are allowlisted:
// the OpenAI Python SDK promotes extra_body onto the HTTP JSON root. Without
// these names, trap 77 returns 400 for legitimate Hermes aux traffic while
// nested options.zerollama on /api/chat works. Bind folds them into
// options.zerollama (see chat_extras.go) — allowlist alone would accept-and-drop.
var chatCompletionPassthroughFields = []string{
	"extra_body",
	"n",
	"user",
	"logit_bias",
	"parallel_tool_calls",
	"max_completion_tokens",
	"functions",
	"function_call",
	"modalities",
	"audio",
	"metadata",
	"service_tier",
	"store",
	"prediction",
	"web_search_options",
	"format", // now also on ChatCompletionRequest; keep for older clients
	// Top-level object after SDK flattens extra_body.zerollama.
	"zerollama",
	// Flat harness aliases (SDK-flattened extra_body.{qos_class,project_name,…}).
	"qos_class",
	"qos_priority",
	"project_id",
	"project_name",
	"client_id",
	"client_name",
	"project",
	"session_group",
	"harness",
	"session_parent",
	"cache_scope",
	"cache_level",
	"cache_tier",
	"cache_reset",
	"fulfillment",
	"fulfill_mode",
	"priority_mode",
}

// chatCompletionZerollamaFlatFields are top-level / extra_body keys folded into
// options.zerollama. The zerollama object itself is handled separately.
// WHY a second list (not only passthrough): passthrough stops the 400; this list
// drives foldFlatZerollamaRaw / FoldFlatZerollamaMap. Keep them in sync when
// adding a harness alias or flat keys will 400 again (or fold will miss them).
var chatCompletionZerollamaFlatFields = []string{
	"qos_class",
	"qos_priority",
	"project_id",
	"project_name",
	"client_id",
	"client_name",
	"project",
	"session_group",
	"harness",
	"session_parent",
	"cache_scope",
	"cache_level",
	"cache_tier",
	"cache_reset",
	"fulfillment",
	"fulfill_mode",
	"priority_mode",
}

var (
	chatCompletionKnownFieldsOnce sync.Once
	chatCompletionKnownFields     map[string]struct{}
)

func knownChatCompletionTopLevelFields() map[string]struct{} {
	chatCompletionKnownFieldsOnce.Do(func() {
		known := make(map[string]struct{}, 48)
		t := reflect.TypeOf(ChatCompletionRequest{})
		for i := 0; i < t.NumField(); i++ {
			tag := t.Field(i).Tag.Get("json")
			if tag == "" || tag == "-" {
				continue
			}
			name, _, _ := strings.Cut(tag, ",")
			if name == "" || name == "-" {
				continue
			}
			known[name] = struct{}{}
		}
		for _, name := range chatCompletionPassthroughFields {
			known[name] = struct{}{}
		}
		chatCompletionKnownFields = known
	})
	return chatCompletionKnownFields
}

// CheckUnknownChatCompletionFields returns an error when the body contains
// top-level keys outside the known request surface (minefield trap 77).
// WHY 400 here: typos and wrong harness knobs must be loud instead of silently
// measuring the lane default (fail-open would poison QoS / latency SLOs).
func CheckUnknownChatCompletionFields(body []byte) error {
	if len(body) == 0 {
		return nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return err
	}
	return rejectUnknownChatCompletionFields(raw)
}

func rejectUnknownChatCompletionFields(raw map[string]json.RawMessage) error {
	known := knownChatCompletionTopLevelFields()
	var unknown []string
	for k := range raw {
		if _, ok := known[k]; !ok {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("unknown field: %s", strings.Join(unknown, ", "))
}
