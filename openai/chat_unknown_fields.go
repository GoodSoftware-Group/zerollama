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
// Deliberately excludes enable_thinking / chat_template_kwargs — those are the
// silent-ignore arms in minefield trap 77; use think / reasoning_effort instead.
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
	"think",
	"format",
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
// A 400 here makes typos and wrong harness knobs loud instead of silently measuring
// the lane default.
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
