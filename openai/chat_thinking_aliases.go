package openai

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
)

// Known nested keys under chat_template_kwargs. Anything else → HTTP 400 (trap 07).
var knownChatTemplateKwargs = map[string]struct{}{
	"enable_thinking":  {},
	"reasoning_effort": {},
}

func validateChatTemplateKwargs(kwargs map[string]any) error {
	if len(kwargs) == 0 {
		return nil
	}
	var unknown []string
	for k := range kwargs {
		if _, ok := knownChatTemplateKwargs[k]; !ok {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("unknown field: chat_template_kwargs.%s", strings.Join(unknown, ", "))
}

// thinkFromEnableThinkingAliases maps vLLM/SGLang-style thinking knobs onto Think.
// Call only when Reasoning / reasoning_effort did not already set think.
func thinkFromEnableThinkingAliases(enableThinking *bool, kwargs map[string]any) (*api.ThinkValue, error) {
	if err := validateChatTemplateKwargs(kwargs); err != nil {
		return nil, err
	}
	if enableThinking != nil {
		return &api.ThinkValue{Value: *enableThinking}, nil
	}
	if kwargs == nil {
		return nil, nil
	}
	if v, ok := kwargs["enable_thinking"]; ok {
		b, ok := v.(bool)
		if !ok {
			return nil, fmt.Errorf("invalid chat_template_kwargs.enable_thinking: must be boolean")
		}
		return &api.ThinkValue{Value: b}, nil
	}
	if v, ok := kwargs["reasoning_effort"]; ok {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("invalid chat_template_kwargs.reasoning_effort: must be string")
		}
		effort := strings.TrimSpace(s)
		if !slices.Contains([]string{"high", "medium", "low", "none"}, effort) {
			return nil, fmt.Errorf("invalid reasoning value: '%s' (must be \"high\", \"medium\", \"low\", or \"none\")", effort)
		}
		if effort == "none" {
			return &api.ThinkValue{Value: false}, nil
		}
		return &api.ThinkValue{Value: effort}, nil
	}
	return nil, nil
}
