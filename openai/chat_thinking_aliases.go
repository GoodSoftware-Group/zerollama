package openai

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// Known nested keys under chat_template_kwargs. Anything else → HTTP 400 (trap 07).
var knownChatTemplateKwargs = map[string]struct{}{
	"thinking":         {},
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

// thinkFromReasoningBudget maps mlx-serve reasoning_budget_tokens onto Think.
// 0 turns thinking off; >0 turns it on. Native Think still wins in FromChatRequest.
// This outranks reasoning_effort / enable_thinking. The integer is not a num_predict
// cap (that mixed visible reply length with hidden think — mlx-serve dropped it).
func thinkFromReasoningBudget(budget *int) (*api.ThinkValue, error) {
	if budget == nil {
		return nil, nil
	}
	if *budget < 0 {
		return nil, fmt.Errorf("reasoning_budget_tokens must be >= 0")
	}
	if *budget == 0 {
		return &api.ThinkValue{Value: false}, nil
	}
	return &api.ThinkValue{Value: true}, nil
}

// ThinkingFromReasoningEffort preserves model-defined names when metadata is present.
func ThinkingFromReasoningEffort(effort string, thinking ...*model.Thinking) (*api.ThinkValue, error) {
	switch effort {
	case "":
		return nil, nil
	case "none":
		return &api.ThinkValue{Value: false}, nil
	}
	requestedEffort := effort
	switch effort {
	case "minimal":
		effort = "low"
	case "xhigh", "ultra":
		effort = "max"
	}
	think := &api.ThinkValue{Value: effort}
	err := api.ValidateLegacyThinking(think)
	if len(thinking) > 0 && thinking[0].Valid() {
		if err == nil && thinking[0].Supports(true) {
			for _, value := range thinking[0].Values {
				if _, named := value.(string); named {
					return &api.ThinkValue{Value: requestedEffort}, nil
				}
			}
			return &api.ThinkValue{Value: true}, nil
		}
		return &api.ThinkValue{Value: requestedEffort}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("invalid reasoning value: %q (must be \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\", \"ultra\", \"max\", or \"none\")", requestedEffort)
	}
	return think, nil
}

func thinkFromReasoningEffort(effort string, thinking ...*model.Thinking) (*api.ThinkValue, error) {
	return ThinkingFromReasoningEffort(effort, thinking...)
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
	if v, ok := kwargs["thinking"]; ok {
		b, ok := v.(bool)
		if !ok {
			return nil, fmt.Errorf("invalid chat_template_kwargs.thinking: must be boolean")
		}
		return &api.ThinkValue{Value: b}, nil
	}
	if v, ok := kwargs["reasoning_effort"]; ok {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("invalid chat_template_kwargs.reasoning_effort: must be string")
		}
		effort := strings.TrimSpace(s)
		if !slices.Contains([]string{"high", "medium", "low", "none", "max", "xhigh"}, effort) {
			return nil, fmt.Errorf("invalid reasoning value: '%s' (must be \"high\", \"medium\", \"low\", \"xhigh\", \"max\", or \"none\")", effort)
		}
		if effort == "none" {
			return &api.ThinkValue{Value: false}, nil
		}
		return &api.ThinkValue{Value: effort}, nil
	}
	return nil, nil
}
