package api

import (
	"fmt"
	"sort"
	"strings"
)

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

func applyThinkingAliasFields(think **ThinkValue, enable *bool, kwargs map[string]any, fromAlias *bool) error {
	if err := validateChatTemplateKwargs(kwargs); err != nil {
		return err
	}
	if think == nil || *think != nil {
		return nil
	}
	set := func(v *ThinkValue) {
		*think = v
		if fromAlias != nil {
			*fromAlias = true
		}
	}
	if enable != nil {
		set(&ThinkValue{Value: *enable})
		return nil
	}
	if kwargs == nil {
		return nil
	}
	if v, ok := kwargs["enable_thinking"]; ok {
		b, ok := v.(bool)
		if !ok {
			return fmt.Errorf("invalid chat_template_kwargs.enable_thinking: must be boolean")
		}
		set(&ThinkValue{Value: b})
		return nil
	}
	if v, ok := kwargs["reasoning_effort"]; ok {
		s, ok := v.(string)
		if !ok {
			return fmt.Errorf("invalid chat_template_kwargs.reasoning_effort: must be string")
		}
		effort := strings.TrimSpace(s)
		if effort != "none" && !validThinkLevel(effort) {
			return fmt.Errorf("invalid reasoning value: '%s' (must be \"high\", \"medium\", \"low\", \"xhigh\", \"max\", or \"none\")", effort)
		}
		if effort == "none" {
			set(&ThinkValue{Value: false})
		} else {
			set(&ThinkValue{Value: effort})
		}
	}
	return nil
}

// ApplyChatThinkingAliases maps enable_thinking / chat_template_kwargs onto Think
// when Think is unset, and rejects unknown nested kwargs (minefield traps 07 + 77).
func ApplyChatThinkingAliases(req *ChatRequest) error {
	if req == nil {
		return nil
	}
	return applyThinkingAliasFields(&req.Think, req.EnableThinking, req.ChatTemplateKwargs, &req.ThinkFromAlias)
}

// ApplyGenerateThinkingAliases is the /api/generate counterpart of ApplyChatThinkingAliases.
func ApplyGenerateThinkingAliases(req *GenerateRequest) error {
	if req == nil {
		return nil
	}
	return applyThinkingAliasFields(&req.Think, req.EnableThinking, req.ChatTemplateKwargs, nil)
}
