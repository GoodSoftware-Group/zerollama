package api

import (
	"fmt"
	"slices"
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

// ApplyChatThinkingAliases maps enable_thinking / chat_template_kwargs onto Think
// when Think is unset, and rejects unknown nested kwargs (minefield traps 07 + 77).
func ApplyChatThinkingAliases(req *ChatRequest) error {
	if req == nil {
		return nil
	}
	if err := validateChatTemplateKwargs(req.ChatTemplateKwargs); err != nil {
		return err
	}
	if req.Think != nil {
		return nil
	}
	if req.EnableThinking != nil {
		req.Think = &ThinkValue{Value: *req.EnableThinking}
		return nil
	}
	if req.ChatTemplateKwargs == nil {
		return nil
	}
	if v, ok := req.ChatTemplateKwargs["enable_thinking"]; ok {
		b, ok := v.(bool)
		if !ok {
			return fmt.Errorf("invalid chat_template_kwargs.enable_thinking: must be boolean")
		}
		req.Think = &ThinkValue{Value: b}
		return nil
	}
	if v, ok := req.ChatTemplateKwargs["reasoning_effort"]; ok {
		s, ok := v.(string)
		if !ok {
			return fmt.Errorf("invalid chat_template_kwargs.reasoning_effort: must be string")
		}
		effort := strings.TrimSpace(s)
		if !slices.Contains([]string{"high", "medium", "low", "none"}, effort) {
			return fmt.Errorf("invalid reasoning value: '%s' (must be \"high\", \"medium\", \"low\", or \"none\")", effort)
		}
		if effort == "none" {
			req.Think = &ThinkValue{Value: false}
		} else {
			req.Think = &ThinkValue{Value: effort}
		}
	}
	return nil
}
