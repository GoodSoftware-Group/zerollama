package template

import (
	"regexp"
	"strings"
)

// goActionRe matches text/template actions so we can see whether anything
// outside {{ ... }} remains (role markup, labels, etc.).
var goActionRe = regexp.MustCompile(`\{\{[\s\S]*?\}\}`)

// IsPassthrough reports whether tmpl is a Go TEMPLATE that only forwards
// .Prompt (optionally .System / .Response) with no chat-role assembly.
//
// Why: operators often ship TEMPLATE {{ .Prompt }} after create-from-GGUF.
// That satisfies "has a template layer" checks (minefield trap 56) but leaves
// chat models without DeepSeek/ChatML framing, so replies look like gibberish
// or instruction-echo. Builtin RENDERER paths fix this; empty TEMPLATE with a
// renderer is fine and is not passthrough.
func IsPassthrough(tmpl string) bool {
	t := strings.TrimSpace(tmpl)
	if t == "" {
		return false
	}
	if !strings.Contains(t, "{{") {
		return false
	}
	if looksStructuredChatTemplate(t) {
		return false
	}
	lower := strings.ToLower(t)
	if strings.Contains(lower, ".messages") {
		// Messages-range templates are real chat assembly even without <|…|>.
		return false
	}
	if !strings.Contains(t, ".Prompt") && !strings.Contains(lower, ".prompt") {
		return false
	}
	// After stripping actions, only whitespace may remain.
	if strings.TrimSpace(goActionRe.ReplaceAllString(t, "")) != "" {
		return false
	}
	return true
}

func looksStructuredChatTemplate(t string) bool {
	markers := []string{
		"<|im_start|>",
		"<|im_end|>",
		"<｜User｜>",
		"<｜Assistant｜>",
		"<|start_header_id|>",
		"<|end_header_id|>",
		"[INST]",
		"<<SYS>>",
		"<start_of_turn>",
		"<end_of_turn>",
		"im_start",
	}
	for _, m := range markers {
		if strings.Contains(t, m) {
			return true
		}
	}
	return false
}
