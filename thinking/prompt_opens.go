package thinking

import "strings"

// PromptOpensTag reports whether prompt has an unclosed `open` after the last `close`.
// WHY bytes not the request flag: LFM2.5 / similar templates can open <think>
// even when think is off. mlx-serve promptOpensThink.
func PromptOpensTag(prompt, open, close string) bool {
	if prompt == "" || open == "" {
		return false
	}
	i := strings.LastIndex(prompt, open)
	if i < 0 {
		return false
	}
	if close == "" {
		return true
	}
	return i > strings.LastIndex(prompt, close)
}

func PromptOpensThink(prompt string) bool {
	return PromptOpensTag(prompt, "<think>", "</think>")
}

// SeedFromPrompt starts in-think when the rendered prompt already opened the tag.
func (s *Parser) SeedFromPrompt(prompt string) {
	if s == nil || s.OpeningTag == "" {
		return
	}
	if PromptOpensTag(prompt, s.OpeningTag, s.ClosingTag) {
		s.AddContent(s.OpeningTag)
	}
}
