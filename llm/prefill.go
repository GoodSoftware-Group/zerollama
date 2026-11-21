package llm

import (
	"context"

	"github.com/ollama/ollama/api"
)

// PrefillSentinel is U+F8FF. The draft appends it to the last message so the
// rendered template can be cut before the end-of-turn marker. A normal chat
// template copies it through unchanged.
//
// WHY a private-use character: cutting the last word of the rendered string
// would also cut the end-of-turn marker, and the cached text would not be a
// prefix of the finished chat.
const PrefillSentinel = "\uF8FF"

// PrefillResult is how much of a draft prompt is now in the slot.
type PrefillResult struct {
	CachedTokens int
	PromptTokens int
}

// PromptPrefiller evaluates a prompt and keeps it, without sampling.
// Messages is the llama-server chat template. Prompt is an already rendered
// string, used when Go rendered the chat itself.
type PromptPrefiller interface {
	PrefillMessages(ctx context.Context, messages []api.Message, think *api.ThinkValue, tools api.Tools) (PrefillResult, error)
	PrefillPrompt(ctx context.Context, prompt string, opts *api.Options) (PrefillResult, error)
}

// llamaServerCachePrompt is false when the client asked this request to miss.
//
// WHY a request flag, not an env var: one chat can force a miss. The
// process-wide LLAMA_* policies stay on for every other request.
func llamaServerCachePrompt(reset bool) bool {
	return !reset
}

// PrefillBeforeLastWord drops the unfinished last word.
// "two words  " is "two". A single unfinished word is empty.
func PrefillBeforeLastWord(text string) string {
	text = trimPrefillEnd(text)
	end := len(text)
	start := 0
	var cp rune
	found := false
	for {
		var ok bool
		start, cp, ok = prefillPrev(text, end)
		if !ok {
			break
		}
		if prefillIsSpace(cp) {
			found = true
			break
		}
		end = start
	}
	if !found {
		return ""
	}
	return trimPrefillEnd(text[:start])
}

func trimPrefillEnd(s string) string {
	end := len(s)
	for {
		start, cp, ok := prefillPrev(s, end)
		if !ok || !prefillIsSpace(cp) {
			break
		}
		end = start
	}
	return s[:end]
}

func prefillIsSpace(cp rune) bool {
	switch cp {
	case 0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
		return true
	default:
		return cp >= 0x2000 && cp <= 0x200A
	}
}

// prefillPrev walks one UTF-8 code point backward. Invalid bytes are consumed
// one at a time so a bad sequence cannot walk off the start of the string.
func prefillPrev(s string, end int) (start int, cp rune, ok bool) {
	if end == 0 || end > len(s) {
		return 0, 0, false
	}
	i := end - 1
	for i > 0 && s[i]&0xC0 == 0x80 {
		i--
	}
	b0 := s[i]
	n := end - i
	switch {
	case b0 < 0x80 && n == 1:
		cp = rune(b0)
	case b0&0xE0 == 0xC0 && n == 2:
		cp = rune(b0&0x1F)<<6 | rune(s[i+1]&0x3F)
	case b0&0xF0 == 0xE0 && n == 3:
		cp = rune(b0&0x0F)<<12 | rune(s[i+1]&0x3F)<<6 | rune(s[i+2]&0x3F)
	case b0&0xF8 == 0xF0 && n == 4:
		cp = rune(b0&0x07)<<18 | rune(s[i+1]&0x3F)<<12 | rune(s[i+2]&0x3F)<<6 | rune(s[i+3]&0x3F)
	default:
		i = end - 1
		cp = rune(s[i])
	}
	return i, cp, true
}
