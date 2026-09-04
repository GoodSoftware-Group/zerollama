package freetokenlab

// ToolEdit is an agent harness truncation: drop everything after the last
// surviving semantic boundary (think / tool / turn), then append a suffix.
type ToolEdit struct {
	// Tokens kept from the previous prompt (prefix that still matches).
	KeptPrefix int
	// Tokens that must be re-prefilled (new suffix after the edit).
	NewSuffix int
}

// PrefillTokensWithoutAnchor is last-checkpoint-before-edit when checkpoints
// are sparse (every N tokens). If the last checkpoint is far before KeptPrefix,
// the engine re-prefills from that checkpoint through the suffix.
func PrefillTokensWithoutAnchor(edit ToolEdit, checkpointEvery int) int {
	if checkpointEvery < 1 {
		checkpointEvery = 1
	}
	last := (edit.KeptPrefix / checkpointEvery) * checkpointEvery
	replay := edit.KeptPrefix - last
	if replay < 0 {
		replay = 0
	}
	return replay + edit.NewSuffix
}

// PrefillTokensWithSemanticAnchor assumes a checkpoint sits on the kept
// prefix boundary (tool/think token), so only the new suffix is recomputed.
func PrefillTokensWithSemanticAnchor(edit ToolEdit) int {
	if edit.NewSuffix < 0 {
		return 0
	}
	return edit.NewSuffix
}

// KeepTailCompressReplay is zerollama chat_compress: preserve system prefix,
// replace old head with a *new* summary, keep recent tail.
//
// Radix/prefix-KV can reuse only PrefixTokens. Tail KV from the previous
// prompt is invalid: it was computed after `head`, at different positions
// than after `summary`. A FreeToken-style checkpoint at the start of tail
// is also invalid (recurrent state there saw `head`, not `summary`).
func KeepTailCompressReplay(prefixTokens, summaryTokens, tailTokens int) (reuse, recompute int) {
	if prefixTokens < 0 {
		prefixTokens = 0
	}
	if summaryTokens < 0 {
		summaryTokens = 0
	}
	if tailTokens < 0 {
		tailTokens = 0
	}
	return prefixTokens, summaryTokens + tailTokens
}

// SuffixStripEdit is the FreeToken/OpenClaw pattern: the surviving prompt is
// an exact prefix of the previous prompt, then a new suffix.
func SuffixStripEdit(keptPrefix, newSuffix int) ToolEdit {
	if keptPrefix < 0 {
		keptPrefix = 0
	}
	if newSuffix < 0 {
		newSuffix = 0
	}
	return ToolEdit{KeptPrefix: keptPrefix, NewSuffix: newSuffix}
}
