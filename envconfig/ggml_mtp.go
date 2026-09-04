package envconfig

import "strings"

// GgmlMTPObserveEnabled reports ZEROLLAMA_GGML_MTP. Default off.
// When on, ollamarunner drafts with in-checkpoint MTP after each decode.
// The next graph may verify a 2-token pair. Reject drops causal KV; hybrid GDN
// serializes the pair, checkpoints after the first token, then Restore+Remove.
func GgmlMTPObserveEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(Var("ZEROLLAMA_GGML_MTP")))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
