// Package modelrepair diagnoses and patches local model tags that fail common
// serving traps (empty generate response on thinking models; slash-collapse on
// system prompts). Used by `zerollama doctor --repair-models`.
//
// Why a separate package (not doctor --fix, not zerollama repair):
//   - doctor --fix is host bootstrap (uv / Metal llama.cpp); mutating Modelfiles
//     under that flag surprises operators who only meant “install the toolchain.”
//   - zerollama repair rewrites GGUF metadata layers without live probes; it cannot
//     see empty-response-in-thinking or slash loops.
//   - Harnesses often score these GGUFs as “broken weights” when the assembled
//     prompt / parser / default think routing is the real fault.
//
// Auto-apply for invasive TEMPLATE rewrites is gated to Qwen3 family only.
// Hygiene recipes that only add PARAMETER stop tokens apply to any ChatML tag.
package modelrepair

import "fmt"

// RecipeID names a repair recipe.
type RecipeID string

const (
	// RecipeThinkGenerateEmpty: default /api/generate parks the answer in thinking.
	// Why: PARSER qwen3-thinking + ChatML without /no_think, combined with older
	// serve Init(Think=nil) → defaultThinking, leaves harnesses that only read
	// `response` with an empty scoreable channel.
	RecipeThinkGenerateEmpty RecipeID = "think_generate_empty"
	// RecipeSlashSystemCollapse: ChatML system (or folded system) triggers "/" loops.
	// Why: some Qwen3-Coder GGUFs collapse on <|im_start|>system and on harness
	// System:/User:/Assistant: text inside the user turn. Drop system +
	// stripRolePrefixes + a one-line anti-filler steer. Do not rely on stop /// —
	// that empties the reply and can poison the runner slot until unload.
	RecipeSlashSystemCollapse RecipeID = "slash_system_collapse"
	// RecipeChatMLMissingStops: ChatML template without <|im_end|>/<|im_start|> stops.
	// Why: without stop tokens the model keeps emitting role markers into the
	// next turn; Unsloth/Ollama-style hygiene always pairs ChatML with these stops.
	// Safe across families — parameters only, no TEMPLATE rewrite.
	RecipeChatMLMissingStops RecipeID = "chatml_missing_stops"
	// RecipeEmptyTemplate: TEMPLATE layer is empty while the tag is a chat model.
	// Why: empty TEMPLATE falls through to raw prompt assembly; chat/tools/think
	// all degrade. Qwen3 auto-patch installs stock ChatML; others → manual_review.
	RecipeEmptyTemplate RecipeID = "empty_template"
	// RecipeMissingResponsePlaceholder: Go TEMPLATE lacks {{ .Response }}.
	// Why: /api/generate continuation and some create/train paths need the
	// assistant-so-far placeholder; Messages-only templates break generate.
	RecipeMissingResponsePlaceholder RecipeID = "missing_response_placeholder"
	// RecipeThinkParserMismatch: thinking PARSER without think markup (or inverse).
	// Why: PARSER qwen3-thinking + plain ChatML (no toggles/<think>) is the same
	// class of trap as think_generate_empty; report explicitly for operators.
	RecipeThinkParserMismatch RecipeID = "think_parser_mismatch"
)

// Finding is one diagnosed issue for a model.
type Finding struct {
	Recipe  RecipeID `json:"recipe"`
	Detail  string   `json:"detail"`
	FixHint string   `json:"fix_hint,omitempty"`
}

// Patch is the proposed Modelfile overlay (FROM + overrides).
type Patch struct {
	Template   string         `json:"template,omitempty"`
	Parser     string         `json:"parser,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
	// Modelfile is a human-readable preview suitable for dry-run output.
	Modelfile string `json:"modelfile,omitempty"`
}

// Report is the repair result for one model tag.
type Report struct {
	Name        string    `json:"name"`
	Findings    []Finding `json:"findings,omitempty"`
	Patch       *Patch    `json:"patch,omitempty"`
	Applied     bool      `json:"applied,omitempty"`
	ApplyError  string    `json:"apply_error,omitempty"`
	StillBroken []string  `json:"still_broken,omitempty"`
	Skipped     bool      `json:"skipped,omitempty"`
	SkipReason  string    `json:"skip_reason,omitempty"`
	Unfixable   []string  `json:"unfixable,omitempty"`
	// ManualReview: symptoms matched but architecture/parser is outside the
	// safe auto-patch family (Qwen3). Never applied automatically.
	// Why: better to leave a broken-looking tag alone than overwrite a Llama
	// template with Qwen3 /think control strings the model was never trained on.
	ManualReview []string `json:"manual_review,omitempty"`
}

// ShowInfo is the subset of /api/show used for diagnosis.
type ShowInfo struct {
	Name         string
	Template     string
	Parser       string
	Parameters   string
	Capabilities []string
	Modelfile    string
	// Architecture is GGUF general.architecture when present (e.g. "qwen3moe").
	Architecture string
	// Renderer is a built-in chat renderer (qwen3.8, gemma4, …). Those paths
	// assemble prompts in Go; a missing {{ .Response }} in TEMPLATE is not a bug.
	Renderer string
}

// GenerateResult is a trimmed /api/generate response.
type GenerateResult struct {
	Response   string
	Thinking   string
	EvalCount  int
	DoneReason string
}

// ChatResult is a trimmed /api/chat response.
type ChatResult struct {
	Content    string
	Thinking   string
	EvalCount  int
	DoneReason string
}

func (r Report) HasFindings() bool {
	return len(r.Findings) > 0
}

func (r Report) Summary() string {
	if r.Skipped {
		return fmt.Sprintf("%s: skipped (%s)", r.Name, r.SkipReason)
	}
	if !r.HasFindings() {
		if len(r.ManualReview) > 0 {
			return fmt.Sprintf("%s: ok (no auto-patch; %d manual-review note(s))", r.Name, len(r.ManualReview))
		}
		return fmt.Sprintf("%s: ok (no repair recipes matched)", r.Name)
	}
	parts := make([]string, 0, len(r.Findings))
	for _, f := range r.Findings {
		parts = append(parts, string(f.Recipe))
	}
	s := fmt.Sprintf("%s: %d finding(s): %v", r.Name, len(r.Findings), parts)
	if r.Applied {
		s += " (applied)"
	}
	return s
}
