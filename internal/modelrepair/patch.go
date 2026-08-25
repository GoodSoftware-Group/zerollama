package modelrepair

import (
	"fmt"
	"strconv"
	"strings"
)

func hasCapability(caps []string, want string) bool {
	for _, c := range caps {
		if strings.EqualFold(c, want) {
			return true
		}
	}
	return false
}

func templateHasThinkToggle(tmpl string) bool {
	return strings.Contains(tmpl, "/no_think") || strings.Contains(tmpl, "/think")
}

func looksChatML(tmpl string) bool {
	return strings.Contains(tmpl, "<|im_start|>") || strings.Contains(tmpl, "im_start")
}

// isQwen3Family is true when auto-applying Qwen3 ChatML templates is safe.
// Heuristic: PARSER name or GGUF architecture must be qwen3*.
// Why: textual ChatML + <think> detection alone matches unrelated families
// (Llama/Hermes finetunes with ChatML-flavored templates). Overwriting those
// with /think|/no_think would inject control strings the model never trained on.
func isQwen3Family(show ShowInfo) bool {
	p := strings.ToLower(strings.TrimSpace(show.Parser))
	if strings.Contains(p, "qwen3") {
		return true
	}
	arch := strings.ToLower(strings.TrimSpace(show.Architecture))
	if arch == "qwen3" || arch == "qwen3moe" || strings.HasPrefix(arch, "qwen3") {
		return true
	}
	// Modelfile PARSER line (show.Parser empty on some tags).
	for _, line := range strings.Split(show.Modelfile, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(line), "PARSER ") {
			if strings.Contains(strings.ToLower(line), "qwen3") {
				return true
			}
		}
	}
	return false
}

func isSlashCollapse(content string) bool {
	s := strings.TrimSpace(content)
	if s == "" {
		return false
	}
	// Why these thresholds: reverse-engineered from moophlo cold probes where
	// collapse was either "///…" runs or content that is majority slashes.
	// Fixed probe prompts keep false positives low; not a general classifier.
	if strings.HasPrefix(s, "///") || strings.Contains(s, "////////") {
		return true
	}
	slashes := strings.Count(s, "/")
	return len(s) >= 8 && slashes*2 >= len(s)
}

func isEmptyCollapse(content string, evalCount int) bool {
	// Why eval≤4: stop /// (or early stop) after a few slash tokens leaves
	// empty content with a tiny eval_count — same failure mode as visible "///".
	return strings.TrimSpace(content) == "" && evalCount > 0 && evalCount <= 4
}

// ParseParameters parses /api/show Parameters text into a create-ready map.
func ParseParameters(s string) map[string]any {
	out := make(map[string]any)
	if strings.TrimSpace(s) == "" {
		return out
	}
	stops := make([]any, 0)
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		key := fields[0]
		raw := strings.TrimSpace(strings.TrimPrefix(line, key))
		val, err := strconv.Unquote(raw)
		if err != nil {
			// %#v for numbers/bools leaves no quotes; try parse
			if f, err2 := strconv.ParseFloat(raw, 64); err2 == nil {
				if strings.Contains(raw, ".") {
					out[key] = f
				} else {
					out[key] = int(f)
				}
				continue
			}
			if b, err2 := strconv.ParseBool(raw); err2 == nil {
				out[key] = b
				continue
			}
			val = raw
		}
		if key == "stop" {
			stops = append(stops, val)
			continue
		}
		out[key] = val
	}
	if len(stops) > 0 {
		out["stop"] = stops
	}
	return out
}

func ensureStop(params map[string]any, token string) {
	if params == nil {
		return
	}
	existing, _ := params["stop"]
	var list []any
	switch v := existing.(type) {
	case []any:
		list = append(list, v...)
	case []string:
		for _, s := range v {
			list = append(list, s)
		}
	case string:
		list = append(list, v)
	}
	for _, s := range list {
		if fmt.Sprint(s) == token {
			params["stop"] = list
			return
		}
	}
	list = append(list, token)
	params["stop"] = list
}

func hasStopToken(params map[string]any, token string) bool {
	existing, _ := params["stop"]
	switch v := existing.(type) {
	case []any:
		for _, s := range v {
			if fmt.Sprint(s) == token {
				return true
			}
		}
	case []string:
		for _, s := range v {
			if s == token {
				return true
			}
		}
	case string:
		return v == token
	}
	return false
}

func chatMLMissingStops(show ShowInfo) bool {
	if !looksChatML(show.Template) {
		return false
	}
	params := ParseParameters(show.Parameters)
	return !hasStopToken(params, "<|im_end|>") || !hasStopToken(params, "<|im_start|>")
}

func templateMissingResponse(tmpl string) bool {
	t := strings.TrimSpace(tmpl)
	if t == "" {
		return false // empty is RecipeEmptyTemplate
	}
	// Only flag Go-style templates (ollama TEMPLATE), not raw Jinja leftovers.
	if !strings.Contains(t, "{{") {
		return false
	}
	return !strings.Contains(t, ".Response")
}

func looksThinkingParser(parser string) bool {
	p := strings.ToLower(parser)
	return strings.Contains(p, "thinking")
}

// BuildPatch combines findings into one overlay Patch.
// Returns (nil, nil) when findings is empty. Returns an error for unknown
// recipe IDs so a future recipe cannot silently produce an empty overlay.
// Why fail closed: an empty TEMPLATE/PARSER on /api/create inherits FROM (safe
// today) but that is incidental — better to error than ship a no-op patch that
// looks applied.
func BuildPatch(name string, show ShowInfo, findings []Finding) (*Patch, error) {
	if len(findings) == 0 {
		return nil, nil
	}
	think := false
	slash := false
	stopsOnly := false
	emptyTmpl := false
	needResponse := false
	for _, f := range findings {
		switch f.Recipe {
		case RecipeThinkGenerateEmpty, RecipeThinkParserMismatch:
			think = true
		case RecipeSlashSystemCollapse:
			slash = true
		case RecipeChatMLMissingStops:
			stopsOnly = true
		case RecipeEmptyTemplate:
			emptyTmpl = true
		case RecipeMissingResponsePlaceholder:
			needResponse = true
		default:
			return nil, fmt.Errorf("unhandled repair recipe %q", f.Recipe)
		}
	}

	p := &Patch{
		Parameters: ParseParameters(show.Parameters),
	}
	if p.Parameters == nil {
		p.Parameters = make(map[string]any)
	}

	thinkingCaps := hasCapability(show.Capabilities, "thinking") ||
		looksThinkingParser(show.Parser) ||
		(strings.Contains(show.Template, "<think>") && strings.Contains(show.Template, "</think>"))

	rewroteTemplate := false
	switch {
	case think && slash:
		p.Template = templateQwen3ThinkNoThinkNoSystem
		p.Parser = "qwen3"
		rewroteTemplate = true
	case think || (emptyTmpl && thinkingCaps):
		p.Template = templateQwen3ThinkNoThink
		p.Parser = "qwen3"
		rewroteTemplate = true
	case slash:
		p.Template = templateChatMLNoSystem
		if show.Parser != "" && strings.Contains(strings.ToLower(show.Parser), "qwen3") {
			p.Parser = show.Parser
		} else {
			p.Parser = "qwen3"
		}
		rewroteTemplate = true
	case emptyTmpl:
		p.Template = templateChatMLStock
		if show.Parser != "" {
			p.Parser = show.Parser
		} else {
			p.Parser = "qwen3"
		}
		rewroteTemplate = true
	case needResponse && looksChatML(show.Template):
		// Keep operator template; only restore generate continuation.
		p.Template = strings.TrimRight(show.Template, "\n") + "\n" + templateResponseSuffix + "\n"
		p.Parser = show.Parser
		rewroteTemplate = true
	case stopsOnly:
		// Parameters-only hygiene: inherit TEMPLATE/PARSER from FROM on create.
		p.Template = ""
		p.Parser = ""
	default:
		return nil, fmt.Errorf("no patchable recipe combination in %d finding(s)", len(findings))
	}

	if rewroteTemplate || stopsOnly || looksChatML(p.Template) || looksChatML(show.Template) {
		ensureStop(p.Parameters, "<|im_start|>")
		ensureStop(p.Parameters, "<|im_end|>")
	}

	p.Modelfile = formatModelfilePreview(name, p)
	return p, nil
}

func formatModelfilePreview(name string, p *Patch) string {
	var b strings.Builder
	fmt.Fprintf(&b, "FROM %s\n", name)
	if p.Template != "" {
		b.WriteString("TEMPLATE \"\"\"")
		b.WriteString(p.Template)
		b.WriteString("\"\"\"\n")
	}
	if p.Parser != "" {
		fmt.Fprintf(&b, "PARSER %s\n", p.Parser)
	}
	if stops, ok := p.Parameters["stop"]; ok {
		switch v := stops.(type) {
		case []any:
			for _, s := range v {
				fmt.Fprintf(&b, "PARAMETER stop %v\n", s)
			}
		case []string:
			for _, s := range v {
				fmt.Fprintf(&b, "PARAMETER stop %s\n", s)
			}
		default:
			fmt.Fprintf(&b, "PARAMETER stop %v\n", v)
		}
	}
	for k, v := range p.Parameters {
		if k == "stop" {
			continue
		}
		fmt.Fprintf(&b, "PARAMETER %s %v\n", k, v)
	}
	return b.String()
}
