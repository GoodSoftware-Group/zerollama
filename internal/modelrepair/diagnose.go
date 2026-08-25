package modelrepair

import (
	"context"
	"fmt"
	"strings"
)

// API is the subset of the Go daemon used for diagnose/apply.
type API interface {
	Show(ctx context.Context, name string) (*ShowInfo, error)
	Generate(ctx context.Context, name string, prompt string, think *bool, opts map[string]any) (*GenerateResult, error)
	Chat(ctx context.Context, name string, messages []map[string]string, opts map[string]any) (*ChatResult, error)
	Create(ctx context.Context, name, from, template, parser string, params map[string]any) error
	ListRunning(ctx context.Context) ([]string, error)
	ListLocal(ctx context.Context) ([]string, error)
	Unload(ctx context.Context, name string) error
}

// Options controls Diagnose/Repair.
type Options struct {
	Apply bool
	// SkipLive skips generate/chat probes (static template checks only).
	// Why: unit tests and offline CI without a GPU; static ChatML+/no_think
	// absence still catches milkey-class Modelfiles.
	SkipLive bool
	// Progress is called with human status lines (optional).
	// Why stderr from the CLI: dry-run Modelfile dumps go to stdout; progress
	// must not corrupt --json or piped reports. Large GGUFs take minutes.
	Progress func(string)
}

func (o Options) progress(format string, args ...any) {
	if o.Progress == nil {
		return
	}
	o.Progress(fmt.Sprintf(format, args...))
}

var probeOpts = map[string]any{
	"num_ctx":     2048,
	"num_predict": 48,
	"temperature": 0,
}

// Diagnose inspects one model and optionally applies a patch.
func Diagnose(ctx context.Context, api API, name string, opts Options) (Report, error) {
	rep := Report{Name: name}
	opts.progress("show %s", name)
	show, err := api.Show(ctx, name)
	if err != nil {
		return rep, fmt.Errorf("show %s: %w", name, err)
	}
	show.Name = name

	qwenOK := isQwen3Family(*show)
	usesRenderer := usesBuiltinRenderer(*show)
	var findings []Finding
	var manual []string

	tmplEmpty := strings.TrimSpace(show.Template) == ""
	thinking := hasCapability(show.Capabilities, "thinking") ||
		looksThinkingParser(show.Parser) ||
		(strings.Contains(show.Template, "<think>") && strings.Contains(show.Template, "</think>"))

	// --- Static template hygiene (Unsloth-inspired: stops, empty TEMPLATE, Response) ---

	if tmplEmpty {
		if usesRenderer {
			// Built-in renderer; empty Go TEMPLATE is expected.
		} else if qwenOK {
			findings = append(findings, Finding{
				Recipe:  RecipeEmptyTemplate,
				Detail:  "TEMPLATE is empty — chat/tools/think will not assemble correctly",
				FixHint: "install stock Qwen3 ChatML TEMPLATE via --apply",
			})
		} else {
			manual = append(manual, fmt.Sprintf(
				"empty_template but not qwen3 family (parser=%q arch=%q) — set TEMPLATE manually or recreate from a known-good Modelfile",
				show.Parser, show.Architecture))
		}
	}

	if !tmplEmpty && looksChatML(show.Template) && chatMLMissingStops(*show) {
		findings = append(findings, Finding{
			Recipe:  RecipeChatMLMissingStops,
			Detail:  "ChatML TEMPLATE without PARAMETER stop <|im_end|> and/or <|im_start|>",
			FixHint: "add ChatML stop tokens (safe for any family)",
		})
	}

	if !tmplEmpty && !usesRenderer && templateMissingResponse(show.Template) {
		if looksChatML(show.Template) {
			findings = append(findings, Finding{
				Recipe:  RecipeMissingResponsePlaceholder,
				Detail:  "Go TEMPLATE lacks {{ .Response }} — /api/generate continuation may break",
				FixHint: "append Response placeholder (keeps existing Messages layout)",
			})
		} else {
			manual = append(manual, "TEMPLATE lacks {{ .Response }} and is not ChatML — review generate path manually")
		}
	}

	// Thinking PARSER without think markup / toggles.
	if !tmplEmpty && looksThinkingParser(show.Parser) && looksChatML(show.Template) && !templateHasThinkToggle(show.Template) {
		if qwenOK {
			findings = append(findings, Finding{
				Recipe:  RecipeThinkParserMismatch,
				Detail:  fmt.Sprintf("PARSER %q implies thinking but TEMPLATE has no /think|/no_think", show.Parser),
				FixHint: "patch TEMPLATE with /no_think + closed <think>; PARSER qwen3",
			})
		} else {
			manual = append(manual, fmt.Sprintf(
				"think_parser_mismatch (parser=%q) but not qwen3 family — refusing auto-patch", show.Parser))
		}
	}

	// Static: thinking-capable ChatML without think toggles (capability/arch path).
	if !tmplEmpty && thinking && looksChatML(show.Template) && !templateHasThinkToggle(show.Template) {
		// Avoid duplicate when ThinkParserMismatch already recorded.
		hasMismatch := false
		for _, f := range findings {
			if f.Recipe == RecipeThinkParserMismatch || f.Recipe == RecipeThinkGenerateEmpty {
				hasMismatch = true
				break
			}
		}
		if !hasMismatch {
			if qwenOK {
				findings = append(findings, Finding{
					Recipe:  RecipeThinkGenerateEmpty,
					Detail:  "thinking model ChatML template lacks /think|/no_think injection",
					FixHint: "patch TEMPLATE with /no_think + closed <think>; PARSER qwen3",
				})
			} else {
				manual = append(manual, fmt.Sprintf(
					"think_generate_empty symptoms (ChatML thinking, no /no_think) but not qwen3 family (parser=%q arch=%q) — refusing auto-patch",
					show.Parser, show.Architecture))
			}
		}
	}

	if !opts.SkipLive {
		opts.progress("live probes %s (may load GGUF; several minutes on large models)", name)
		live, unfixable, liveManual, err := liveProbes(ctx, api, name, thinking, qwenOK)
		if err != nil {
			return rep, err
		}
		rep.Unfixable = append(rep.Unfixable, unfixable...)
		manual = append(manual, liveManual...)
		findings = mergeFindings(findings, live)
	}

	rep.Findings = findings
	rep.ManualReview = manual

	patch, err := BuildPatch(name, *show, findings)
	if err != nil {
		return rep, err
	}
	rep.Patch = patch

	if opts.Apply && rep.Patch != nil {
		opts.progress("apply %s via /api/create", name)
		if err := Apply(ctx, api, name, rep.Patch); err != nil {
			rep.ApplyError = err.Error()
			return rep, nil
		}
		rep.Applied = true
		if !opts.SkipLive {
			opts.progress("re-probe %s after apply", name)
			post, still, postManual, err := liveProbes(ctx, api, name, thinking, qwenOK)
			if err == nil {
				rep.StillBroken = append(rep.StillBroken, still...)
				rep.ManualReview = append(rep.ManualReview, postManual...)
				for _, f := range post {
					rep.StillBroken = append(rep.StillBroken, string(f.Recipe)+": "+f.Detail)
				}
			}
		}
	}

	return rep, nil
}

func mergeFindings(base, extra []Finding) []Finding {
	seen := make(map[RecipeID]bool, len(base))
	for _, f := range base {
		seen[f.Recipe] = true
	}
	out := append([]Finding{}, base...)
	for _, f := range extra {
		if seen[f.Recipe] {
			continue
		}
		seen[f.Recipe] = true
		out = append(out, f)
	}
	return out
}

func liveProbes(ctx context.Context, api API, name string, thinking, qwenOK bool) (findings []Finding, unfixable, manual []string, err error) {
	_ = api.Unload(ctx, name)

	if thinking {
		genDefault, gerr := api.Generate(ctx, name, "Return exactly: <answer>42</answer>", nil, probeOpts)
		if gerr != nil {
			return nil, nil, nil, fmt.Errorf("generate probe: %w", gerr)
		}
		thinkFalse := false
		genOff, gerr := api.Generate(ctx, name, "Return exactly: <answer>42</answer>", &thinkFalse, probeOpts)
		if gerr != nil {
			return nil, nil, nil, fmt.Errorf("generate think=false probe: %w", gerr)
		}
		defEmpty := strings.TrimSpace(genDefault.Response) == "" && strings.TrimSpace(genDefault.Thinking) != ""
		offOK := strings.Contains(genOff.Response, "42") || strings.TrimSpace(genOff.Response) != ""
		thinkHit := defEmpty || (strings.TrimSpace(genDefault.Response) == "" && strings.TrimSpace(genDefault.Thinking) == "" && offOK)
		if thinkHit {
			detail := "default /api/generate empty while think=false returns content"
			if defEmpty {
				detail = fmt.Sprintf("default /api/generate empty response with thinking_len=%d (trap 12/64)", len(genDefault.Thinking))
			}
			if qwenOK {
				findings = append(findings, Finding{
					Recipe:  RecipeThinkGenerateEmpty,
					Detail:  detail,
					FixHint: "zerollama doctor --repair-models --apply; rebuild serve so Think defaults before parser Init",
				})
			} else {
				manual = append(manual, "think_generate_empty live symptom but not qwen3 family — refusing auto-patch: "+detail)
			}
		}
	}

	_ = api.Unload(ctx, name)
	userOnly, cerr := api.Chat(ctx, name, []map[string]string{
		{"role": "user", "content": "Return exactly: <answer>42</answer>"},
	}, withStops(probeOpts, []string{"<|im_end|>", "<|im_start|>"}))
	if cerr != nil {
		return findings, unfixable, manual, fmt.Errorf("chat user-only probe: %w", cerr)
	}
	userOK := strings.Contains(userOnly.Content, "42") && !isSlashCollapse(userOnly.Content)

	_ = api.Unload(ctx, name)
	sysChat, cerr := api.Chat(ctx, name, []map[string]string{
		{"role": "system", "content": "You output XML only."},
		{"role": "user", "content": "Return <answer>42</answer> and nothing else."},
	}, withStops(probeOpts, []string{"<|im_end|>", "<|im_start|>"}))
	if cerr != nil {
		return findings, unfixable, manual, fmt.Errorf("chat system probe: %w", cerr)
	}

	sysBad := isSlashCollapse(sysChat.Content) || (userOK && isEmptyCollapse(sysChat.Content, sysChat.EvalCount))
	if userOK && sysBad {
		detail := fmt.Sprintf("user-only ok but system+user collapsed (content=%q eval=%d)", truncate(sysChat.Content, 40), sysChat.EvalCount)
		if qwenOK {
			findings = append(findings, Finding{
				Recipe:  RecipeSlashSystemCollapse,
				Detail:  detail,
				FixHint: "drop system role from TEMPLATE; stripRolePrefixes + anti-slash steer (needs serve with stripRolePrefixes)",
			})
		} else {
			manual = append(manual, "slash_system_collapse live symptom but not qwen3 family — refusing auto-patch: "+detail)
		}
	}

	// Roleplay generate with System:/Assistant: cannot be fixed via template.
	_ = api.Unload(ctx, name)
	roleplay, gerr := api.Generate(ctx, name,
		"System: You output XML only.\nUser: Return <answer>42</answer> and nothing else.\nAssistant:",
		nil, withStops(probeOpts, []string{"<|im_end|>", "<|im_start|>"}))
	if gerr == nil {
		if isSlashCollapse(roleplay.Response) || isEmptyCollapse(roleplay.Response, roleplay.EvalCount) {
			unfixable = append(unfixable,
				"roleplay System:/Assistant: in prompt text still collapses — template cannot strip user text; change the bench prompt")
		}
	}

	return findings, unfixable, manual, nil
}

func withStops(base map[string]any, stops []string) map[string]any {
	out := make(map[string]any, len(base)+1)
	for k, v := range base {
		out[k] = v
	}
	out["stop"] = stops
	// Why temp 0.7 for slash probes: collapse often appears at sampling temps
	// while temp=0 can look clean on the same GGUF (moophlo cold series).
	out["temperature"] = 0.7
	return out
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func usesBuiltinRenderer(show ShowInfo) bool {
	if strings.TrimSpace(show.Renderer) != "" {
		return true
	}
	p := strings.ToLower(strings.TrimSpace(show.Parser))
	if strings.HasPrefix(p, "qwen3.5") || strings.HasPrefix(p, "qwen3.8") || p == "qwen35" {
		return true
	}
	return rendererFromModelfile(show.Modelfile) != ""
}

func rendererFromModelfile(mf string) string {
	for _, line := range strings.Split(mf, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		if strings.EqualFold(fields[0], "RENDERER") {
			return fields[1]
		}
	}
	return ""
}

// ListTargets returns model names to scan: explicit args, all local tags, or loaded runners.
// Why warm-only when args empty (and allLocal false): same safety rule as live doctor —
// never surprise-cold-load the whole library on an operator's production stack.
// Why allLocal is explicit: --all-local opts into /api/tags; each diagnose may load GGUF.
func ListTargets(ctx context.Context, api API, args []string, allLocal bool) ([]string, error) {
	if len(args) > 0 {
		return args, nil
	}
	if allLocal {
		return api.ListLocal(ctx)
	}
	return api.ListRunning(ctx)
}

// DiagnoseAll runs Diagnose for each target name.
func DiagnoseAll(ctx context.Context, api API, names []string, opts Options) ([]Report, error) {
	out := make([]Report, 0, len(names))
	for i, name := range names {
		opts.progress("[%d/%d] diagnosing %s", i+1, len(names), name)
		r, err := Diagnose(ctx, api, name, opts)
		if err != nil {
			r.SkipReason = err.Error()
			r.Skipped = true
			out = append(out, r)
			continue
		}
		out = append(out, r)
	}
	return out, nil
}
