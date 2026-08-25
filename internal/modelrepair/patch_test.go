package modelrepair

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestParseParameters(t *testing.T) {
	raw := "num_ctx                        8192\ntemperature                    0.6\nstop                           \"<|im_start|>\"\nstop                           \"<|im_end|>\"\n"
	p := ParseParameters(raw)
	if v, ok := p["num_ctx"].(int); !ok || v != 8192 {
		t.Fatalf("num_ctx=%v (%T)", p["num_ctx"], p["num_ctx"])
	}
	stops, ok := p["stop"].([]any)
	if !ok || len(stops) != 2 {
		t.Fatalf("stop=%v", p["stop"])
	}
}

func TestTemplateHasThinkToggle(t *testing.T) {
	if templateHasThinkToggle("hello") {
		t.Fatal("expected false")
	}
	if !templateHasThinkToggle(templateQwen3ThinkNoThink) {
		t.Fatal("stock think template should have toggles")
	}
}

func TestIsSlashCollapse(t *testing.T) {
	if !isSlashCollapse("///////////////////////////////") {
		t.Fatal("expected slash collapse")
	}
	if isSlashCollapse("<answer>42</answer>") {
		t.Fatal("xml should not be slash collapse")
	}
}

func TestBuildPatchThinkOnly(t *testing.T) {
	show := ShowInfo{
		Template:   "<|im_start|>user\n{{ .Content }}",
		Parameters: "temperature                    0.6\n",
		Parser:     "qwen3-thinking",
	}
	p, err := BuildPatch("m:latest", show, []Finding{{Recipe: RecipeThinkGenerateEmpty}})
	if err != nil {
		t.Fatal(err)
	}
	if p == nil || p.Parser != "qwen3" {
		t.Fatalf("patch=%+v", p)
	}
	if !strings.Contains(p.Template, "/no_think") {
		t.Fatal("expected /no_think in template")
	}
	if !strings.Contains(p.Modelfile, "FROM m:latest") {
		t.Fatal("modelfile preview missing FROM")
	}
}

func TestBuildPatchSlashOnly(t *testing.T) {
	show := ShowInfo{Parameters: "stop                           \"<|im_end|>\"\n", Parser: "qwen3"}
	p, err := BuildPatch("m:latest", show, []Finding{{Recipe: RecipeSlashSystemCollapse}})
	if err != nil {
		t.Fatal(err)
	}
	if p == nil {
		t.Fatal("nil patch")
	}
	if !strings.Contains(p.Template, `ne $m.Role "system"`) {
		t.Fatal("expected system drop in template")
	}
	if !strings.Contains(p.Template, "stripRolePrefixes") {
		t.Fatal("expected stripRolePrefixes in slash-collapse template")
	}
	if !strings.Contains(p.Template, "Reply with useful content only.") {
		t.Fatal("expected anti-slash steer in template")
	}
	stops, _ := p.Parameters["stop"].([]any)
	for _, s := range stops {
		if fmt.Sprint(s) == "///" {
			t.Fatalf("did not expect /// stop (poisons runner); got %v", stops)
		}
	}
}

func TestBuildPatchBoth(t *testing.T) {
	p, err := BuildPatch("m:latest", ShowInfo{}, []Finding{
		{Recipe: RecipeThinkGenerateEmpty},
		{Recipe: RecipeSlashSystemCollapse},
	})
	if err != nil {
		t.Fatal(err)
	}
	if p.Parser != "qwen3" {
		t.Fatal(p.Parser)
	}
	if !strings.Contains(p.Template, "/no_think") {
		t.Fatal("need think toggles")
	}
	if strings.Contains(p.Template, "or .System .Tools") {
		t.Fatal("combined patch should not emit system from .System/.Tools")
	}
}

func TestBuildPatchUnknownRecipe(t *testing.T) {
	_, err := BuildPatch("m:latest", ShowInfo{}, []Finding{{Recipe: RecipeID("nope")}})
	if err == nil {
		t.Fatal("expected error for unknown recipe")
	}
}

func TestIsQwen3Family(t *testing.T) {
	if !isQwen3Family(ShowInfo{Parser: "qwen3-thinking"}) {
		t.Fatal("parser")
	}
	if !isQwen3Family(ShowInfo{Architecture: "qwen3moe"}) {
		t.Fatal("arch")
	}
	if isQwen3Family(ShowInfo{Parser: "llama", Architecture: "llama"}) {
		t.Fatal("llama should not match")
	}
}

func TestStaticThinkFinding(t *testing.T) {
	tmpl := `<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
`
	if templateHasThinkToggle(tmpl) {
		t.Fatal("stock milkey-like tmpl should lack toggles")
	}
	if !looksChatML(tmpl) {
		t.Fatal("expected chatml")
	}
}

func TestDiagnoseStaticThinkWithFakeAPI(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:         "milkey:latest",
			Template:     "<|im_start|>user\n{{ .Content }}<|im_end|>\n<|im_start|>assistant\n{{ .Response }}",
			Parser:       "qwen3-thinking",
			Capabilities: []string{"completion", "thinking"},
			Parameters:   "temperature                    0.6\nstop                           \"<|im_end|>\"\nstop                           \"<|im_start|>\"\n",
		},
	}
	rep, err := Diagnose(t.Context(), api, "milkey:latest", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	got := map[RecipeID]bool{}
	for _, f := range rep.Findings {
		got[f.Recipe] = true
	}
	// PARSER thinking without toggles → ThinkParserMismatch (not duplicate ThinkGenerateEmpty).
	if !got[RecipeThinkParserMismatch] && !got[RecipeThinkGenerateEmpty] {
		t.Fatalf("findings=%+v", rep.Findings)
	}
	if rep.Patch == nil || rep.Patch.Parser != "qwen3" {
		t.Fatalf("patch=%+v", rep.Patch)
	}
}

func TestDiagnoseRefusesInvasiveNonQwen(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:         "other:latest",
			Template:     "<|im_start|>user\n{{ .Content }}<|im_end|>\n<|im_start|>assistant\n{{ .Response }}",
			Parser:       "llama",
			Architecture: "llama",
			Capabilities: []string{"thinking"},
			Parameters:   "stop                           \"<|im_end|>\"\nstop                           \"<|im_start|>\"\n",
		},
	}
	rep, err := Diagnose(t.Context(), api, "other:latest", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range rep.Findings {
		if f.Recipe == RecipeThinkGenerateEmpty || f.Recipe == RecipeThinkParserMismatch ||
			f.Recipe == RecipeEmptyTemplate || f.Recipe == RecipeSlashSystemCollapse {
			t.Fatalf("invasive recipe must not auto-apply on non-qwen3: %+v", f)
		}
	}
	if len(rep.ManualReview) == 0 {
		t.Fatal("expected manual-review note for think symptoms")
	}
}

func TestDiagnoseRendererSkipsResponsePlaceholder(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:       "qwen3.8:27b",
			Template:   "{% for message in messages %}{{ message.content }}{% endfor %}",
			Parser:     "qwen3.5",
			Renderer:   "qwen3.8",
			Modelfile:  "RENDERER qwen3.8\nPARSER qwen3.5\n",
			Parameters: "temperature                    0.7\n",
		},
	}
	rep, err := Diagnose(t.Context(), api, "qwen3.8:27b", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.ManualReview) != 0 {
		t.Fatalf("renderer models must not flag {{ .Response }}: %v", rep.ManualReview)
	}
	for _, f := range rep.Findings {
		if f.Recipe == RecipeMissingResponsePlaceholder {
			t.Fatalf("unexpected finding: %+v", f)
		}
	}
}

func TestBuildPatchStopsOnly(t *testing.T) {
	show := ShowInfo{
		Template:   "<|im_start|>user\n{{ .Content }}<|im_end|>\n{{ .Response }}",
		Parameters: "temperature                    0.6\n",
		Parser:     "llama",
	}
	p, err := BuildPatch("m:latest", show, []Finding{{Recipe: RecipeChatMLMissingStops}})
	if err != nil {
		t.Fatal(err)
	}
	if p.Template != "" || p.Parser != "" {
		t.Fatalf("stops-only must inherit TEMPLATE/PARSER, got tmpl=%q parser=%q", p.Template, p.Parser)
	}
	stops, _ := p.Parameters["stop"].([]any)
	got := map[string]bool{}
	for _, s := range stops {
		got[fmt.Sprint(s)] = true
	}
	if !got["<|im_end|>"] || !got["<|im_start|>"] {
		t.Fatalf("stops=%v", stops)
	}
}

func TestBuildPatchEmptyTemplate(t *testing.T) {
	p, err := BuildPatch("m:latest", ShowInfo{Parser: "qwen3"}, []Finding{{Recipe: RecipeEmptyTemplate}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(p.Template, "<|im_start|>") || !strings.Contains(p.Template, ".Response") {
		t.Fatalf("expected stock ChatML, got %q", p.Template[:min(80, len(p.Template))])
	}
}

func TestBuildPatchMissingResponseAppend(t *testing.T) {
	show := ShowInfo{
		Template: "<|im_start|>user\n{{ .Content }}<|im_end|>\n<|im_start|>assistant\n",
		Parser:   "qwen3",
	}
	p, err := BuildPatch("m:latest", show, []Finding{{Recipe: RecipeMissingResponsePlaceholder}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(p.Template, ".Response") {
		t.Fatal("expected Response append")
	}
	if !strings.Contains(p.Template, "{{ .Content }}") {
		t.Fatal("should keep original Messages layout")
	}
}

func TestDiagnoseHygieneStopsAndEmpty(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:       "chatml:latest",
			Template:   "<|im_start|>user\n{{ .Content }}<|im_end|>\n<|im_start|>assistant\n{{ .Response }}",
			Parser:     "llama",
			Parameters: "temperature                    0.7\n",
		},
	}
	rep, err := Diagnose(t.Context(), api, "chatml:latest", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	got := map[RecipeID]bool{}
	for _, f := range rep.Findings {
		got[f.Recipe] = true
	}
	if !got[RecipeChatMLMissingStops] {
		t.Fatalf("expected chatml_missing_stops, findings=%+v", rep.Findings)
	}
	if rep.Patch == nil {
		t.Fatal("expected patch")
	}
}

func TestDiagnoseEmptyTemplateQwen(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:         "empty:latest",
			Template:     "",
			Parser:       "qwen3",
			Architecture: "qwen3",
			Capabilities: []string{"completion"},
		},
	}
	rep, err := Diagnose(t.Context(), api, "empty:latest", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 1 || rep.Findings[0].Recipe != RecipeEmptyTemplate {
		t.Fatalf("findings=%+v", rep.Findings)
	}
}

func TestDiagnoseThinkParserMismatch(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:       "mismatch:latest",
			Template:   "<|im_start|>user\n{{ .Content }}<|im_end|>\n{{ .Response }}",
			Parser:     "qwen3-thinking",
			Parameters: "stop \"<|im_end|>\"\nstop \"<|im_start|>\"\n",
		},
	}
	rep, err := Diagnose(t.Context(), api, "mismatch:latest", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	got := map[RecipeID]bool{}
	for _, f := range rep.Findings {
		got[f.Recipe] = true
	}
	if !got[RecipeThinkParserMismatch] {
		t.Fatalf("expected think_parser_mismatch, got %+v", rep.Findings)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestDiagnoseLiveThinkAndSlash(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:         "broken:latest",
			Template:     "<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .Content }}",
			Parser:       "qwen3-thinking",
			Capabilities: []string{"thinking"},
		},
		genDefault: &GenerateResult{Thinking: "<answer>42</answer>", EvalCount: 9},
		genOff:     &GenerateResult{Response: "<answer>42</answer>", EvalCount: 9},
		chatUser:   &ChatResult{Content: "<answer>42</answer>", EvalCount: 9},
		chatSys:    &ChatResult{Content: "///////////////////////////////", EvalCount: 48},
		genRole:    &GenerateResult{Response: "///", EvalCount: 3},
	}
	rep, err := Diagnose(t.Context(), api, "broken:latest", Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := map[RecipeID]bool{}
	for _, f := range rep.Findings {
		got[f.Recipe] = true
	}
	if !got[RecipeThinkGenerateEmpty] || !got[RecipeSlashSystemCollapse] {
		t.Fatalf("findings=%+v", rep.Findings)
	}
	if len(rep.Unfixable) == 0 {
		t.Fatal("expected unfixable roleplay note")
	}
}

type fakeAPI struct {
	show       *ShowInfo
	genDefault *GenerateResult
	genOff     *GenerateResult
	chatUser   *ChatResult
	chatSys    *ChatResult
	genRole    *GenerateResult
	created    bool
}

func (f *fakeAPI) Show(ctx context.Context, name string) (*ShowInfo, error) {
	return f.show, nil
}

func (f *fakeAPI) Generate(ctx context.Context, name string, prompt string, think *bool, opts map[string]any) (*GenerateResult, error) {
	if strings.Contains(prompt, "System:") {
		return f.genRole, nil
	}
	if think != nil && !*think {
		return f.genOff, nil
	}
	return f.genDefault, nil
}

func (f *fakeAPI) Chat(ctx context.Context, name string, messages []map[string]string, opts map[string]any) (*ChatResult, error) {
	for _, m := range messages {
		if m["role"] == "system" {
			return f.chatSys, nil
		}
	}
	return f.chatUser, nil
}

func (f *fakeAPI) Create(ctx context.Context, name, from, template, parser string, params map[string]any) error {
	f.created = true
	return nil
}

func (f *fakeAPI) ListRunning(ctx context.Context) ([]string, error) {
	return []string{"broken:latest"}, nil
}

func (f *fakeAPI) ListLocal(ctx context.Context) ([]string, error) {
	return []string{"broken:latest", "other:latest"}, nil
}

func (f *fakeAPI) Unload(ctx context.Context, name string) error { return nil }
