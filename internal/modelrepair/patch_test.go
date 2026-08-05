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
			Template:     "<|im_start|>user\n{{ .Content }}<|im_end|>\n<|im_start|>assistant\n",
			Parser:       "qwen3-thinking",
			Capabilities: []string{"completion", "thinking"},
			Parameters:   "temperature                    0.6\n",
		},
	}
	rep, err := Diagnose(t.Context(), api, "milkey:latest", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 1 || rep.Findings[0].Recipe != RecipeThinkGenerateEmpty {
		t.Fatalf("findings=%+v", rep.Findings)
	}
	if rep.Patch == nil || rep.Patch.Parser != "qwen3" {
		t.Fatalf("patch=%+v", rep.Patch)
	}
}

func TestDiagnoseRefusesNonQwen(t *testing.T) {
	api := &fakeAPI{
		show: &ShowInfo{
			Name:         "other:latest",
			Template:     "<|im_start|>user\n{{ .Content }}<|im_end|>\n<|im_start|>assistant\n<think>\n",
			Parser:       "llama",
			Architecture: "llama",
			Capabilities: []string{"thinking"},
		},
	}
	rep, err := Diagnose(t.Context(), api, "other:latest", Options{SkipLive: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 0 {
		t.Fatalf("expected no auto findings, got %+v", rep.Findings)
	}
	if len(rep.ManualReview) == 0 {
		t.Fatal("expected manual-review note")
	}
	if rep.Patch != nil {
		t.Fatal("must not propose patch for non-qwen3")
	}
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

func (f *fakeAPI) Unload(ctx context.Context, name string) error { return nil }
