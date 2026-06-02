package server

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func TestRenderPromptTokenBudget(t *testing.T) {
	t.Parallel()
	if got := renderPromptTokenBudget(8192, 0); got != 8192-256 {
		t.Fatalf("large ctx: got %d", got)
	}
	if got := renderPromptTokenBudget(512, 0); got != 512-256 {
		t.Fatalf("512 ctx: got %d", got)
	}
	if got := renderPromptTokenBudget(8192, 512); got != 8192-512 {
		t.Fatalf("num_predict reserve: got %d", got)
	}
	if got := renderPromptTokenBudget(200, 0); got != 100 {
		t.Fatalf("tight ctx caps reserve at half: got %d", got)
	}
	if got := renderPromptTokenBudget(128, 8); got != 120 {
		t.Fatalf("small num_predict reserve: got %d", got)
	}
	if got := renderPromptTokenBudget(0, 64); got != 0 {
		t.Fatalf("zero: got %d", got)
	}
}

func TestEstimatePromptTokens(t *testing.T) {
	t.Parallel()
	if estimatePromptTokens("") != 0 {
		t.Fatal("empty")
	}
	if estimatePromptTokens("abcd") != 1 {
		t.Fatalf("short string")
	}
	if estimatePromptTokens(string(make([]byte, 400))) != 100 {
		t.Fatalf("400 bytes")
	}
}

func TestRenderChatPromptTokenizedTruncatesOldMessages(t *testing.T) {
	t.Parallel()
	tmpl, err := template.Parse(`{{- range .Messages }}{{ .Role }}: {{ .Content }}
{{- end }}`)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Template: tmpl}
	msgs := []api.Message{
		{Role: "user", Content: "one two three four five"},
		{Role: "assistant", Content: "six seven eight"},
		{Role: "user", Content: "nine ten"},
	}
	// One token per whitespace-separated word.
	tokenize := func(_ context.Context, s string) ([]int, error) {
		parts := strings.Fields(s)
		out := make([]int, len(parts))
		for i := range parts {
			out[i] = i
		}
		return out, nil
	}
	prompt, dropped, err := renderChatPromptTokenized(
		context.Background(),
		m,
		msgs,
		nil,
		nil,
		tokenize,
		6,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !dropped {
		t.Fatal("expected droppedPrefix when early turns removed")
	}
	if strings.Contains(prompt, "one two three") {
		t.Fatalf("expected early user turn dropped, got %q", prompt)
	}
	if !strings.Contains(prompt, "nine ten") {
		t.Fatalf("expected latest user turn kept, got %q", prompt)
	}
}

func TestRenderChatPromptWithTruncateMode(t *testing.T) {
	t.Parallel()
	tmpl, err := template.Parse(`{{ .Prompt }}`)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Template: tmpl}
	msgs := []api.Message{{Role: "user", Content: "hello"}}
	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{}}}

	_, mode, dropped, _, err := s.renderChatPromptWithTruncate(
		context.Background(), m, msgs, nil, nil, 0, 0, false,
	)
	if err != nil || mode != "none" || dropped {
		t.Fatalf("no truncate: mode=%q dropped=%v err=%v", mode, dropped, err)
	}

	_, mode, dropped, _, err = s.renderChatPromptWithTruncate(
		context.Background(), m, msgs, nil, nil, 8192, 128, true,
	)
	if err != nil || mode != "heuristic" || dropped {
		t.Fatalf("heuristic short prompt: mode=%q dropped=%v err=%v", mode, dropped, err)
	}

	key := schedulerModelKey(m)
	s.sched.loaded[key] = &runnerRef{
		llama: &mockLlm{
			tokenizeResp: []int{1, 2, 3},
		},
		model: m,
	}
	_, mode, dropped, _, err = s.renderChatPromptWithTruncate(
		context.Background(), m, msgs, nil, nil, 8192, 128, true,
	)
	if err != nil || mode != "tokenize" || dropped {
		t.Fatalf("tokenize short prompt: mode=%q dropped=%v err=%v", mode, dropped, err)
	}
}

func TestChatPromptTokenBudgetReservesNumPredict(t *testing.T) {
	t.Parallel()
	if got := chatPromptTokenBudget(&api.Options{Runner: api.Runner{NumCtx: 8192}, NumPredict: 512}); got != 8192-512 {
		t.Fatalf("got %d", got)
	}
	if got := chatPromptTokenBudget(&api.Options{Runner: api.Runner{NumCtx: 8192}, NumPredict: -1}); got != 8192 {
		t.Fatalf("unset num_predict uses full ctx: got %d", got)
	}
}

func TestRenderChatPromptHeuristicDroppedPrefix(t *testing.T) {
	t.Parallel()
	tmpl, err := template.Parse(`{{- range .Messages }}{{ .Content }}|{{- end }}`)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Template: tmpl}
	msgs := []api.Message{
		{Role: "user", Content: strings.Repeat("word ", 200)},
		{Role: "user", Content: "tail"},
	}
	_, dropped, err := renderChatPromptHeuristic(m, msgs, nil, nil, 64, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !dropped {
		t.Fatal("expected droppedPrefix under tight num_ctx")
	}
}
