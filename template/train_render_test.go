package template

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestRenderTrainChatML(t *testing.T) {
	raw, err := templatesFS.ReadFile("chatml.gotmpl")
	if err != nil {
		t.Fatal(err)
	}
	out, err := RenderTrain(string(raw), []api.Message{
		{Role: "system", Content: "Be nice"},
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Content: "Hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "<|im_start|>system\nBe nice<|im_end|>") {
		t.Fatalf("missing system turn: %q", out)
	}
	if !strings.Contains(out, "<|im_start|>assistant\nHello<|im_end|>") {
		t.Fatalf("missing filled assistant: %q", out)
	}
	if strings.HasSuffix(strings.TrimRight(out, "\n"), "<|im_start|>assistant") {
		t.Fatalf("left open generation prompt: %q", out)
	}
	// Inference render still primes assistant
	tmpl, err := Parse(string(raw))
	if err != nil {
		t.Fatal(err)
	}
	var infer strings.Builder
	if err := tmpl.Execute(&infer, Values{Messages: []api.Message{
		{Role: "user", Content: "Hi"},
	}}); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(infer.String(), "<|im_start|>assistant\n") &&
		!strings.HasSuffix(infer.String(), "<|im_start|>assistant") {
		t.Fatalf("inference should still prime assistant, got %q", infer.String())
	}
}

func TestRenderTrainLlama3(t *testing.T) {
	raw, err := templatesFS.ReadFile("llama3-instruct.gotmpl")
	if err != nil {
		t.Fatal(err)
	}
	out, err := RenderTrain(string(raw), []api.Message{
		{Role: "user", Content: "Q"},
		{Role: "assistant", Content: "A"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Q<|eot_id|>") || !strings.Contains(out, "A<|eot_id|>") {
		t.Fatalf("missing turns: %q", out)
	}
	if strings.Contains(out, "<|start_header_id|>assistant<|end_header_id|>\n\n<|") {
		// empty primed assistant after content would look like header then another token start
	}
	if strings.HasSuffix(strings.TrimRight(out, "\n"), "<|end_header_id|>") {
		t.Fatalf("left open llama3 assistant header: %q", out)
	}
}

func TestStripTrainGenerationPrompt(t *testing.T) {
	in := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n<|im_start|>assistant\n"
	got := stripTrainGenerationPrompt(in)
	want := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
