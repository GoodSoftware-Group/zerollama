package modality

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func fakeTokenize(_ context.Context, s string) ([]int, error) {
	out := make([]int, len(s))
	for i := range s {
		out[i] = int(s[i])
	}
	return out, nil
}

func TestBuildPaddedCompletionPromptTokens_splice(t *testing.T) {
	t.Parallel()
	rendered := "<|im_start|>user\n" + "CLIP" + "<|im_end|>\n<|im_start|>assistant\n"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "CLIP",
		PaddedInputIDs: []int{151652, 151655, 151653, 99},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeQwen3VLHF}
	got, ok := BuildPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("<|im_start|>user\n")
	suffixLen := len("<|im_end|>\n<|im_start|>assistant\n")
	wantLen := prefixLen + 4 + suffixLen
	if len(got) != wantLen {
		t.Fatalf("len=%d want %d", len(got), wantLen)
	}
	if got[prefixLen] != 151652 || got[prefixLen+3] != 99 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+4])
	}
}

func TestBuildPaddedCompletionPromptTokens_wrongMode(t *testing.T) {
	t.Parallel()
	_, ok := BuildPaddedCompletionPromptTokens(context.Background(), fakeTokenize, "", nil, PaddedLayoutConsumePlan{Mode: PaddedLayoutConsumeDeferred})
	if ok {
		t.Fatal("expected miss")
	}
}

func TestBuildPaddedCompletionPromptTokens_multimodalHistory(t *testing.T) {
	t.Parallel()
	rendered := "<|im_start|>user\n" + "OLD" + "<|im_end|>\n" +
		"<|im_start|>assistant\nok<|im_end|>\n" +
		"<|im_start|>user\n" + "NEW" + "<|im_end|>\n<|im_start|>assistant\n"
	msgs := []api.Message{
		{Role: "user", Content: "OLD", Images: []api.ImageData{{1}}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "NEW", PaddedInputIDs: []int{151652, 151655, 151653}, Images: []api.ImageData{{2}}},
	}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeQwen3VLHF}
	got, ok := BuildPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	lastUserStart := strings.LastIndex(rendered, qwenVLUserStart) + len(qwenVLUserStart)
	prefixLen := len(rendered[:lastUserStart])
	suffixLen := len(rendered[lastUserStart+len("NEW"):])
	wantLen := prefixLen + 3 + suffixLen
	if len(got) != wantLen {
		t.Fatalf("len=%d want %d", len(got), wantLen)
	}
	if got[prefixLen] != 151652 {
		t.Fatalf("splice at last user: %v", got[prefixLen:prefixLen+3])
	}
}

func TestBuildPaddedCompletionPromptTokens_dualPaddedTurns(t *testing.T) {
	t.Parallel()
	rendered := "<|im_start|>user\n" + "T1" + "<|im_end|>\n" +
		"<|im_start|>assistant\nok<|im_end|>\n" +
		"<|im_start|>user\n" + "T2" + "<|im_end|>\n<|im_start|>assistant\n"
	msgs := []api.Message{
		{Role: "user", Content: "T1", PaddedInputIDs: []int{10, 11}, Images: []api.ImageData{{1}}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "T2", PaddedInputIDs: []int{20, 21, 22}, Images: []api.ImageData{{2}}},
	}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeQwen3VLHF}
	got, ok := BuildPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	// prefix through first user block + padded T1 + middle + padded T2 + suffix
	t1Start := len("<|im_start|>user\n")
	midStart := t1Start + 2 + len("<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n<|im_start|>user\n")
	if got[t1Start] != 10 || got[t1Start+1] != 11 {
		t.Fatalf("turn1 padded at %d: %v", t1Start, got[t1Start:t1Start+2])
	}
	if got[midStart] != 20 || got[midStart+2] != 22 {
		t.Fatalf("turn2 padded at %d: %v", midStart, got[midStart:midStart+3])
	}
}

func TestBuildPaddedCompletionPromptTokens_toolTurnExcluded(t *testing.T) {
	t.Parallel()
	rendered := "<|im_start|>user\n" + "T1" + "<|im_end|>\n" +
		"<|im_start|>assistant\n<tool_call>\n{}</tool_call><|im_end|>\n" +
		"<|im_start|>user\n<tool_response>\nresult\n</tool_response><|im_end|>\n" +
		"<|im_start|>assistant\nok<|im_end|>\n" +
		"<|im_start|>user\n" + "T2" + "<|im_end|>\n<|im_start|>assistant\n"
	msgs := []api.Message{
		{Role: "user", Content: "T1", PaddedInputIDs: []int{10, 11}, Images: []api.ImageData{{1}}},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
		{Role: "tool", Content: "result"},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "T2", PaddedInputIDs: []int{20, 21}, Images: []api.ImageData{{2}}},
	}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeQwen3VLHF}
	got, ok := BuildPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok with tool pseudo-user span excluded")
	}
	t1Start := len("<|im_start|>user\n")
	if got[t1Start] != 10 || got[t1Start+1] != 11 {
		t.Fatalf("turn1 padded at %d: %v", t1Start, got[t1Start:t1Start+2])
	}
	lastUserStart := strings.LastIndex(rendered, qwenVLUserStart) + len(qwenVLUserStart)
	midStart := len(rendered[:lastUserStart])
	if got[midStart] != 20 {
		t.Fatalf("turn2 padded at %d: %v", midStart, got[midStart:midStart+2])
	}
}

func TestQwenVLUserContentSpans_skipsToolResponse(t *testing.T) {
	t.Parallel()
	rendered := "<|im_start|>user\nreal<|im_end|>\n" +
		"<|im_start|>user\n<tool_response>\nx\n</tool_response><|im_end|>\n"
	spans := qwenVLUserContentSpans(rendered)
	if len(spans) != 1 {
		t.Fatalf("spans=%d want 1", len(spans))
	}
	if rendered[spans[0].contentStart:spans[0].contentEnd] != "real" {
		t.Fatalf("span=%q", rendered[spans[0].contentStart:spans[0].contentEnd])
	}
}

func TestQwenVLUserContentBounds(t *testing.T) {
	t.Parallel()
	s := "x<|im_start|>user\nbody<|im_end|>y"
	start, end, ok := qwenVLUserContentBounds(s)
	if !ok || start != len("x<|im_start|>user\n") || end != len("x<|im_start|>user\nbody") {
		t.Fatalf("bounds=%d,%d ok=%v", start, end, ok)
	}
}
