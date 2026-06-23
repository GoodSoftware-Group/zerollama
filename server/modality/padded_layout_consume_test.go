package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPaddedLayoutConsumePlan_qwen3vlHF(t *testing.T) {
	t.Parallel()
	msgs := []api.Message{{
		Role:           "user",
		Content:        "clip",
		Images:         []api.ImageData{{1}, {2}},
		VideoSpans:     []api.VideoSpan{{FrameCount: 2}},
		PaddedInputIDs: []int{1, 2, 3},
	}}
	plan := PaddedLayoutConsumePlanForChat("qwen3-vl-instruct", nil, msgs, false)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeQwen3VLHF {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestPaddedLayoutConsumePlan_productionImgTagsRunnerInject(t *testing.T) {
	t.Parallel()
	msgs := []api.Message{{
		Role:           "user",
		PaddedInputIDs: []int{9, 8},
		Images:         []api.ImageData{{1}},
	}}
	plan := PaddedLayoutConsumePlanForChat("qwen3-vl-instruct", []string{"qwen3vl"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeQwen3VLHF {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestPaddedLayoutConsumePlan_multimodalHistoryHF(t *testing.T) {
	t.Parallel()
	msgs := []api.Message{
		{Role: "user", Images: []api.ImageData{{1}}, VideoSpans: []api.VideoSpan{{FrameCount: 1}}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", PaddedInputIDs: []int{5}, Images: []api.ImageData{{2}}},
	}
	plan := PaddedLayoutConsumePlanForChat("qwen3-vl-instruct", nil, msgs, false)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeQwen3VLHF {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestPaddedLayoutConsumePlan_dualPaddedRunnerInject(t *testing.T) {
	t.Parallel()
	msgs := []api.Message{
		{Role: "user", PaddedInputIDs: []int{1, 2}, Images: []api.ImageData{{1}}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", PaddedInputIDs: []int{5}, Images: []api.ImageData{{2}}},
	}
	plan := PaddedLayoutConsumePlanForChat("qwen3-vl-instruct", nil, msgs, false)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeQwen3VLHF {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestMessageSkipsVisionPlaceholders(t *testing.T) {
	t.Parallel()
	msg := api.Message{PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}
	if !MessageSkipsVisionPlaceholders(msg, false) {
		t.Fatal("expected skip in HF mode")
	}
	if !MessageSkipsVisionPlaceholders(msg, true) {
		t.Fatal("production img tags should skip when padded_input_ids set")
	}
}

func TestMessageSkipsVisionPlaceholdersForChat_priorWithTools(t *testing.T) {
	t.Parallel()
	msgs := []api.Message{
		{Role: "user", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}},
		{Role: "assistant", Content: "ok"},
		{Role: "tool", Content: "data"},
		{Role: "user", PaddedInputIDs: []int{2}, Images: []api.ImageData{{2}}},
	}
	if MessageSkipsVisionPlaceholdersForChat(msgs, 0, false) {
		t.Fatal("prior padded user should keep placeholders when tool messages exist")
	}
	if !MessageSkipsVisionPlaceholdersForChat(msgs, 3, false) {
		t.Fatal("latest padded user should skip placeholders")
	}
}
