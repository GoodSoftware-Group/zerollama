package server

import (
	"context"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/freetokenlab"
)

// ChatCompressLabRow is one policy's estimated KV reuse vs re-prefill (no inference).
type ChatCompressLabRow struct {
	Name       string `json:"name"`
	Original   int    `json:"original_tokens"`
	Compressed int    `json:"compressed_tokens"`
	Reuse      int    `json:"prefix_reuse_tokens"`
	Recompute  int    `json:"recompute_tokens"`
	Mode       string `json:"mode,omitempty"`
}

// ChatCompressLabCompare runs none / placeholder / summary plus FreeToken
// suffix-strip vs sparse-checkpoint on a synthetic agent thread. No model load.
func ChatCompressLabCompare(numCtx int) ([]ChatCompressLabRow, error) {
	if numCtx <= 0 {
		numCtx = 4096
	}
	msgs := agentLabThread()
	orig := estimateMessagesTokens(msgs)
	on := true
	ctx := context.Background()

	none := ChatCompressLabRow{
		Name: "none", Original: orig, Compressed: orig, Reuse: 0, Recompute: orig, Mode: "off",
	}

	phPolicy := api.ChatCompressionConfig{Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01}
	_, phMeta, err := compressChatMessages(ctx, phPolicy, numCtx, "lab", msgs, nil)
	if err != nil {
		return nil, fmt.Errorf("placeholder: %w", err)
	}
	ph := rowFromMeta("placeholder", orig, phMeta)

	sumPolicy := api.ChatCompressionConfig{Enabled: &on, Mode: "summary", TriggerAtRatio: 0.01, KeepTailTokens: 256, MaxSummaryTokens: 128}
	fake := func(_ context.Context, _ string, _ []api.Message, maxTokens int) (string, int, error) {
		if maxTokens <= 0 {
			maxTokens = 128
		}
		return strings.Repeat("old tool rounds. ", 8), maxTokens / 4, nil
	}
	_, sumMeta, err := compressChatMessages(ctx, sumPolicy, numCtx, "lab", msgs, fake)
	if err != nil {
		return nil, fmt.Errorf("summary: %w", err)
	}
	sum := rowFromMeta("summary", orig, sumMeta)

	const newSuffix = 400
	ed := freetokenlab.SuffixStripEdit(ph.Reuse, newSuffix)
	strip := ChatCompressLabRow{
		Name: "suffix-strip+anchor", Original: orig, Compressed: ph.Reuse + newSuffix,
		Reuse: ph.Reuse, Recompute: freetokenlab.PrefillTokensWithSemanticAnchor(ed), Mode: "freetoken",
	}
	sparse := ChatCompressLabRow{
		Name: "suffix-strip+ckpt@4k", Original: orig, Compressed: ph.Reuse + newSuffix,
		Reuse: ph.Reuse, Recompute: freetokenlab.PrefillTokensWithoutAnchor(ed, 4096), Mode: "freetoken",
	}
	return []ChatCompressLabRow{none, ph, sum, strip, sparse}, nil
}

// ChatCompressLabSummary is one doctor/CLI line for the agent-thread prefill lab.
func ChatCompressLabSummary() string {
	rows, err := ChatCompressLabCompare(4096)
	if err != nil {
		return "agent KV lab: " + err.Error()
	}
	by := map[string]ChatCompressLabRow{}
	for _, r := range rows {
		by[r.Name] = r
	}
	n, ph, sum, strip := by["none"], by["placeholder"], by["summary"], by["suffix-strip+anchor"]
	return fmt.Sprintf("agent KV lab: none recompute=%d; placeholder reuse=%d recompute=%d; summary reuse=%d recompute=%d; suffix-strip+anchor=%d; sticky=prompt_cache_key",
		n.Recompute, ph.Reuse, ph.Recompute, sum.Reuse, sum.Recompute, strip.Recompute)
}

func rowFromMeta(name string, orig int, meta *api.ChatCompressionMeta) ChatCompressLabRow {
	row := ChatCompressLabRow{Name: name, Original: orig, Mode: name}
	if meta == nil {
		row.Compressed = orig
		row.Recompute = orig
		return row
	}
	row.Compressed = meta.CompressedTokens
	row.Reuse = meta.PrefixReuseTokens
	row.Recompute = meta.RecomputeTokens
	row.Mode = meta.Mode
	return row
}

func agentLabThread() []api.Message {
	tool := strings.Repeat("retrieved document chunk for the agent.\n", 80)
	think := strings.Repeat("planning the next tool call. ", 40)
	return []api.Message{
		{Role: "system", Content: "You are a local coding agent."},
		{Role: "user", Content: "Inspect the repo and fix the flaky test."},
		{Role: "assistant", Content: "I'll search.", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "grep"}}}},
		{Role: "tool", Content: tool, ToolName: "grep"},
		{Role: "assistant", Content: "Reading files.", Thinking: think, ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "read"}}}},
		{Role: "tool", Content: tool, ToolName: "read"},
		{Role: "tool", Content: tool, ToolName: "read"},
		{Role: "assistant", Content: "Running tests.", Thinking: think, ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "shell"}}}},
		{Role: "tool", Content: tool + tool, ToolName: "shell"},
		{Role: "user", Content: "Also check the other package."},
	}
}
