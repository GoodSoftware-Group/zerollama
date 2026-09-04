package server

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestCompressChatSkippedBelowThreshold(t *testing.T) {
	on := true
	msgs := []api.Message{{Role: "user", Content: "hi"}}
	got, meta, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{Enabled: &on, TriggerAtRatio: 0.75}, 1000, "m", msgs, func(context.Context, string, []api.Message, int) (string, int, error) {
		t.Fatal("summarizer should not run")
		return "", 0, nil
	})
	if err != nil || meta != nil || len(got) != 1 {
		t.Fatalf("err=%v meta=%v got=%d", err, meta, len(got))
	}
}

func TestCompressChatHeadKeepsTail(t *testing.T) {
	on := true
	var saw []api.Message
	msgs := []api.Message{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: strings.Repeat("old question ", 40)},
		{Role: "assistant", Content: strings.Repeat("old answer ", 40)},
		{Role: "user", Content: "latest"},
	}
	got, meta, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, TriggerAtRatio: 0.01, KeepTailTokens: 8, MaxSummaryTokens: 16, CompressorModel: "fast",
	}, 200, "primary", msgs, func(_ context.Context, model string, head []api.Message, maxTokens int) (string, int, error) {
		if model != "fast" || maxTokens != 16 {
			t.Fatalf("model=%s max=%d", model, maxTokens)
		}
		saw = append([]api.Message(nil), head...)
		return "facts retained", 2, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(saw) == 0 {
		t.Fatal("expected head")
	}
	if len(got) < 3 || got[0].Content != "rules" {
		t.Fatalf("got=%+v", got)
	}
	if !strings.HasPrefix(got[1].Content, chatCompressionSummaryPrefix) {
		t.Fatalf("summary %q", got[1].Content)
	}
	if got[len(got)-1].Content != "latest" {
		t.Fatalf("tail %+v", got[len(got)-1])
	}
	if meta == nil || meta.DroppedTurns == 0 || meta.Compressor != "fast" {
		t.Fatalf("meta=%+v", meta)
	}
	if meta.PrefixReuseTokens <= 0 {
		t.Fatalf("expected system prefix reuse, meta=%+v", meta)
	}
	if meta.RecomputeTokens <= 0 || meta.PrefixReuseTokens+meta.RecomputeTokens != meta.CompressedTokens {
		t.Fatalf("reuse+recompute should cover compressed prompt: %+v", meta)
	}
}

func TestCompressChatOverflowError(t *testing.T) {
	on := true
	msgs := []api.Message{
		{Role: "user", Content: strings.Repeat("x", 80)},
		{Role: "assistant", Content: strings.Repeat("y", 80)},
		{Role: "user", Content: "now"},
	}
	_, _, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, TriggerAtRatio: 0.01, KeepTailTokens: 4, MaxSummaryTokens: 4,
		OnPostCompressionOverflow: "error",
	}, 20, "m", msgs, func(context.Context, string, []api.Message, int) (string, int, error) {
		return strings.Repeat("summary remains much too large ", 20), 40, nil
	})
	if !isChatCompressOverflow(err) {
		t.Fatalf("err=%v", err)
	}
}

func TestCompressChatDropOldestSummary(t *testing.T) {
	on := true
	msgs := []api.Message{
		{Role: "system", Content: chatCompressionSummaryPrefix + "old]"},
		{Role: "user", Content: strings.Repeat("q ", 30)},
		{Role: "assistant", Content: strings.Repeat("a ", 30)},
		{Role: "user", Content: "now"},
	}
	got, meta, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, TriggerAtRatio: 0.01, KeepTailTokens: 8,
		OnPostCompressionOverflow: "drop_oldest_summary",
	}, 8, "m", msgs, func(context.Context, string, []api.Message, int) (string, int, error) {
		return "tiny", 1, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if meta == nil {
		t.Fatal("meta")
	}
	for _, m := range got {
		if m.Content == chatCompressionSummaryPrefix+"old]" {
			t.Fatalf("old summary still present: %+v", got)
		}
	}
}

func TestResolveChatCompressionRequestOffWins(t *testing.T) {
	off := false
	req := &api.ChatRequest{Compression: &api.ChatCompressionConfig{Enabled: &off}}
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "1")
	cfg := resolveChatCompression(req)
	if chatCompressionEnabled(cfg) {
		t.Fatal("request false should win")
	}
}

func TestResolveAutoPlaceholderForToolThread(t *testing.T) {
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION_MODE", "")
	req := &api.ChatRequest{Messages: []api.Message{
		{Role: "user", Content: "look this up"},
		{Role: "tool", Content: strings.Repeat("x", 200)},
	}}
	cfg := resolveChatCompression(req)
	if !chatCompressionEnabled(cfg) {
		t.Fatal("agent tool threads should auto-enable placeholder")
	}
	if chatCompressionMode(cfg) != "placeholder" {
		t.Fatalf("mode=%q", cfg.Mode)
	}
}

func TestResolvePlainChatNoAutoCompress(t *testing.T) {
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION_MODE", "")
	req := &api.ChatRequest{Messages: []api.Message{{Role: "user", Content: "hi"}}}
	cfg := resolveChatCompression(req)
	if chatCompressionEnabled(cfg) {
		t.Fatal("plain chat must stay off without env")
	}
}

func TestApplyChatCompressionAutoPlaceholder(t *testing.T) {
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "")
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION_MODE", "")
	req := &api.ChatRequest{
		Options: map[string]any{"num_ctx": 40},
		Messages: []api.Message{
			{Role: "user", Content: "q"},
			{Role: "assistant", Content: "call", Thinking: strings.Repeat("t", 80)},
			{Role: "tool", Content: strings.Repeat("tool-bytes ", 40), ToolCallID: "c1"},
			{Role: "user", Content: "latest"},
		},
	}
	got, meta, err := applyChatCompressionForRequest(t.Context(), req, req.Messages, 40, "m", 0, func(context.Context, string, []api.Message, int) (string, int, error) {
		t.Fatal("auto placeholder must not summarize")
		return "", 0, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if meta == nil || meta.Mode != "placeholder" {
		t.Fatalf("meta=%+v", meta)
	}
	found := false
	for _, m := range got {
		if m.Role == "tool" && m.Content == chatCompressionToolPlaceholder {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected elided tool: %+v", got)
	}
}

func TestPartitionKeepsToolChain(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: strings.Repeat("old ", 50)},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "lookup"}}}},
		{Role: "tool", Content: "result", ToolCallID: "c1"},
		{Role: "user", Content: "latest"},
	}
	_, tail := partitionChatTail(msgs, 4)
	if len(tail) < 1 || tail[len(tail)-1].Content != "latest" {
		t.Fatalf("tail=%+v", tail)
	}
}

func TestCompressPlaceholderElidesToolKeepsPrefix(t *testing.T) {
	on := true
	longTool := strings.Repeat("tool-bytes ", 80)
	msgs := []api.Message{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: "old question"},
		{Role: "assistant", Content: "calling", Thinking: strings.Repeat("scratch ", 40)},
		{Role: "tool", Content: longTool, ToolCallID: "c1"},
		{Role: "user", Content: "latest"},
	}
	got, meta, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01, KeepTailTokens: 4,
	}, 80, "primary", msgs, func(context.Context, string, []api.Message, int) (string, int, error) {
		t.Fatal("placeholder mode must not call summarizer")
		return "", 0, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if meta == nil || meta.Mode != "placeholder" || meta.Compressor != "placeholder" {
		t.Fatalf("meta=%+v", meta)
	}
	if got[len(got)-1].Content != "latest" {
		t.Fatalf("tail %+v", got[len(got)-1])
	}
	foundElide := false
	for _, m := range got {
		if m.Role == "tool" && m.Content == chatCompressionToolPlaceholder {
			foundElide = true
		}
	}
	if !foundElide {
		t.Fatalf("expected elided tool: %+v", got)
	}
	if meta.PrefixReuseTokens < estimateMessageTokens(msgs[0]) {
		t.Fatalf("reuse should include system prefix: %+v", meta)
	}
	if meta.CompressedTokens >= meta.OriginalTokens {
		t.Fatalf("elide should shrink: %+v", meta)
	}
	if meta.PrefixReuseTokens+meta.RecomputeTokens != meta.CompressedTokens {
		t.Fatalf("reuse+recompute=%d+%d compressed=%d", meta.PrefixReuseTokens, meta.RecomputeTokens, meta.CompressedTokens)
	}
}

func TestCompressPlaceholderElidesNewestKeepsEarlyTool(t *testing.T) {
	on := true
	early := strings.Repeat("early-tool ", 20)
	late := strings.Repeat("late-tool-bytes ", 200)
	msgs := []api.Message{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: "q"},
		{Role: "tool", Content: early, ToolName: "grep"},
		{Role: "assistant", Content: "next"},
		{Role: "tool", Content: late, ToolName: "read"},
		{Role: "user", Content: "latest"},
	}
	got, meta, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01,
	}, 200, "primary", msgs, nil)
	if err != nil {
		t.Fatal(err)
	}
	if meta == nil {
		t.Fatal("expected compression")
	}
	if got[2].Content != early {
		t.Fatalf("early tool should stay exact for prefix KV, got %q", got[2].Content)
	}
	if got[4].Content != chatCompressionToolPlaceholder {
		t.Fatalf("newest fat tool should elide first: %q", got[4].Content)
	}
	sysUser := estimateMessageTokens(msgs[0]) + estimateMessageTokens(msgs[1]) + estimateMessageTokens(msgs[2])
	if meta.PrefixReuseTokens < sysUser {
		t.Fatalf("reuse=%d want >= %d (system+user+early tool)", meta.PrefixReuseTokens, sysUser)
	}
	if meta.ElideFrom != 4 {
		t.Fatalf("elide_from=%d want 4 (newest fat tool)", meta.ElideFrom)
	}
}

func TestCompressPlaceholderStickyElideFromKeepsPriorCut(t *testing.T) {
	on := true
	early := strings.Repeat("early-tool ", 20)
	late := strings.Repeat("late-tool-bytes ", 200)
	newer := strings.Repeat("newer-tool-bytes ", 200)
	turn1 := []api.Message{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: "q"},
		{Role: "tool", Content: early, ToolName: "grep"},
		{Role: "assistant", Content: "next"},
		{Role: "tool", Content: late, ToolName: "read"},
		{Role: "user", Content: "latest"},
	}
	got1, meta1, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01,
	}, 200, "primary", turn1, nil)
	if err != nil || meta1 == nil {
		t.Fatalf("turn1: %v meta=%+v", err, meta1)
	}
	turn2 := append(append([]api.Message(nil), turn1[:len(turn1)-1]...),
		api.Message{Role: "assistant", Content: "more"},
		api.Message{Role: "tool", Content: newer, ToolName: "shell"},
		api.Message{Role: "user", Content: "follow-up"},
	)
	loose, metaLoose, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01,
	}, 5000, "primary", turn2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if metaLoose != nil && loose[4].Content == chatCompressionToolPlaceholder {
		t.Fatal("roomy follow-up without sticky should restore the old tool body")
	}
	if loose[4].Content != late {
		t.Fatalf("roomy follow-up should keep full late tool, got %q", loose[4].Content)
	}
	sticky := meta1.ElideFrom
	got2, meta2, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01, ElideFrom: &sticky,
	}, 5000, "primary", turn2, nil)
	if err != nil || meta2 == nil {
		t.Fatalf("turn2 sticky: %v meta=%+v", err, meta2)
	}
	if got2[4].Content != chatCompressionToolPlaceholder {
		t.Fatalf("sticky must keep prior cut: %q", got2[4].Content)
	}
	reuse := longestExactPrefixTokens(got1, got2)
	if reuse < estimateMessageTokens(got1[0])+estimateMessageTokens(got1[1])+estimateMessageTokens(got1[2])+estimateMessageTokens(got1[3]) {
		t.Fatalf("sticky follow-up should share prefix through early turns, reuse=%d got1=%+v got2=%+v", reuse, got1, got2)
	}
}

func TestCompressPlaceholderPeelsOldestWhenStillOver(t *testing.T) {
	on := true
	msgs := []api.Message{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: strings.Repeat("old-a ", 40)},
		{Role: "assistant", Content: strings.Repeat("old-b ", 40)},
		{Role: "tool", Content: strings.Repeat("tool-bytes ", 40), ToolCallID: "c1"},
		{Role: "user", Content: "latest"},
	}
	got, meta, err := compressChatMessages(t.Context(), api.ChatCompressionConfig{
		Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01,
	}, 16, "primary", msgs, func(context.Context, string, []api.Message, int) (string, int, error) {
		t.Fatal("placeholder must not summarize")
		return "", 0, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if meta == nil || estimateMessagesTokens(got) > 16 {
		t.Fatalf("still over ctx: tokens=%d meta=%+v got=%+v", estimateMessagesTokens(got), meta, got)
	}
	if got[0].Content != "rules" || got[len(got)-1].Content != "latest" {
		t.Fatalf("must keep system + last turn: %+v", got)
	}
	for _, m := range got {
		if strings.Contains(m.Content, "old-a") {
			t.Fatalf("expected oldest user peeled: %+v", got)
		}
	}
}

func TestApplyChatCompressionStickyByPromptCacheKey(t *testing.T) {
	resetStickyElideForTest()
	t.Cleanup(resetStickyElideForTest)
	on := true
	early := strings.Repeat("early-tool ", 20)
	late := strings.Repeat("late-tool-bytes ", 200)
	newer := strings.Repeat("newer-tool-bytes ", 200)
	turn1 := []api.Message{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: "q"},
		{Role: "tool", Content: early, ToolName: "grep"},
		{Role: "assistant", Content: "next"},
		{Role: "tool", Content: late, ToolName: "read"},
		{Role: "user", Content: "latest"},
	}
	req1 := &api.ChatRequest{
		Model:       "m",
		Messages:    turn1,
		Options:     map[string]any{"prompt_cache_key": "hermes:agent:test:1"},
		Compression: &api.ChatCompressionConfig{Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01},
	}
	got1, meta1, err := applyChatCompressionForRequest(t.Context(), req1, turn1, 200, "m", 0, nil)
	if err != nil || meta1 == nil {
		t.Fatalf("turn1: %v meta=%+v", err, meta1)
	}
	turn2 := append(append([]api.Message(nil), turn1[:len(turn1)-1]...),
		api.Message{Role: "assistant", Content: "more"},
		api.Message{Role: "tool", Content: newer, ToolName: "shell"},
		api.Message{Role: "user", Content: "follow-up"},
	)
	req2 := &api.ChatRequest{
		Model:       "m",
		Messages:    turn2,
		Options:     map[string]any{"prompt_cache_key": "hermes:agent:test:1"},
		Compression: &api.ChatCompressionConfig{Enabled: &on, Mode: "placeholder"},
	}
	got2, meta2, err := applyChatCompressionForRequest(t.Context(), req2, turn2, 5000, "m", 0, nil)
	if err != nil || meta2 == nil {
		t.Fatalf("turn2: %v meta=%+v", err, meta2)
	}
	if got2[4].Content != chatCompressionToolPlaceholder {
		t.Fatalf("server sticky by prompt_cache_key must keep prior cut: %q", got2[4].Content)
	}
	_ = got1
}

func TestApplyChatCompressionStickyRequestRelativeOrigin(t *testing.T) {
	resetStickyElideForTest()
	t.Cleanup(resetStickyElideForTest)
	on := true
	late := strings.Repeat("late-tool-bytes ", 200)
	newer := strings.Repeat("newer-tool-bytes ", 200)
	reqMsgs := []api.Message{
		{Role: "user", Content: "q"},
		{Role: "tool", Content: strings.Repeat("early-tool ", 20), ToolName: "grep"},
		{Role: "assistant", Content: "next"},
		{Role: "tool", Content: late, ToolName: "read"},
		{Role: "user", Content: "latest"},
	}
	assembled := append([]api.Message{{Role: "system", Content: "modelfile"}}, reqMsgs...)
	req1 := &api.ChatRequest{
		Model:       "m",
		Messages:    reqMsgs,
		Options:     map[string]any{"prompt_cache_key": "hermes:origin:1"},
		Compression: &api.ChatCompressionConfig{Enabled: &on, Mode: "placeholder", TriggerAtRatio: 0.01},
	}
	_, meta1, err := applyChatCompressionForRequest(t.Context(), req1, assembled, 200, "m", 1, nil)
	if err != nil || meta1 == nil {
		t.Fatalf("turn1: %v meta=%+v", err, meta1)
	}
	if meta1.ElideFrom != 3 {
		t.Fatalf("elide_from=%d want 3 (request-relative, not assembled 4)", meta1.ElideFrom)
	}
	turn2req := append(append([]api.Message(nil), reqMsgs[:len(reqMsgs)-1]...),
		api.Message{Role: "assistant", Content: "more"},
		api.Message{Role: "tool", Content: newer, ToolName: "shell"},
		api.Message{Role: "user", Content: "follow-up"},
	)
	assembled2 := append([]api.Message{{Role: "system", Content: "modelfile"}}, turn2req...)
	req2 := &api.ChatRequest{
		Model:       "m",
		Messages:    turn2req,
		Options:     map[string]any{"prompt_cache_key": "hermes:origin:1"},
		Compression: &api.ChatCompressionConfig{Enabled: &on, Mode: "placeholder"},
	}
	got2, meta2, err := applyChatCompressionForRequest(t.Context(), req2, assembled2, 5000, "m", 1, nil)
	if err != nil || meta2 == nil {
		t.Fatalf("turn2: %v meta=%+v", err, meta2)
	}
	if got2[4].Content != chatCompressionToolPlaceholder {
		t.Fatalf("assembled index 4 (request 3) should stay elided: %q", got2[4].Content)
	}
}

func TestStickyElideRewindAndCacheReset(t *testing.T) {
	resetStickyElideForTest()
	t.Cleanup(resetStickyElideForTest)
	rememberStickyElide("k", 4, 8)
	if _, ok := lookupStickyElide("k", 3); ok {
		t.Fatal("rewind should drop sticky")
	}
	rememberStickyElide("k", 4, 8)
	if _, ok := lookupStickyElide("k", 10); !ok {
		t.Fatal("fresh remember should hit")
	}
	stickyElideMu.Lock()
	rec := stickyElideByKey["k"]
	rec.at = time.Now().Add(-time.Hour)
	stickyElideByKey["k"] = rec
	stickyElideMu.Unlock()
	if _, ok := lookupStickyElide("k", 10); ok {
		t.Fatal("expired sticky should miss")
	}
}
