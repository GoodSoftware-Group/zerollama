package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality"
)

const chatCompressionSummaryPrefix = "[COMPRESSED: "
const chatCompressionToolPlaceholder = "[elided tool output]"

type chatCompressOverflowError struct {
	tokens, limit int
}

func (e *chatCompressOverflowError) Error() string {
	return fmt.Sprintf("compressed request (%d tokens) exceeds context size (%d)", e.tokens, e.limit)
}

func isChatCompressOverflow(err error) bool {
	var target *chatCompressOverflowError
	return errors.As(err, &target)
}

type chatSummarizer func(ctx context.Context, model string, head []api.Message, maxTokens int) (string, int, error)

func resolveChatCompression(req *api.ChatRequest) api.ChatCompressionConfig {
	var cfg api.ChatCompressionConfig
	if req != nil && req.Compression != nil {
		cfg = *req.Compression
	}
	if cfg.Enabled != nil && !*cfg.Enabled {
		return cfg
	}
	if cfg.CompressorModel == "" {
		cfg.CompressorModel = envconfig.ChatCompressor()
	}
	if strings.TrimSpace(cfg.Mode) == "" {
		cfg.Mode = envconfig.ChatCompressionMode()
	}
	msgs := []api.Message(nil)
	if req != nil {
		msgs = req.Messages
	}
	return finalizeChatCompression(cfg, msgs)
}

// finalizeChatCompression picks mode and enablement so operators do not need
// ZEROLLAMA_CHAT_COMPRESSION_MODE. Agent threads (tool/think) get in-place
// placeholder elide with no extra model; summary stays opt-in (env or enabled).
func finalizeChatCompression(cfg api.ChatCompressionConfig, msgs []api.Message) api.ChatCompressionConfig {
	if cfg.Enabled != nil && !*cfg.Enabled {
		return cfg
	}
	agent := chatThreadHasElidableContext(msgs)
	if strings.TrimSpace(cfg.Mode) == "" {
		if agent {
			cfg.Mode = "placeholder"
		} else if chatCompressionEnabled(cfg) || envconfig.ChatCompression() {
			cfg.Mode = "summary"
		}
	}
	if cfg.Enabled != nil {
		return cfg
	}
	if chatCompressionMode(cfg) == "placeholder" && agent {
		on := true
		cfg.Enabled = &on
		return cfg
	}
	if envconfig.ChatCompression() {
		on := true
		cfg.Enabled = &on
	}
	return cfg
}

func chatThreadHasElidableContext(msgs []api.Message) bool {
	for _, m := range msgs {
		if strings.EqualFold(m.Role, "tool") || len(m.ToolCalls) > 0 || strings.TrimSpace(m.Thinking) != "" {
			return true
		}
	}
	return false
}

func numCtxFromChatOptions(opts map[string]any) int {
	if opts != nil {
		switch v := opts["num_ctx"].(type) {
		case float64:
			if int(v) > 0 {
				return int(v)
			}
		case int:
			if v > 0 {
				return v
			}
		case json.Number:
			n, err := v.Int64()
			if err == nil && n > 0 {
				return int(n)
			}
		}
	}
	n := int(envconfig.ContextLength())
	if n <= 0 {
		return 8192
	}
	return n
}

func applyChatCompressionForRequest(ctx context.Context, req *api.ChatRequest, msgs []api.Message, numCtx int, model string, origin int, summarize chatSummarizer) ([]api.Message, *api.ChatCompressionMeta, error) {
	if req == nil {
		return msgs, nil, nil
	}
	if origin < 0 || origin > len(msgs) {
		origin = 0
	}
	policy := finalizeChatCompression(resolveChatCompression(req), msgs)
	if !chatCompressionEnabled(policy) {
		return msgs, nil, nil
	}
	if numCtx <= 0 {
		numCtx = numCtxFromChatOptions(req.Options)
	}
	reqN := len(msgs) - origin
	if reqN < 0 {
		reqN = len(msgs)
		origin = 0
	}
	key := stickyElideMapKey(model, modality.ExtractPromptCacheKey(req.Options))
	clientElide := policy.ElideFrom != nil
	if clientElide {
		v := *policy.ElideFrom + origin
		policy.ElideFrom = &v
	}
	if chatOptionsCacheReset(req.Options) {
		forgetStickyElide(key)
	} else if !clientElide {
		if from, ok := lookupStickyElide(key, reqN); ok {
			v := from + origin
			policy.ElideFrom = &v
		}
	}
	out, meta, err := compressChatMessages(ctx, policy, numCtx, model, msgs, summarize)
	if err == nil && meta != nil {
		if meta.ElideFrom >= origin {
			meta.ElideFrom -= origin
		}
		rememberStickyElide(key, meta.ElideFrom, reqN)
	}
	return out, meta, err
}

func chatCompressionEnabled(cfg api.ChatCompressionConfig) bool {
	return cfg.Enabled != nil && *cfg.Enabled
}

func estimateMessageTokens(m api.Message) int {
	n := len(m.Content) + len(m.Thinking) + len(m.ToolName) + len(m.ToolCallID)
	for _, tc := range m.ToolCalls {
		n += len(tc.Function.Name)
		if b, err := json.Marshal(tc.Function.Arguments); err == nil {
			n += len(b)
		}
	}
	n += 256 * (len(m.Images) + len(m.Videos) + len(m.AudioClips))
	if n <= 0 {
		return 1
	}
	return (n + 3) / 4
}

func estimateMessagesTokens(msgs []api.Message) int {
	total := 0
	for _, m := range msgs {
		total += estimateMessageTokens(m)
	}
	return total
}

func isPreservedChatPrefix(m api.Message) bool {
	role := strings.ToLower(strings.TrimSpace(m.Role))
	return role == "system" || role == "developer"
}

// keepTailCompressReplay matches x/freetokenlab.KeepTailCompressReplay: radix
// can reuse the exact system/developer prefix only; summary+tail must re-prefill.
func keepTailCompressReplay(prefixTokens, summaryTokens, tailTokens int) (reuse, recompute int) {
	if prefixTokens < 0 {
		prefixTokens = 0
	}
	if summaryTokens < 0 {
		summaryTokens = 0
	}
	if tailTokens < 0 {
		tailTokens = 0
	}
	return prefixTokens, summaryTokens + tailTokens
}

func partitionChatTail(messages []api.Message, keepTokens int) (head, tail []api.Message) {
	if keepTokens <= 0 {
		keepTokens = 2048
	}
	cut := len(messages)
	for cut > 1 {
		start := cut - 1
		if strings.EqualFold(messages[start].Role, "tool") {
			for start > 0 && strings.EqualFold(messages[start-1].Role, "tool") {
				start--
			}
			if start > 0 && len(messages[start-1].ToolCalls) > 0 {
				start--
			}
		}
		candidate := messages[start:]
		tokens := estimateMessagesTokens(candidate)
		if cut != len(messages) && tokens > keepTokens {
			break
		}
		cut = start
	}
	return messages[:cut], messages[cut:]
}

func oldestCompressionSummary(messages []api.Message) int {
	for i, m := range messages {
		if strings.EqualFold(m.Role, "system") && strings.HasPrefix(m.Content, chatCompressionSummaryPrefix) {
			return i
		}
	}
	return -1
}

func chatCompressionMode(policy api.ChatCompressionConfig) string {
	switch strings.ToLower(strings.TrimSpace(policy.Mode)) {
	case "placeholder":
		return "placeholder"
	default:
		return "summary"
	}
}

func messagesTokenEqual(a, b api.Message) bool {
	if !strings.EqualFold(a.Role, b.Role) || a.Content != b.Content || a.Thinking != b.Thinking {
		return false
	}
	if a.ToolName != b.ToolName || a.ToolCallID != b.ToolCallID {
		return false
	}
	if len(a.ToolCalls) != len(b.ToolCalls) || len(a.Images) != len(b.Images) {
		return false
	}
	return true
}

func longestExactPrefixTokens(before, after []api.Message) int {
	n := len(before)
	if len(after) < n {
		n = len(after)
	}
	tok := 0
	for i := 0; i < n; i++ {
		if !messagesTokenEqual(before[i], after[i]) {
			break
		}
		tok += estimateMessageTokens(before[i])
	}
	return tok
}

func cloneMessages(in []api.Message) []api.Message {
	out := make([]api.Message, len(in))
	copy(out, in)
	return out
}

func elideMessage(m *api.Message) bool {
	changed := false
	if strings.EqualFold(m.Role, "tool") && m.Content != chatCompressionToolPlaceholder {
		if estimateMessageTokens(*m) > estimateMessageTokens(api.Message{Role: "tool", Content: chatCompressionToolPlaceholder}) {
			m.Content = chatCompressionToolPlaceholder
			changed = true
		}
	}
	if m.Thinking != "" {
		m.Thinking = ""
		changed = true
	}
	return changed
}

func compressChatMessages(ctx context.Context, policy api.ChatCompressionConfig, contextSize int, primaryModel string, messages []api.Message, summarize chatSummarizer) ([]api.Message, *api.ChatCompressionMeta, error) {
	if !chatCompressionEnabled(policy) || len(messages) == 0 {
		return messages, nil, nil
	}
	original := estimateMessagesTokens(messages)
	if contextSize <= 0 {
		return nil, nil, &chatCompressOverflowError{tokens: original, limit: contextSize}
	}
	ratio := policy.TriggerAtRatio
	if ratio <= 0 {
		ratio = 0.75
	}
	if original < int(float64(contextSize)*ratio) {
		stickyPlaceholder := policy.ElideFrom != nil && chatCompressionMode(policy) == "placeholder"
		if !stickyPlaceholder {
			return messages, nil, nil
		}
	}
	if len(messages) < 2 {
		if original <= contextSize {
			return messages, nil, nil
		}
		return nil, nil, &chatCompressOverflowError{tokens: original, limit: contextSize}
	}

	prefixEnd := 0
	for prefixEnd < len(messages) && isPreservedChatPrefix(messages[prefixEnd]) {
		prefixEnd++
	}
	prefix := messages[:prefixEnd]
	if chatCompressionMode(policy) == "placeholder" {
		rest := messages[prefixEnd:]
		var head, tail []api.Message
		if len(rest) > 1 {
			head, tail = rest[:len(rest)-1], rest[len(rest)-1:]
		} else {
			head, tail = nil, rest
		}
		return compressChatPlaceholder(contextSize, messages, prefix, head, tail, original, policy.ElideFrom)
	}
	head, tail := partitionChatTail(messages[prefixEnd:], policy.KeepTailTokens)
	if len(head) == 0 {
		if original <= contextSize {
			return messages, nil, nil
		}
		return nil, nil, &chatCompressOverflowError{tokens: original, limit: contextSize}
	}
	compressor := strings.TrimSpace(policy.CompressorModel)
	if compressor == "" {
		compressor = primaryModel
	}
	maxSummary := policy.MaxSummaryTokens
	if maxSummary <= 0 {
		maxSummary = 512
	}
	if summarize == nil {
		return nil, nil, fmt.Errorf("compress chat history: summarizer unavailable")
	}
	summary, summaryTokens, err := summarize(ctx, compressor, head, maxSummary)
	if err != nil {
		return nil, nil, fmt.Errorf("compress chat history: %w", err)
	}
	content := chatCompressionSummaryPrefix + strings.TrimSpace(summary) + "]"
	result := append([]api.Message(nil), prefix...)
	result = append(result, api.Message{Role: "system", Content: content})
	result = append(result, tail...)
	compressed := estimateMessagesTokens(result)
	summaryTok := estimateMessageTokens(api.Message{Role: "system", Content: content})
	reuse, recompute := keepTailCompressReplay(estimateMessagesTokens(prefix), summaryTok, estimateMessagesTokens(tail))
	meta := &api.ChatCompressionMeta{
		OriginalTokens:    original,
		CompressedTokens:  compressed,
		DroppedTurns:      len(head),
		Compressor:        compressor,
		SummaryTokens:     summaryTokens,
		PrefixReuseTokens: reuse,
		RecomputeTokens:   recompute,
		Mode:              "summary",
	}
	if compressed <= contextSize {
		return result, meta, nil
	}
	if strings.ToLower(strings.TrimSpace(policy.OnPostCompressionOverflow)) != "drop_oldest_summary" {
		return nil, nil, &chatCompressOverflowError{tokens: compressed, limit: contextSize}
	}
	for recoveries := 0; recoveries < 2; recoveries++ {
		idx := oldestCompressionSummary(result)
		if idx < 0 {
			break
		}
		result = append(result[:idx], result[idx+1:]...)
		meta.OverflowRecoveries++
		compressed = estimateMessagesTokens(result)
		meta.CompressedTokens = compressed
		if compressed <= contextSize {
			meta.PrefixReuseTokens = 0
			meta.RecomputeTokens = compressed
			return result, meta, nil
		}
	}
	return nil, nil, &chatCompressOverflowError{tokens: compressed, limit: contextSize}
}

func compressChatPlaceholder(contextSize int, originalMsgs, prefix, head, tail []api.Message, original int, stickyFrom *int) ([]api.Message, *api.ChatCompressionMeta, error) {
	elided := cloneMessages(head)
	mutated := 0
	prefixEnd := len(prefix)
	minElide := -1
	noteElide := func(origIdx int) {
		if minElide < 0 || origIdx < minElide {
			minElide = origIdx
		}
	}
	assemble := func() []api.Message {
		out := append([]api.Message(nil), prefix...)
		out = append(out, elided...)
		return append(out, tail...)
	}
	result := assemble()
	if stickyFrom != nil {
		for i := range elided {
			if prefixEnd+i >= *stickyFrom && elideMessage(&elided[i]) {
				mutated++
				noteElide(prefixEnd + i)
			}
		}
		result = assemble()
	}
	// Elide newest head messages first so the start of the thread stays an
	// exact prefix (FreeToken KV reuse). Peel oldest only if still over.
	for i := len(elided) - 1; i >= 0 && estimateMessagesTokens(result) > contextSize; i-- {
		if elideMessage(&elided[i]) {
			mutated++
			noteElide(prefixEnd + i)
			result = assemble()
		}
	}
	dropped := mutated
	frontDropped := 0
	for estimateMessagesTokens(result) > contextSize && len(elided) > 0 {
		noteElide(prefixEnd + frontDropped)
		elided = elided[1:]
		frontDropped++
		dropped++
		result = assemble()
	}
	compressed := estimateMessagesTokens(result)
	reuse := longestExactPrefixTokens(originalMsgs, result)
	recompute := compressed - reuse
	if recompute < 0 {
		recompute = 0
	}
	meta := &api.ChatCompressionMeta{
		OriginalTokens:    original,
		CompressedTokens:  compressed,
		DroppedTurns:      dropped,
		Compressor:        "placeholder",
		PrefixReuseTokens: reuse,
		RecomputeTokens:   recompute,
		Mode:              "placeholder",
	}
	if minElide >= 0 {
		meta.ElideFrom = minElide
	} else if stickyFrom != nil && *stickyFrom >= 0 {
		meta.ElideFrom = *stickyFrom
	}
	if dropped == 0 {
		if original <= contextSize {
			return originalMsgs, nil, nil
		}
		return nil, nil, &chatCompressOverflowError{tokens: original, limit: contextSize}
	}
	if compressed <= contextSize {
		return result, meta, nil
	}
	if original <= contextSize {
		return originalMsgs, nil, nil
	}
	return nil, nil, &chatCompressOverflowError{tokens: compressed, limit: contextSize}
}
