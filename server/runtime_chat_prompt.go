package server

import (
	"context"
	"errors"
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

// estimatePromptTokens is a conservative heuristic when no runner tokenize is available.
func estimatePromptTokens(prompt string) int {
	if prompt == "" {
		return 0
	}
	n := len(prompt) / 4
	if n < 1 {
		return 1
	}
	return n
}

// chatPromptTokenBudget is the prompt token limit for ggml chatPrompt truncation (Phase 12).
// When num_predict > 0, reserves completion headroom like /internal/render-chat; otherwise uses full num_ctx.
func chatPromptTokenBudget(opts *api.Options) int {
	if opts == nil || opts.NumCtx <= 0 {
		return 0
	}
	if opts.NumPredict > 0 {
		return renderPromptTokenBudget(opts.NumCtx, opts.NumPredict)
	}
	return opts.NumCtx
}

// renderPromptTokenBudget is the prompt token budget for heuristic truncation (Phase 12).
// Reserves headroom for completion (num_predict when set, else 256); not tokenizer-exact.
func renderPromptTokenBudget(numCtx, numPredict int) int {
	if numCtx <= 0 {
		return 0
	}
	reserve := 256
	if numPredict > 0 {
		reserve = numPredict
	}
	maxReserve := numCtx / 2
	if maxReserve < 32 {
		maxReserve = 32
	}
	if reserve > maxReserve {
		reserve = maxReserve
	}
	if numCtx <= reserve+64 {
		reserve = numCtx / 4
		if reserve < 32 {
			reserve = 32
		}
	}
	out := numCtx - reserve
	if out < 1 {
		return 1
	}
	return out
}

func prepareRenderMessages(m *Model, msgs []api.Message) []api.Message {
	out := append([]api.Message(nil), msgs...)
	if len(out) > 0 && out[0].Role != "system" && m.System != "" {
		out = append([]api.Message{{Role: "system", Content: m.System}}, out...)
	}
	return filterThinkTags(out, m)
}

func prepareToolsForRender(
	m *Model,
	tools api.Tools,
	msgs []api.Message,
	think *api.ThinkValue,
) (api.Tools, parsers.Parser, bool) {
	if len(tools) == 0 || m.Config.Parser == "" {
		return tools, nil, false
	}
	p := parsers.ParserForName(m.Config.Parser)
	if p == nil {
		return tools, nil, false
	}
	var last *api.Message
	if len(msgs) > 0 {
		last = &msgs[len(msgs)-1]
	}
	return p.Init(tools, last, think), p, p.HasToolSupport()
}

// tokenizeForLoadedModel returns Tokenize from a loaded ggml runner for m, if any.
// Runtime-only inference (Python llama-server) usually has no loaded runner — callers fall back to heuristic truncation.
func (s *Server) tokenizeForLoadedModel(m *Model) tokenizeFunc {
	if s == nil || s.sched == nil || m == nil {
		return nil
	}
	key := schedulerModelKey(m)
	s.sched.loadedMu.Lock()
	ref, ok := s.sched.loaded[key]
	s.sched.loadedMu.Unlock()
	if !ok || ref == nil || ref.llama == nil || ref.loading {
		return nil
	}
	tok := ref.llama.Tokenize
	return func(ctx context.Context, content string) ([]int, error) {
		return tok(ctx, content)
	}
}

// renderChatPromptTokenized mirrors chatPrompt truncation using a real tokenizer and prompt token budget.
func renderChatPromptTokenized(
	ctx context.Context,
	m *Model,
	msgs []api.Message,
	tools api.Tools,
	think *api.ThinkValue,
	tokenize tokenizeFunc,
	maxPromptTokens int,
) (prompt string, droppedPrefix bool, err error) {
	if maxPromptTokens <= 0 {
		p, err := renderPrompt(m, msgs, tools, think)
		return p, false, err
	}
	lastIdx := len(msgs) - 1
	var system []api.Message
	for i := 0; i <= lastIdx; i++ {
		system = system[:0]
		for j := range i {
			if msgs[j].Role == "system" {
				system = append(system, msgs[j])
			}
		}
		p, err := renderPrompt(m, append(system, msgs[i:]...), tools, think)
		if err != nil {
			return "", false, err
		}
		tokens, err := tokenize(ctx, p)
		if err != nil {
			return "", false, err
		}
		if len(tokens) <= maxPromptTokens {
			return p, i > 0, nil
		}
		if i == lastIdx {
			return p, i > 0, nil
		}
	}
	p, err := renderPrompt(m, msgs, tools, think)
	return p, false, err
}

func renderChatPromptHeuristic(
	m *Model,
	msgs []api.Message,
	tools api.Tools,
	think *api.ThinkValue,
	numCtx int,
	numPredict int,
) (prompt string, droppedPrefix bool, err error) {
	lastIdx := len(msgs) - 1
	var system []api.Message
	budget := renderPromptTokenBudget(numCtx, numPredict)
	for i := 0; i <= lastIdx; i++ {
		system = system[:0]
		for j := range i {
			if msgs[j].Role == "system" {
				system = append(system, msgs[j])
			}
		}
		p, err := renderPrompt(m, append(system, msgs[i:]...), tools, think)
		if err != nil {
			return "", false, err
		}
		if estimatePromptTokens(p) <= budget {
			return p, i > 0, nil
		}
		if i == lastIdx {
			return p, i > 0, nil
		}
	}
	p, err := renderPrompt(m, msgs, tools, think)
	return p, false, err
}

// renderChatPromptPrepared renders/truncates using messages already passed through prepareRenderMessages.
func (s *Server) renderChatPromptPrepared(
	ctx context.Context,
	m *Model,
	msgs []api.Message,
	tools api.Tools,
	think *api.ThinkValue,
	numCtx int,
	numPredict int,
	truncate bool,
) (prompt string, truncateMode string, droppedPrefix bool, hasToolSupport bool, err error) {
	processedTools, _, hasToolSupport := prepareToolsForRender(m, tools, msgs, think)

	if !truncate || numCtx <= 0 {
		p, err := renderPrompt(m, msgs, processedTools, think)
		return p, "none", false, hasToolSupport, err
	}

	budget := renderPromptTokenBudget(numCtx, numPredict)
	// Truncation order (Phase 12 + 14): ggml runner tokenize (legacy path) → Python vocab
	// tokenize (runtime/embed) → len/4 heuristic. Why not heuristic first: tools templates
	// can blow past num_ctx with a small char/4 underestimate and drop the wrong turns.
	if tokenize := s.tokenizeForLoadedModel(m); tokenize != nil {
		p, dropped, err := renderChatPromptTokenized(
			ctx, m, msgs, processedTools, think, memoizeTokenize(tokenize), budget,
		)
		return p, "tokenize", dropped, hasToolSupport, err
	}
	if tokenize := s.tokenizeForRuntimeModel(m); tokenize != nil {
		p, dropped, err := renderChatPromptTokenized(
			ctx, m, msgs, processedTools, think, memoizeTokenize(tokenize), budget,
		)
		if err == nil {
			return p, "tokenize", dropped, hasToolSupport, nil
		}
		// Why only ErrRuntimeTokenizeUnavailable falls back: unreachable runtime is degraded
		// mode; HTTP 4xx/5xx from /internal/tokenize is an operator-visible misconfig.
		if errors.Is(err, ErrRuntimeTokenizeUnavailable) {
			slog.Debug("render-chat: runtime tokenize unavailable, using heuristic", "model", m.Name)
		} else {
			return "", "", false, hasToolSupport, err
		}
	}
	p, dropped, err := renderChatPromptHeuristic(m, msgs, processedTools, think, numCtx, numPredict)
	return p, "heuristic", dropped, hasToolSupport, err
}

// renderChatPromptWithTruncate renders chat like ChatHandler without scheduling a runner.
// Uses the loaded ggml runner tokenizer when available; otherwise len/4 heuristic (Phase 12).
func (s *Server) renderChatPromptWithTruncate(
	ctx context.Context,
	m *Model,
	msgs []api.Message,
	tools api.Tools,
	think *api.ThinkValue,
	numCtx int,
	numPredict int,
	truncate bool,
) (prompt string, truncateMode string, droppedPrefix bool, hasToolSupport bool, err error) {
	msgs = prepareRenderMessages(m, msgs)
	return s.renderChatPromptPrepared(ctx, m, msgs, tools, think, numCtx, numPredict, truncate)
}
