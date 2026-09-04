package server

import (
	"context"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func buildChatCompressionPrompt(head []api.Message) string {
	var b strings.Builder
	b.WriteString("Summarize the following conversation for later context. Keep facts, names, decisions, and open tasks. Be concise.\n\n")
	for _, m := range head {
		role := m.Role
		if role == "" {
			role = "user"
		}
		b.WriteString(role)
		b.WriteString(": ")
		b.WriteString(m.Content)
		if m.Thinking != "" {
			b.WriteString("\n")
			b.WriteString(m.Thinking)
		}
		b.WriteString("\n")
	}
	return b.String()
}

func (s *Server) summarizeChatHead(ctx context.Context, modelName string, head []api.Message, maxTokens int) (string, int, error) {
	r, _, opts, _, releaseQoS, err := s.scheduleRunner(ctx, modelName, []model.Capability{model.CapabilityCompletion}, nil, nil, nil, nil, nil)
	if err != nil {
		return "", 0, err
	}
	defer releaseQoS()
	if opts == nil {
		o := api.DefaultOptions()
		opts = &o
	}
	if maxTokens > 0 {
		opts.NumPredict = maxTokens
	}
	opts.Temperature = 0
	var out strings.Builder
	err = r.Completion(ctx, llm.CompletionRequest{
		Prompt:  buildChatCompressionPrompt(head),
		Options: opts,
	}, func(cr llm.CompletionResponse) {
		out.WriteString(cr.Content)
	})
	if err != nil {
		return "", 0, err
	}
	text := strings.TrimSpace(out.String())
	if text == "" {
		return "", 0, fmt.Errorf("empty compressor output")
	}
	return text, estimateMessageTokens(api.Message{Content: text}), nil
}
