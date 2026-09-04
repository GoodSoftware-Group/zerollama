package api

import (
	"context"
	"encoding/json"
	"errors"
)

// ChatThread is append-only /api/chat history with sticky placeholder
// elide_from. Call NextRequest or Chat after appending the latest user/tool
// message; Observe (or Chat) records Compression from the done response.
type ChatThread struct {
	Client    *Client
	Model     string
	Messages  []Message
	Stream    *bool
	Format    json.RawMessage
	KeepAlive *Duration
	Timeout   *Duration
	Tools     Tools
	Options   map[string]any
	Think     *ThinkValue
	// Compression is optional extra knobs. ElideFrom is filled from the last
	// done meta unless already set.
	Compression *ChatCompressionConfig

	sticky *ChatCompressionMeta
}

// CompressionMeta is the last done ChatCompressionMeta, if any.
func (t *ChatThread) CompressionMeta() *ChatCompressionMeta {
	if t == nil {
		return nil
	}
	return t.sticky
}

// ClearHistory drops messages and the sticky cut (same as CLI /clear).
func (t *ChatThread) ClearHistory() {
	if t == nil {
		return
	}
	t.Messages = nil
	t.sticky = nil
}

// NextRequest builds a ChatRequest with sticky elide_from applied.
func (t *ChatThread) NextRequest() *ChatRequest {
	if t == nil {
		return &ChatRequest{}
	}
	req := &ChatRequest{
		Model:     t.Model,
		Messages:  t.Messages,
		Stream:    t.Stream,
		Format:    t.Format,
		KeepAlive: t.KeepAlive,
		Timeout:   t.Timeout,
		Tools:     t.Tools,
		Options:   t.Options,
		Think:     t.Think,
	}
	if t.Compression != nil {
		cp := *t.Compression
		req.Compression = &cp
	}
	ApplyStickyChatCompression(req, t.sticky)
	return req
}

// Observe records sticky elide_from from a done chat response.
func (t *ChatThread) Observe(resp ChatResponse) {
	if t == nil || resp.Compression == nil || resp.Compression.Mode == "" {
		return
	}
	meta := *resp.Compression
	t.sticky = &meta
}

// Chat POSTs NextRequest and Observes the last streamed chunk.
func (t *ChatThread) Chat(ctx context.Context, fn ChatResponseFunc) error {
	if t == nil || t.Client == nil {
		return errors.New("nil chat thread client")
	}
	var last ChatResponse
	err := t.Client.Chat(ctx, t.NextRequest(), func(resp ChatResponse) error {
		last = resp
		if fn != nil {
			return fn(resp)
		}
		return nil
	})
	if err != nil {
		return err
	}
	t.Observe(last)
	return nil
}
