package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
)

// PrefillMessages posts the chat to llama-server /v1/prefill.
// The server cuts the last word, renders its own template, and keeps the prefix.
func (s *llamaServerRunner) PrefillMessages(ctx context.Context, messages []api.Message, think *api.ThinkValue, tools api.Tools) (PrefillResult, error) {
	if len(messages) == 0 {
		return PrefillResult{}, errors.New("prefill requires messages")
	}
	converted := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		one, err := llamaServerChatMessage(MessageFromAPI(msg))
		if err != nil {
			return PrefillResult{}, err
		}
		converted = append(converted, one)
	}
	body := map[string]any{"messages": converted}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	if kwargs := llamaServerChatTemplateKwargs(think); kwargs != nil {
		body["chat_template_kwargs"] = kwargs
	}
	return s.prefillPost(ctx, "/v1/prefill", body, "cached_tokens", "prompt_tokens")
}

// PrefillPrompt evaluates an already rendered draft and keeps it.
// prefill_only stops llama-server before it samples. Sampling fields match the
// follow-up chat so a saved logit row still covers that request.
//
// WHY a map, not llamaServerCompletionRequest: NPredict is omitempty, so 0
// would be dropped and the server would sample. A saved top-k covers a later
// k only when the saved k is at least that large, so the sampler fields go too.
func (s *llamaServerRunner) PrefillPrompt(ctx context.Context, prompt string, opts *api.Options) (PrefillResult, error) {
	if prompt == "" {
		return PrefillResult{}, nil
	}
	body := map[string]any{
		"prompt":       prompt,
		"n_predict":    0,
		"stream":       false,
		"cache_prompt": true,
		"prefill_only": true,
	}
	if opts != nil {
		body["temperature"] = opts.Temperature
		body["top_k"] = opts.TopK
		body["top_p"] = opts.TopP
		body["min_p"] = opts.MinP
		body["typical_p"] = opts.TypicalP
		body["repeat_penalty"] = opts.RepeatPenalty
		body["repeat_last_n"] = opts.RepeatLastN
		body["frequency_penalty"] = opts.FrequencyPenalty
		body["presence_penalty"] = opts.PresencePenalty
		body["seed"] = opts.Seed
	}
	return s.prefillPost(ctx, "/completion", body, "tokens_cached", "tokens_evaluated")
}

func (s *llamaServerRunner) prefillPost(ctx context.Context, path string, payload any, cachedKey, promptKey string) (PrefillResult, error) {
	if s == nil || s.port <= 0 {
		return PrefillResult{}, errors.New("llama-server prefill has no port")
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return PrefillResult{}, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return PrefillResult{}, err
	}
	if status != ServerStatusReady {
		return PrefillResult{}, fmt.Errorf("unexpected server status: %s", status)
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return PrefillResult{}, err
	}
	endpoint := fmt.Sprintf("http://127.0.0.1:%d%s", s.port, path)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return PrefillResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return PrefillResult{}, err
		}
		return PrefillResult{}, fmt.Errorf("llama-server prefill: %w", err)
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return PrefillResult{}, err
	}
	if res.StatusCode >= 400 {
		return PrefillResult{}, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(body)}
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		return PrefillResult{}, fmt.Errorf("llama-server prefill: %w", err)
	}
	return PrefillResult{
		CachedTokens: jsonInt(parsed[cachedKey]),
		PromptTokens: jsonInt(parsed[promptKey]),
	}, nil
}

func jsonInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	default:
		return 0
	}
}
