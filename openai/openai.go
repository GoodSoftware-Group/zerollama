// openai package provides core transformation logic for partial compatibility with the OpenAI REST API
package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

var finishReasonToolCalls = "tool_calls"
var finishReasonLength = "length"

// chatFinishReason maps Ollama done_reason. Length and preempted win over
// tool_calls so a truncated tool parse is not reported as a finished call
// (mlx-serve toolCallFinishReason).
func chatFinishReason(doneReason string, hasTools bool) *string {
	if doneReason == "" {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(doneReason)) {
	case "length":
		return &finishReasonLength
	case "preempted":
		s := doneReason
		return &s
	}
	if hasTools {
		return &finishReasonToolCalls
	}
	return &doneReason
}

type Error struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   any     `json:"param"`
	Code    *string `json:"code"`
	// MediaSession / MissingLabels support agent re-upload loops for /v1/media.
	MediaSession  string   `json:"media_session,omitempty"`
	MissingLabels []string `json:"missing_labels,omitempty"`
}

type ErrorResponse struct {
	Error Error `json:"error"`
}

// NewMediaMissingError is returned when POST /v1/videos references expired or evicted media labels.
// WHY a dedicated code + missing_labels: agents can re-PUT without scraping English text
// or inventing digests (server hashes on PUT). See docs/media-uploads.md.
func NewMediaMissingError(session string, missing []string) ErrorResponse {
	code := "media_missing"
	msg := "media missing — re-upload PUT /v1/media/{session}/{label}"
	return ErrorResponse{Error: Error{
		Type:          "invalid_request_error",
		Message:       msg,
		Code:          &code,
		MediaSession:  session,
		MissingLabels: missing,
	}}
}

// NewMediaTypeMismatchError is returned when media kinds do not match the video backend.
// WHY: same /v1/media store holds future video clips; Wan keyframes must be images today.
func NewMediaTypeMismatchError(session string, labels []string, message string) ErrorResponse {
	code := "media_type_mismatch"
	if message == "" {
		message = "media type mismatch for video generation"
	}
	return ErrorResponse{Error: Error{
		Type:          "invalid_request_error",
		Message:       message,
		Code:          &code,
		MediaSession:  session,
		MissingLabels: labels,
	}}
}

type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	Reasoning  string     `json:"reasoning,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ChoiceLogprobs struct {
	Content []api.Logprob `json:"content"`
}

type Choice struct {
	Index         int                `json:"index"`
	Message       Message            `json:"message"`
	FinishReason  *string            `json:"finish_reason"`
	FinishDetails *api.FinishDetails `json:"finish_details,omitempty"`
	Logprobs      *ChoiceLogprobs    `json:"logprobs,omitempty"`
}

type ChunkChoice struct {
	Index         int                `json:"index"`
	Delta         Message            `json:"delta"`
	FinishReason  *string            `json:"finish_reason"`
	FinishDetails *api.FinishDetails `json:"finish_details,omitempty"`
	Logprobs      *ChoiceLogprobs    `json:"logprobs,omitempty"`
}

type CompleteChunkChoice struct {
	Text          string              `json:"text"`
	Index         int                 `json:"index"`
	FinishReason  *string             `json:"finish_reason"`
	FinishDetails *api.FinishDetails  `json:"finish_details,omitempty"`
	Logprobs      *CompletionLogprobs `json:"logprobs,omitempty"`
}

// CompletionLogprobs is the OpenAI /v1/completions shape (four parallel arrays).
type CompletionLogprobs struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float64            `json:"token_logprobs"`
	TopLogprobs   []map[string]float64 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

type PromptTokensDetails struct {
	CachedTokens       *int `json:"cached_tokens,omitempty"`
	CreatedCacheTokens *int `json:"created_cache_tokens,omitempty"`
	ImageTokens        *int `json:"image_tokens,omitempty"`
	AudioTokens        *int `json:"audio_tokens,omitempty"`
	VideoTokens        *int `json:"video_tokens,omitempty"`
}

// CachedTokensDetails breaks down prefix-cache hits by tier (SGLang sglext shape).
// Native path maps L3 disk restore → host and federated blob → storage;
// device remains llama-server / slot cache_n (CachedPromptTokens).
type CachedTokensDetails struct {
	Device         int     `json:"device,omitempty"`
	Host           int     `json:"host,omitempty"`
	Storage        *int    `json:"storage,omitempty"`
	StorageBackend *string `json:"storage_backend,omitempty"`
}

// SglExt carries SGLang-compatible response extensions for agent clients.
type SglExt struct {
	CachedTokensDetails *CachedTokensDetails `json:"cached_tokens_details,omitempty"`
}

type Usage struct {
	PromptTokens        int                      `json:"prompt_tokens"`
	CompletionTokens    int                      `json:"completion_tokens"`
	TotalTokens         int                      `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	Compression         *api.ChatCompressionMeta `json:"compression_meta,omitempty"`
}

type ResponseFormat struct {
	Type       string          `json:"type"`
	JsonSchema *JsonSchema     `json:"json_schema,omitempty"`
	Schema     json.RawMessage `json:"schema,omitempty"` // flat json_schema (mlx-serve)
}

type JsonSchema struct {
	Schema json.RawMessage `json:"schema"`
}

type EmbedRequest struct {
	Input          any    `json:"input"`
	Model          string `json:"model"`
	Dimensions     int    `json:"dimensions,omitempty"`
	EncodingFormat string `json:"encoding_format,omitempty"` // "float" or "base64"
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type Reasoning struct {
	Effort string `json:"effort,omitempty"`
}

type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []Message       `json:"messages"`
	Stream              bool            `json:"stream"`
	StreamOptions       *StreamOptions  `json:"stream_options"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Seed                *int            `json:"seed"`
	Stop                any             `json:"stop"`
	Temperature         *float64        `json:"temperature"`
	FrequencyPenalty    *float64        `json:"frequency_penalty"`
	PresencePenalty     *float64        `json:"presence_penalty"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"`
	MinP                *float64        `json:"min_p"`
	TypicalP            *float64        `json:"typical_p"`
	RepetitionPenalty   *float64        `json:"repetition_penalty"`
	RepeatPenalty       *float64        `json:"repeat_penalty"`
	ResponseFormat      *ResponseFormat `json:"response_format"`
	Tools               []api.Tool      `json:"tools"`
	// Functions is the pre-tools OpenAI chat array. When Tools is empty it is
	// mapped onto Tools (trap 77 used to allowlist-and-drop it).
	Functions []api.ToolFunction `json:"functions,omitempty"`
	// ToolChoice gates tools for this turn ("none" | "auto" | "required" | object).
	// "none" omits tools (trap 78). A named function object keeps that tool only.
	ToolChoice any `json:"tool_choice,omitempty"`
	// FunctionCall is the pre-tools OpenAI gate ("none"|"auto"|{name}). Mapped
	// onto ToolChoice when ToolChoice is unset.
	FunctionCall    any        `json:"function_call,omitempty"`
	Reasoning       *Reasoning `json:"reasoning,omitempty"`
	ReasoningEffort *string    `json:"reasoning_effort,omitempty"`
	Logprobs        *bool      `json:"logprobs"`
	TopLogprobs     int        `json:"top_logprobs"`
	DebugRenderOnly bool       `json:"_debug_render_only"`
	// PromptCacheKey pins L3 prefix cache + session video expansion (same semantics as /api/chat
	// options.prompt_cache_key). Why on OpenAI surface: repeat video_url agent loops need per-thread
	// ffmpeg cache without forcing clients to use the native /api/chat JSON shape.
	PromptCacheKey *string `json:"prompt_cache_key,omitempty"`
	// SessionID is SGLang's first-class session identity (#29436). When prompt_cache_key is
	// unset, it aliases into options.prompt_cache_key so OpenAI/SGLang clients share L3 +
	// session video/ViT caches without a second field name.
	SessionID *string `json:"session_id,omitempty"`
	// EnablePrefixMMCache mirrors SGLang server flag. WHY top-level field: OpenAI clients
	// send this beside prompt_cache_key without nesting in options. Session ViT overlay
	// still requires prompt_cache_key — flag alone logs a hint on /api/chat (see prefix_mm_cache.go).
	EnablePrefixMMCache *bool `json:"enable_prefix_mm_cache,omitempty"`
	// CacheSalt isolates L3 slot hashing across tenants (vLLM cache_salt analog).
	CacheSalt *string `json:"cache_salt,omitempty"`
	// Options passes Ollama-native request options (eliza.conversationId, num_ctx, …). Merged after
	// standard OpenAI fields so operators can set L3 keys and ctx without a second HTTP API.
	Options map[string]any `json:"options,omitempty"`
	// KeepAlive mirrors /api/chat keep_alive (e.g. "30m") so agent clients pin MLX runners.
	KeepAlive *api.Duration `json:"keep_alive,omitempty"`
	// Timeout mirrors /api/chat timeout (duration string or nanoseconds JSON).
	// WHY: Hermes per-call deadline; folded from extra_body like keep_alive.
	Timeout *api.Duration `json:"timeout,omitempty"`
	// Format is the native Ollama structured-output / GBNF field.
	// WHY on /v1: GBNF is not an OpenAI response_format type — Hermes sends
	// {"type":"gbnf","grammar":"..."} via extra_body.format (M15f).
	Format json.RawMessage `json:"format,omitempty"`
	// Think is the native Ollama think knob (bool or "high"|"medium"|"low").
	// WHY bind on /v1 (not passthrough-only): Hermes and native clients send "think"
	// on chat completions; allowlisting alone silently dropped it (Hermes gap).
	// Precedence in FromChatRequest: Think > reasoning_budget_tokens > reasoning_* > enable_thinking.
	Think *api.ThinkValue `json:"think,omitempty"`
	// EnableThinking is a common harness alias (vLLM/SGLang). Mapped to Think in FromChatRequest.
	// Prefer think / reasoning_effort on this stack; accepted so thinking-off arms are not silent no-ops.
	EnableThinking *bool `json:"enable_thinking,omitempty"`
	// ReasoningBudgetTokens is mlx-serve's think opt-in. 0 off, >0 on. Outranks
	// reasoning_effort / enable_thinking. Not a generation token cap.
	ReasoningBudgetTokens *int `json:"reasoning_budget_tokens,omitempty"`
	// ChatTemplateKwargs carries template knobs (enable_thinking, reasoning_effort).
	// Unknown nested keys are rejected (minefield traps 07 + 77).
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
	// Compression is LA22 opt-in history compression (also extra_body.compression).
	Compression *api.ChatCompressionConfig `json:"compression,omitempty"`
	// ContinueFinalMessage extends a trailing assistant message (mlx-serve).
	ContinueFinalMessage bool  `json:"continue_final_message,omitempty"`
	EnablePLD            *bool `json:"enable_pld,omitempty"`
	EnableMTP            *bool `json:"enable_mtp,omitempty"`
	EnableDrafter        *bool `json:"enable_drafter,omitempty"`
	// ParallelToolCalls false keeps the first tool call only (mlx-serve).
	// Nil / true leave parser output unchanged.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
	// Store true would persist the completion. We have no response store — 400.
	Store *bool `json:"store,omitempty"`
	// N is OpenAI's sample count. mlx-serve 400s n>1; we do too (no n-best).
	N *int `json:"n,omitempty"`
	// ServiceTier is OpenAI's capacity tier. auto/default/omit only; flex/scale/priority 400.
	ServiceTier string `json:"service_tier,omitempty"`
	// LogitBias adds a constant to listed token ids before sampling (OpenAI).
	LogitBias map[string]float64 `json:"logit_bias,omitempty"`
}

type ChatCompletion struct {
	Id                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint"`
	Choices           []Choice       `json:"choices"`
	Usage             Usage          `json:"usage,omitempty"`
	Sglext            *SglExt        `json:"sglext,omitempty"`
	DebugInfo         *api.DebugInfo `json:"_debug_info,omitempty"`
}

type ChatCompletionChunk struct {
	Id                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	SystemFingerprint string        `json:"system_fingerprint"`
	Choices           []ChunkChoice `json:"choices"`
	Usage             *Usage        `json:"usage,omitempty"`
	Sglext            *SglExt       `json:"sglext,omitempty"`
}

// TODO (https://github.com/ollama/ollama/issues/5259): support []string, []int and [][]int
type CompletionRequest struct {
	Model             string             `json:"model"`
	Prompt            string             `json:"prompt"`
	FrequencyPenalty  *float32           `json:"frequency_penalty"`
	MaxTokens         *int               `json:"max_tokens"`
	PresencePenalty   *float32           `json:"presence_penalty"`
	Seed              *int               `json:"seed"`
	Stop              any                `json:"stop"`
	Stream            bool               `json:"stream"`
	StreamOptions     *StreamOptions     `json:"stream_options"`
	Temperature       *float32           `json:"temperature"`
	TopP              *float32           `json:"top_p"`
	TopK              *int               `json:"top_k"`
	MinP              *float32           `json:"min_p"`
	TypicalP          *float32           `json:"typical_p"`
	RepetitionPenalty *float32           `json:"repetition_penalty"`
	RepeatPenalty     *float32           `json:"repeat_penalty"`
	LogitBias         map[string]float64 `json:"logit_bias,omitempty"`
	Suffix            string             `json:"suffix"`
	Logprobs          *int               `json:"logprobs"`
	N                 *int               `json:"n,omitempty"`
	BestOf            *int               `json:"best_of,omitempty"`
	ServiceTier       string             `json:"service_tier,omitempty"`
	DebugRenderOnly   bool               `json:"_debug_render_only"`
	EnablePLD         *bool              `json:"enable_pld,omitempty"`
	EnableMTP         *bool              `json:"enable_mtp,omitempty"`
	EnableDrafter     *bool              `json:"enable_drafter,omitempty"`
	Echo              bool               `json:"echo,omitempty"`
}

type Completion struct {
	Id                string                `json:"id"`
	Object            string                `json:"object"`
	Created           int64                 `json:"created"`
	Model             string                `json:"model"`
	SystemFingerprint string                `json:"system_fingerprint"`
	Choices           []CompleteChunkChoice `json:"choices"`
	Usage             Usage                 `json:"usage,omitempty"`
}

type CompletionChunk struct {
	Id                string                `json:"id"`
	Object            string                `json:"object"`
	Created           int64                 `json:"created"`
	Choices           []CompleteChunkChoice `json:"choices"`
	Model             string                `json:"model"`
	SystemFingerprint string                `json:"system_fingerprint"`
	Usage             *Usage                `json:"usage,omitempty"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Index    int    `json:"index"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type Model struct {
	Id              string     `json:"id"`
	Object          string     `json:"object"`
	Created         int64      `json:"created"`
	OwnedBy         string     `json:"owned_by"`
	ContextLength   int        `json:"context_length,omitempty"`
	MaxModelLen     int        `json:"max_model_len,omitempty"`
	ModelMaxTokens  int        `json:"model_max_tokens,omitempty"`
	Capabilities    []string   `json:"capabilities,omitempty"`
	InputModalities []string   `json:"input_modalities,omitempty"`
	SupportsMTP     bool       `json:"supports_mtp,omitempty"`
	Meta            *ModelMeta `json:"meta,omitempty"`
}

// ModelMeta twins top-level context fields for clients that still read meta.*.
type ModelMeta struct {
	ContextLength int    `json:"context_length,omitempty"`
	Architecture  string `json:"architecture,omitempty"`
	SupportsMTP   bool   `json:"supports_mtp,omitempty"`
}

type Embedding struct {
	Object    string `json:"object"`
	Embedding any    `json:"embedding"` // Can be []float32 (float format) or string (base64 format)
	Index     int    `json:"index"`
}

type ListCompletion struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

type EmbeddingList struct {
	Object string         `json:"object"`
	Data   []Embedding    `json:"data"`
	Model  string         `json:"model"`
	Usage  EmbeddingUsage `json:"usage,omitempty"`
}

type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func NewError(code int, message string) ErrorResponse {
	var etype string
	switch code {
	case http.StatusBadRequest:
		etype = "invalid_request_error"
	case http.StatusNotFound:
		etype = "not_found_error"
	default:
		etype = "api_error"
	}

	return ErrorResponse{Error{Type: etype, Message: message}}
}

// ToUsage converts an api.ChatResponse to Usage
func ToUsage(r api.ChatResponse) Usage {
	u := usageFromMetrics(r.Metrics)
	u.Compression = r.Compression
	return u
}

// ToUsageGenerate converts an api.GenerateResponse to Usage
func ToUsageGenerate(r api.GenerateResponse) Usage {
	return usageFromMetrics(r.Metrics)
}

func usageFromMetrics(m api.Metrics) Usage {
	u := Usage{
		PromptTokens:     m.PromptEvalCount,
		CompletionTokens: m.EvalCount,
		TotalTokens:      m.PromptEvalCount + m.EvalCount,
	}
	if d := promptTokensDetailsFromMetrics(m); d != nil {
		u.PromptTokensDetails = d
	}
	return u
}

// SglExtFromMetrics builds SGLang-compatible sglext when prefix cache hits are present.
func SglExtFromMetrics(m api.Metrics) *SglExt {
	if d := cachedTokensDetailsFromMetrics(m); d != nil {
		return &SglExt{CachedTokensDetails: d}
	}
	return nil
}

func cachedTokensDetailsFromMetrics(m api.Metrics) *CachedTokensDetails {
	if m.CachedPromptTokens <= 0 && m.CachedTokensHost <= 0 && m.CachedTokensStorage <= 0 {
		return nil
	}
	d := &CachedTokensDetails{
		Device: m.CachedPromptTokens,
		Host:   m.CachedTokensHost,
	}
	if m.CachedTokensStorage > 0 {
		v := m.CachedTokensStorage
		d.Storage = &v
		if m.CachedTokensStorageBackend != "" {
			b := m.CachedTokensStorageBackend
			d.StorageBackend = &b
		}
	}
	return d
}

func promptTokensDetailsFromMetrics(m api.Metrics) *PromptTokensDetails {
	d := &PromptTokensDetails{}
	cached := m.CachedPromptTokens
	d.CachedTokens = &cached
	if m.CacheCreationTokens > 0 {
		v := m.CacheCreationTokens
		d.CreatedCacheTokens = &v
	}
	if m.ImageTokens > 0 {
		v := m.ImageTokens
		d.ImageTokens = &v
	}
	if m.VideoTokens > 0 {
		v := m.VideoTokens
		d.VideoTokens = &v
	}
	if m.AudioTokens > 0 {
		v := m.AudioTokens
		d.AudioTokens = &v
	}
	return d
}

// ToToolCalls converts api.ToolCall to OpenAI ToolCall format
func ToToolCalls(tc []api.ToolCall) []ToolCall {
	toolCalls := make([]ToolCall, len(tc))
	for i, tc := range tc {
		toolCalls[i].ID = api.EnsureToolCallID(tc.ID, i)
		toolCalls[i].Type = "function"
		toolCalls[i].Function.Name = tc.Function.Name
		toolCalls[i].Index = tc.Function.Index

		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil || len(args) == 0 || string(args) == "null" {
			args = []byte("{}")
		}

		toolCalls[i].Function.Arguments = string(args)
	}
	return toolCalls
}

// ToChatCompletion converts an api.ChatResponse to ChatCompletion
func ToChatCompletion(id string, r api.ChatResponse) ChatCompletion {
	toolCalls := ToToolCalls(r.Message.ToolCalls)

	return ChatCompletion{
		Id:                id,
		Object:            "chat.completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []Choice{{
			Index:         0,
			Message:       Message{Role: r.Message.Role, Content: r.Message.Content, ToolCalls: toolCalls, Reasoning: r.Message.Thinking},
			FinishReason:  chatFinishReason(r.DoneReason, len(toolCalls) > 0),
			Logprobs:      contentLogprobs(r.Message.Content, r.Logprobs),
			FinishDetails: r.FinishDetails,
		}},
		Usage:     ToUsage(r),
		Sglext:    SglExtFromMetrics(r.Metrics),
		DebugInfo: r.DebugInfo,
	}
}

func toChunk(id string, r api.ChatResponse, toolCallSent bool) ChatCompletionChunk {
	toolCalls := ToToolCalls(r.Message.ToolCalls)

	return ChatCompletionChunk{
		Id:                id,
		Object:            "chat.completion.chunk",
		Created:           time.Now().Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []ChunkChoice{{
			Index:         0,
			Delta:         Message{Role: "assistant", Content: r.Message.Content, ToolCalls: toolCalls, Reasoning: r.Message.Thinking},
			FinishReason:  chatFinishReason(r.DoneReason, toolCallSent || len(toolCalls) > 0),
			Logprobs:      contentLogprobs(r.Message.Content, r.Logprobs),
			FinishDetails: r.FinishDetails,
		}},
	}
}

func choiceLogprobs(lps []api.Logprob) *ChoiceLogprobs {
	if len(lps) == 0 {
		return nil
	}
	return &ChoiceLogprobs{Content: lps}
}

func contentLogprobs(content string, lps []api.Logprob) *ChoiceLogprobs {
	if content == "" {
		return nil
	}
	return choiceLogprobs(lps)
}

func completionLogprobs(lps []api.Logprob) *CompletionLogprobs {
	if len(lps) == 0 {
		return nil
	}
	out := &CompletionLogprobs{
		Tokens:        make([]string, 0, len(lps)),
		TokenLogprobs: make([]float64, 0, len(lps)),
		TopLogprobs:   make([]map[string]float64, 0, len(lps)),
		TextOffset:    make([]int, 0, len(lps)),
	}
	off := 0
	for _, lp := range lps {
		tok := lp.Token
		out.Tokens = append(out.Tokens, tok)
		out.TokenLogprobs = append(out.TokenLogprobs, lp.Logprob)
		top := make(map[string]float64, len(lp.TopLogprobs))
		for _, alt := range lp.TopLogprobs {
			top[alt.Token] = alt.Logprob
		}
		out.TopLogprobs = append(out.TopLogprobs, top)
		out.TextOffset = append(out.TextOffset, off)
		off += utf8.RuneCountInString(tok)
	}
	return out
}

// ToChunks converts an api.ChatResponse to one or more ChatCompletionChunk values.
func ToChunks(id string, r api.ChatResponse, toolCallSent bool) []ChatCompletionChunk {
	hasMixedResponse := r.Message.Thinking != "" && (r.Message.Content != "" || len(r.Message.ToolCalls) > 0)
	if !hasMixedResponse {
		return []ChatCompletionChunk{toChunk(id, r, toolCallSent)}
	}

	reasoningChunk := toChunk(id, r, toolCallSent)
	reasoningChunk.Choices[0].Delta.Content = ""
	reasoningChunk.Choices[0].Delta.ToolCalls = nil
	reasoningChunk.Choices[0].FinishReason = nil
	reasoningChunk.Choices[0].FinishDetails = nil
	// logprobs.content describes message.content (mlx-serve). Reasoning never carries them.
	reasoningChunk.Choices[0].Logprobs = nil

	contentOrToolCallsChunk := toChunk(id, r, toolCallSent)
	contentOrToolCallsChunk.Created = reasoningChunk.Created
	contentOrToolCallsChunk.Choices[0].Delta.Reasoning = ""

	return []ChatCompletionChunk{
		reasoningChunk,
		contentOrToolCallsChunk,
	}
}

// Deprecated: use ToChunks for streaming conversion.
func ToChunk(id string, r api.ChatResponse, toolCallSent bool) ChatCompletionChunk {
	return toChunk(id, r, toolCallSent)
}

// KeepaliveChunk is an OpenAI-compatible SSE heartbeat with an empty delta and no
// finish_reason. Strict clients (Mercury, Hermes) ignore SSE comment frames.
func KeepaliveChunk(id, model string) ChatCompletionChunk {
	return ChatCompletionChunk{
		Id:                id,
		Object:            "chat.completion.chunk",
		Created:           time.Now().Unix(),
		Model:             model,
		SystemFingerprint: "fp_ollama",
		Choices: []ChunkChoice{{
			Index: 0,
			Delta: Message{Role: "assistant"},
		}},
	}
}

// CompletionKeepaliveChunk is the text-completion SSE heartbeat analogue of KeepaliveChunk.
func CompletionKeepaliveChunk(id, model string) CompletionChunk {
	return CompletionChunk{
		Id:                id,
		Object:            "text_completion",
		Created:           time.Now().Unix(),
		Model:             model,
		SystemFingerprint: "fp_ollama",
		Choices: []CompleteChunkChoice{{
			Index: 0,
		}},
	}
}

// ToCompletion converts an api.GenerateResponse to Completion
func ToCompletion(id string, r api.GenerateResponse) Completion {
	return Completion{
		Id:                id,
		Object:            "text_completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []CompleteChunkChoice{{
			Text:  r.Response,
			Index: 0,
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
			FinishDetails: r.FinishDetails,
			Logprobs:      completionLogprobs(r.Logprobs),
		}},
		Usage: ToUsageGenerate(r),
	}
}

// ToCompleteChunk converts an api.GenerateResponse to CompletionChunk
func ToCompleteChunk(id string, r api.GenerateResponse) CompletionChunk {
	return CompletionChunk{
		Id:                id,
		Object:            "text_completion",
		Created:           time.Now().Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []CompleteChunkChoice{{
			Text:  r.Response,
			Index: 0,
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
			FinishDetails: r.FinishDetails,
			Logprobs:      completionLogprobs(r.Logprobs),
		}},
	}
}

// ToListCompletion converts an api.ListResponse to ListCompletion
func ToListCompletion(r api.ListResponse) ListCompletion {
	var data []Model
	for _, m := range r.Models {
		id := m.Model
		if id == "" {
			id = m.Name
		}

		ctx := advertisedContextLen(m.Details.ContextLength, m.HostMaxContext, nil)
		row := Model{
			Id:             id,
			Object:         "model",
			Created:        m.ModifiedAt.Unix(),
			OwnedBy:        model.ParseName(id).Namespace,
			ContextLength:  ctx,
			MaxModelLen:    ctx,
			ModelMaxTokens: ctx,
		}
		attachOpenAIModelCaps(&row, m.Capabilities, m.Details)
		attachOpenAISupportsMTP(&row, m.SupportsMTP)
		data = append(data, row)
	}

	return ListCompletion{
		Object: "list",
		Data:   data,
	}
}

// ToEmbeddingList converts an api.EmbedResponse to EmbeddingList
// encodingFormat can be "float", "base64", or empty (defaults to "float")
func ToEmbeddingList(model string, r api.EmbedResponse, encodingFormat string) EmbeddingList {
	if r.Embeddings != nil {
		var data []Embedding
		for i, e := range r.Embeddings {
			var embedding any
			if strings.EqualFold(encodingFormat, "base64") {
				embedding = floatsToBase64(e)
			} else {
				embedding = e
			}

			data = append(data, Embedding{
				Object:    "embedding",
				Embedding: embedding,
				Index:     i,
			})
		}

		return EmbeddingList{
			Object: "list",
			Data:   data,
			Model:  model,
			Usage: EmbeddingUsage{
				PromptTokens: r.PromptEvalCount,
				TotalTokens:  r.PromptEvalCount,
			},
		}
	}

	return EmbeddingList{}
}

// floatsToBase64 encodes a []float32 to a base64 string
func floatsToBase64(floats []float32) string {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, floats)
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// ToModel converts an api.ShowResponse to Model
func ToModel(r api.ShowResponse, m string) Model {
	ctx := advertisedContextLen(r.Details.ContextLength, 0, r.ModelInfo)
	out := Model{
		Id:             m,
		Object:         "model",
		Created:        r.ModifiedAt.Unix(),
		OwnedBy:        model.ParseName(m).Namespace,
		ContextLength:  ctx,
		MaxModelLen:    ctx,
		ModelMaxTokens: ctx,
	}
	attachOpenAIModelCaps(&out, r.Capabilities, r.Details)
	attachOpenAISupportsMTP(&out, r.SupportsMTP)
	return out
}

// attachOpenAIModelCaps maps native tags to mlx-serve /v1/models names
// (chat, tool_use, streaming, vision, reasoning, json_schema, embeddings).
func attachOpenAIModelCaps(row *Model, caps []model.Capability, details api.ModelDetails) {
	has := func(want model.Capability) bool {
		for _, c := range caps {
			if c == want {
				return true
			}
		}
		return false
	}
	chat := has(model.CapabilityCompletion)
	embed := has(model.CapabilityEmbedding)
	var out []string
	if chat {
		out = append(out, "chat")
	}
	if has(model.CapabilityTools) {
		out = append(out, "tool_use")
	}
	if chat {
		out = append(out, "streaming")
	}
	if has(model.CapabilityVision) {
		out = append(out, "vision")
	}
	if has(model.CapabilityThinking) {
		out = append(out, "reasoning")
	}
	if chat {
		out = append(out, "json_schema")
	}
	if embed {
		out = append(out, "embeddings")
	}
	row.Capabilities = out

	var mods []string
	if chat || embed {
		mods = append(mods, "text")
	}
	if has(model.CapabilityVision) || has(model.CapabilityImage) {
		mods = append(mods, "image")
	}
	if has(model.CapabilityVideo) || has(model.CapabilityVideoGen) {
		mods = append(mods, "video")
	}
	if has(model.CapabilityAudio) || has(model.CapabilitySpeech) {
		mods = append(mods, "audio")
	}
	row.InputModalities = mods

	arch := details.ArchitectureType
	if arch == "" {
		arch = details.Family
	}
	if row.ContextLength > 0 || arch != "" {
		if row.Meta == nil {
			row.Meta = &ModelMeta{}
		}
		if row.ContextLength > 0 {
			row.Meta.ContextLength = row.ContextLength
		}
		row.Meta.Architecture = arch
	}
}

func attachOpenAISupportsMTP(row *Model, ok bool) {
	if row == nil || !ok {
		return
	}
	row.SupportsMTP = true
	if row.Meta == nil {
		row.Meta = &ModelMeta{}
	}
	row.Meta.SupportsMTP = true
}

func advertisedContextLen(detailsCtx, hostMax int, modelInfo map[string]any) int {
	if detailsCtx > 0 {
		return detailsCtx
	}
	if v := contextFromModelInfo(modelInfo); v > 0 {
		return v
	}
	if hostMax > 0 {
		return hostMax
	}
	return 0
}

func contextFromModelInfo(info map[string]any) int {
	if info == nil {
		return 0
	}
	if v := intFromAny(info["general.context_length"]); v > 0 {
		return v
	}
	for k, val := range info {
		if strings.HasSuffix(k, ".context_length") {
			if v := intFromAny(val); v > 0 {
				return v
			}
		}
	}
	return 0
}

func intFromAny(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int32:
		return int(n)
	case int64:
		return int(n)
	case uint32:
		return int(n)
	case uint64:
		return int(n)
	case float64:
		if n > 0 {
			return int(n)
		}
	}
	return 0
}

// FromChatRequest converts a ChatCompletionRequest to api.ChatRequest.
// Callers without a request scope use [context.Background] for remote video_url fetches;
// HTTP middleware should use [FromChatRequestWithContext] so clients can cancel downloads.
// joinedTextParts concatenates content-array text parts in order (mlx-serve #195).
// No extra separator — callers that need a newline include it in the part.
func joinedTextParts(parts []string) string {
	return strings.Join(parts, "")
}

func rejectServiceTier(tier string) error {
	switch strings.ToLower(strings.TrimSpace(tier)) {
	case "", "auto", "default":
		return nil
	default:
		return fmt.Errorf("service_tier %q is not supported (omit, auto, or default; flex/scale/priority are 400)", tier)
	}
}

func rejectStore(store *bool) error {
	if store != nil && *store {
		return fmt.Errorf("store:true is not supported (no response store); omit store or set false")
	}
	return nil
}

func toolChoiceRequiresCall(choice any) bool {
	if choice == nil {
		return false
	}
	if s, ok := toolChoiceString(choice); ok {
		return api.ToolChoiceRequiresCall(s)
	}
	name, err := toolChoiceNamedFunction(choice)
	return err == nil && name != ""
}

func rejectNBest(n *int) error {
	if n != nil && *n > 1 {
		return fmt.Errorf("n=%d is not supported (n>1 is 400; use n=1)", *n)
	}
	return nil
}

func rejectBestOf(n *int) error {
	if n != nil && *n > 1 {
		return fmt.Errorf("best_of=%d is not supported (best_of>1 is 400; use best_of=1)", *n)
	}
	return nil
}

func rejectCompletionsLogprobs(n *int) error {
	if n == nil {
		return nil
	}
	if *n < 0 || *n > 5 {
		return fmt.Errorf("logprobs=%d is not supported (use 0–5)", *n)
	}
	return nil
}

// applyLegacyFunctions maps OpenAI functions / function_call onto tools / tool_choice.
// tools and tool_choice win when both shapes are present.
func applyLegacyFunctions(r *ChatCompletionRequest) error {
	if len(r.Tools) == 0 && len(r.Functions) > 0 {
		tools, err := toolsFromFunctions(r.Functions)
		if err != nil {
			return err
		}
		r.Tools = tools
	}
	if r.ToolChoice == nil && r.FunctionCall != nil {
		tc, err := toolChoiceFromFunctionCall(r.FunctionCall)
		if err != nil {
			return err
		}
		r.ToolChoice = tc
	}
	return nil
}

func toolsFromFunctions(fns []api.ToolFunction) ([]api.Tool, error) {
	out := make([]api.Tool, 0, len(fns))
	for i, f := range fns {
		name := strings.TrimSpace(f.Name)
		if name == "" {
			return nil, fmt.Errorf("functions[%d].name is required", i)
		}
		f.Name = name
		out = append(out, api.Tool{Type: "function", Function: f})
	}
	return out, nil
}

func toolChoiceFromFunctionCall(v any) (any, error) {
	if s, ok := toolChoiceString(v); ok {
		switch strings.ToLower(strings.TrimSpace(s)) {
		case "none", "auto":
			return s, nil
		default:
			return nil, fmt.Errorf("unsupported function_call %q", s)
		}
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("invalid function_call: %w", err)
	}
	var obj struct {
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &obj) != nil || strings.TrimSpace(obj.Name) == "" {
		return nil, fmt.Errorf("invalid function_call")
	}
	return map[string]any{
		"type":     "function",
		"function": map[string]any{"name": strings.TrimSpace(obj.Name)},
	}, nil
}

func FromChatRequest(r ChatCompletionRequest) (*api.ChatRequest, error) {
	return FromChatRequestWithContext(context.Background(), r)
}

// FromChatRequestWithContext converts a ChatCompletionRequest to api.ChatRequest.
// Context is threaded to remote video_url GETs so disconnect aborts work; data: URIs ignore it.
func FromChatRequestWithContext(ctx context.Context, r ChatCompletionRequest) (*api.ChatRequest, error) {
	if err := applyLegacyFunctions(&r); err != nil {
		return nil, err
	}
	if err := rejectNBest(r.N); err != nil {
		return nil, err
	}
	if err := rejectServiceTier(r.ServiceTier); err != nil {
		return nil, err
	}
	if err := rejectStore(r.Store); err != nil {
		return nil, err
	}
	var messages []api.Message
	for _, msg := range r.Messages {
		toolName := ""
		if strings.ToLower(msg.Role) == "tool" {
			toolName = msg.Name
			if toolName == "" && msg.ToolCallID != "" {
				toolName = nameFromToolCallID(r.Messages, msg.ToolCallID)
			}
		}
		switch content := msg.Content.(type) {
		case string:
			toolCalls, err := FromCompletionToolCall(msg.ToolCalls)
			if err != nil {
				return nil, err
			}
			messages = append(messages, api.Message{Role: msg.Role, Content: content, Thinking: msg.Reasoning, ToolCalls: toolCalls, ToolName: toolName, ToolCallID: msg.ToolCallID})
		case []any:
			var textParts []string
			var images []api.ImageData
			var videos []api.VideoData
			var audioClips []api.AudioData
			for _, c := range content {
				data, ok := c.(map[string]any)
				if !ok {
					return nil, errors.New("invalid message format")
				}
				switch data["type"] {
				case "text":
					text, ok := data["text"].(string)
					if !ok {
						return nil, errors.New("invalid message format")
					}
					textParts = append(textParts, text)
				case "image_url":
					var url string
					if urlMap, ok := data["image_url"].(map[string]any); ok {
						if url, ok = urlMap["url"].(string); !ok {
							return nil, errors.New("invalid message format")
						}
					} else {
						if url, ok = data["image_url"].(string); !ok {
							return nil, errors.New("invalid message format")
						}
					}

					img, err := decodeImageURL(url)
					if err != nil {
						return nil, err
					}

					images = append(images, img)
				case "video_url":
					var videoURL string
					if vmap, ok := data["video_url"].(map[string]any); ok {
						if videoURL, ok = vmap["url"].(string); !ok {
							return nil, errors.New("invalid message format")
						}
					} else {
						if videoURL, ok = data["video_url"].(string); !ok {
							return nil, errors.New("invalid message format")
						}
					}
					vb, err := decodeVideoURL(ctx, videoURL)
					if err != nil {
						return nil, err
					}
					videos = append(videos, vb)
				case "input_audio":
					audioMap, ok := data["input_audio"].(map[string]any)
					if !ok {
						return nil, errors.New("invalid input_audio format")
					}
					b64Data, ok := audioMap["data"].(string)
					if !ok {
						return nil, errors.New("invalid input_audio format: missing data")
					}
					audioBytes, err := base64.StdEncoding.DecodeString(b64Data)
					if err != nil {
						return nil, fmt.Errorf("invalid input_audio base64 data: %w", err)
					}
					audioClips = append(audioClips, audioBytes)
				default:
					return nil, errors.New("invalid message format")
				}
			}
			contentJoined := joinedTextParts(textParts)
			messages = append(messages, api.Message{
				Role:       msg.Role,
				Content:    contentJoined,
				Images:     images,
				AudioClips: audioClips,
				Videos:     videos,
			})
			// SGLang #33898: always keep tool metadata on multipart tool messages
			// (image tool results often have no ToolCalls array).
			if msg.Role == "tool" || len(msg.ToolCalls) > 0 {
				if len(msg.ToolCalls) > 0 {
					toolCalls, err := FromCompletionToolCall(msg.ToolCalls)
					if err != nil {
						return nil, err
					}
					messages[len(messages)-1].ToolCalls = toolCalls
				}
				messages[len(messages)-1].ToolName = toolName
				messages[len(messages)-1].ToolCallID = msg.ToolCallID
				messages[len(messages)-1].Thinking = msg.Reasoning
			}
		default:
			// content is only optional if tool calls are present
			if msg.ToolCalls == nil {
				return nil, fmt.Errorf("invalid message content type: %T", content)
			}

			toolCalls, err := FromCompletionToolCall(msg.ToolCalls)
			if err != nil {
				return nil, err
			}
			messages = append(messages, api.Message{Role: msg.Role, Thinking: msg.Reasoning, ToolCalls: toolCalls, ToolCallID: msg.ToolCallID})
		}
	}

	options := make(map[string]any)

	switch stop := r.Stop.(type) {
	case string:
		options["stop"] = []string{stop}
	case []any:
		var stops []string
		for _, s := range stop {
			if str, ok := s.(string); ok {
				stops = append(stops, str)
			}
		}
		options["stop"] = stops
	}

	samplingOpts{
		Temperature:         r.Temperature,
		TopP:                r.TopP,
		MinP:                r.MinP,
		TypicalP:            r.TypicalP,
		FrequencyPenalty:    r.FrequencyPenalty,
		PresencePenalty:     r.PresencePenalty,
		RepetitionPenalty:   r.RepetitionPenalty,
		RepeatPenalty:       r.RepeatPenalty,
		TopK:                r.TopK,
		Seed:                r.Seed,
		MaxTokens:           r.MaxTokens,
		MaxCompletionTokens: r.MaxCompletionTokens,
	}.apply(options)

	if r.Options != nil {
		for k, v := range r.Options {
			options[k] = v
		}
	}
	// Top-level prompt_cache_key wins over options map — explicit agent thread id for caches.
	// SGLang session_id (#29436) fills the same slot when prompt_cache_key is omitted.
	if r.PromptCacheKey != nil && strings.TrimSpace(*r.PromptCacheKey) != "" {
		options["prompt_cache_key"] = strings.TrimSpace(*r.PromptCacheKey)
	} else if r.SessionID != nil && strings.TrimSpace(*r.SessionID) != "" {
		options["prompt_cache_key"] = strings.TrimSpace(*r.SessionID)
	}
	if r.EnablePrefixMMCache != nil && *r.EnablePrefixMMCache {
		options["enable_prefix_mm_cache"] = true
	}
	if r.CacheSalt != nil && strings.TrimSpace(*r.CacheSalt) != "" {
		options["cache_salt"] = strings.TrimSpace(*r.CacheSalt)
	}
	if r.EnablePLD != nil {
		options["enable_pld"] = *r.EnablePLD
	}
	if r.EnableMTP != nil {
		options["enable_mtp"] = *r.EnableMTP
	}
	if r.EnableDrafter != nil {
		options["enable_drafter"] = *r.EnableDrafter
	}
	if err := putLogitBias(options, r.LogitBias); err != nil {
		return nil, err
	}

	var format json.RawMessage
	if r.ResponseFormat != nil {
		format = formatFromStructuredOutput(r.ResponseFormat.Type, r.ResponseFormat.JsonSchema, r.ResponseFormat.Schema)
	}
	// Native format (incl. GBNF) wins when set — OpenAI response_format has no GBNF type.
	if len(r.Format) > 0 {
		format = r.Format
	}
	if len(r.Tools) > 0 && formatHasGrammar(format) {
		return nil, fmt.Errorf("grammar is not supported together with tools")
	}

	var think *api.ThinkValue
	var effort string
	// OpenAI reasoning_* / enable_thinking / chat_template_kwargs are soft on
	// non-thinking models (ThinkFromAlias). Explicit top-level think is native-strict
	// (same as /api/chat) — not ThinkFromAlias.
	thinkFromAlias := false

	// WHY Think first: once bound on /v1 it is the most explicit native shape;
	// reasoning_budget_tokens then reasoning_* / enable_thinking are aliases.
	if r.Think != nil && r.Think.Value != nil {
		if !r.Think.IsValid() {
			return nil, fmt.Errorf("invalid think value: must be boolean or \"high\", \"medium\", \"low\"")
		}
		think = r.Think
	}

	if think == nil {
		if t, err := thinkFromReasoningBudget(r.ReasoningBudgetTokens); err != nil {
			return nil, err
		} else if t != nil {
			think = t
			thinkFromAlias = true
		}
	}

	if think == nil {
		if r.Reasoning != nil {
			effort = r.Reasoning.Effort
		} else if r.ReasoningEffort != nil {
			effort = *r.ReasoningEffort
		}
		if t, err := thinkFromReasoningEffort(effort); err != nil {
			return nil, err
		} else if t != nil {
			think = t
			thinkFromAlias = true
		}
	}

	// Harness aliases (chat_template_kwargs / enable_thinking) after OpenAI reasoning_*.
	if think == nil {
		if t, err := thinkFromEnableThinkingAliases(r.EnableThinking, r.ChatTemplateKwargs); err != nil {
			return nil, err
		} else if t != nil {
			think = t
			thinkFromAlias = true
		}
	}

	tools, err := applyToolChoice(r.Tools, r.ToolChoice)
	if err != nil {
		return nil, err
	}
	messages = api.AppendRequiredToolCallHint(messages, tools, toolChoiceRequiresCall(r.ToolChoice))
	messages = api.AppendOutputBudgetGuidance(messages, api.NumPredictFromMap(options))

	return &api.ChatRequest{
		Model:                r.Model,
		Messages:             messages,
		Format:               format,
		Options:              options,
		Stream:               &r.Stream,
		Tools:                tools,
		Think:                think,
		ThinkFromAlias:       thinkFromAlias,
		Logprobs:             r.Logprobs != nil && *r.Logprobs,
		TopLogprobs:          r.TopLogprobs,
		DebugRenderOnly:      r.DebugRenderOnly,
		KeepAlive:            chatKeepAliveFromRequest(r, options),
		Timeout:              chatTimeoutFromRequest(r, options),
		Compression:          r.Compression,
		ContinueFinalMessage: r.ContinueFinalMessage,
		EnablePLD:            r.EnablePLD,
		EnableMTP:            r.EnableMTP,
		EnableDrafter:        r.EnableDrafter,
		ParallelToolCalls:    r.ParallelToolCalls,
	}, nil
}

// applyToolChoice gates the tool list for this turn.
// none omits tools (trap 78). auto / required / any keep the list.
// A named function object keeps only that tool (400 if unknown).
func applyToolChoice(tools []api.Tool, choice any) ([]api.Tool, error) {
	if choice == nil {
		return tools, nil
	}
	if s, ok := toolChoiceString(choice); ok {
		switch strings.ToLower(strings.TrimSpace(s)) {
		case "", "auto", "required", "any":
			return tools, nil
		case "none":
			return nil, nil
		default:
			return nil, fmt.Errorf("unsupported tool_choice %q", s)
		}
	}
	name, err := toolChoiceNamedFunction(choice)
	if err != nil {
		return nil, err
	}
	if name == "" {
		return tools, nil
	}
	filtered := api.FilterToolsByName(tools, name)
	if len(filtered) == 0 {
		return nil, fmt.Errorf("tool_choice names unknown function %q", name)
	}
	return filtered, nil
}

func toolChoiceString(v any) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case json.RawMessage:
		var s string
		if json.Unmarshal(t, &s) == nil {
			return s, true
		}
	}
	return "", false
}

func toolChoiceNamedFunction(v any) (string, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("invalid tool_choice: %w", err)
	}
	var obj struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &obj) != nil {
		return "", fmt.Errorf("invalid tool_choice")
	}
	typ := strings.ToLower(strings.TrimSpace(obj.Type))
	name := strings.TrimSpace(obj.Function.Name)
	if name == "" {
		name = strings.TrimSpace(obj.Name)
	}
	switch typ {
	case "", "function", "tool":
		if typ != "" && name == "" {
			return "", fmt.Errorf("tool_choice.%s requires a function name", typ)
		}
		return name, nil
	default:
		return "", fmt.Errorf("unsupported tool_choice.type %q", obj.Type)
	}
}

// toolChoiceMeansNone reports whether tool_choice is the string "none"
// (OpenAI Chat Completions / Responses). Object forms are never "none".
func toolChoiceMeansNone(v any) bool {
	s, ok := toolChoiceString(v)
	return ok && strings.EqualFold(strings.TrimSpace(s), "none")
}

func chatKeepAliveFromRequest(r ChatCompletionRequest, options map[string]any) *api.Duration {
	if r.KeepAlive != nil {
		return r.KeepAlive
	}
	if options == nil {
		return nil
	}
	raw, ok := options["keep_alive"]
	if !ok || raw == nil {
		return nil
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return nil
	}
	var d api.Duration
	if err := json.Unmarshal(b, &d); err != nil {
		return nil
	}
	delete(options, "keep_alive")
	return &d
}

func chatTimeoutFromRequest(r ChatCompletionRequest, options map[string]any) *api.Duration {
	if r.Timeout != nil {
		return r.Timeout
	}
	if options == nil {
		return nil
	}
	raw, ok := options["timeout"]
	if !ok || raw == nil {
		return nil
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return nil
	}
	var d api.Duration
	if err := json.Unmarshal(b, &d); err != nil {
		return nil
	}
	delete(options, "timeout")
	return &d
}

func nameFromToolCallID(messages []Message, toolCallID string) string {
	// iterate backwards to be more resilient to duplicate tool call IDs (this
	// follows "last one wins")
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		for _, tc := range msg.ToolCalls {
			if tc.ID == toolCallID {
				return tc.Function.Name
			}
		}
	}
	return ""
}

// decodeImageURL decodes a base64 data URI into raw image bytes.
func decodeImageURL(url string) (api.ImageData, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return nil, errors.New("image URLs are not currently supported, please use base64 encoded data instead")
	}

	types := []string{"jpeg", "jpg", "png", "webp"}

	// Support blank mime type to match /api/chat's behavior of taking just unadorned base64
	if strings.HasPrefix(url, "data:;base64,") {
		url = strings.TrimPrefix(url, "data:;base64,")
	} else {
		valid := false
		for _, t := range types {
			prefix := "data:image/" + t + ";base64,"
			if strings.HasPrefix(url, prefix) {
				url = strings.TrimPrefix(url, prefix)
				valid = true
				break
			}
		}
		if !valid {
			return nil, errors.New("invalid image input")
		}
	}

	img, err := base64.StdEncoding.DecodeString(url)
	if err != nil {
		return nil, errors.New("invalid image input")
	}
	return img, nil
}

// FromCompletionToolCall converts OpenAI ToolCall format to api.ToolCall
func FromCompletionToolCall(toolCalls []ToolCall) ([]api.ToolCall, error) {
	apiToolCalls := make([]api.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		apiToolCalls[i].ID = tc.ID
		apiToolCalls[i].Function.Name = tc.Function.Name
		s := strings.TrimSpace(tc.Function.Arguments)
		if s == "" || s == "null" {
			s = "{}"
		}
		err := json.Unmarshal([]byte(s), &apiToolCalls[i].Function.Arguments)
		if err != nil {
			return nil, errors.New("invalid tool call arguments")
		}
	}

	return apiToolCalls, nil
}

// FromCompleteRequest converts a CompletionRequest to api.GenerateRequest
func FromCompleteRequest(r CompletionRequest) (api.GenerateRequest, error) {
	if err := rejectNBest(r.N); err != nil {
		return api.GenerateRequest{}, err
	}
	if err := rejectBestOf(r.BestOf); err != nil {
		return api.GenerateRequest{}, err
	}
	if err := rejectCompletionsLogprobs(r.Logprobs); err != nil {
		return api.GenerateRequest{}, err
	}
	if err := rejectServiceTier(r.ServiceTier); err != nil {
		return api.GenerateRequest{}, err
	}
	options := make(map[string]any)

	switch stop := r.Stop.(type) {
	case string:
		options["stop"] = []string{stop}
	case []any:
		var stops []string
		for _, s := range stop {
			if str, ok := s.(string); ok {
				stops = append(stops, str)
			} else {
				return api.GenerateRequest{}, fmt.Errorf("invalid type for 'stop' field: %T", s)
			}
		}
		options["stop"] = stops
	}

	samplingOpts{
		Temperature:       f32as64(r.Temperature),
		TopP:              f32as64(r.TopP),
		MinP:              f32as64(r.MinP),
		TypicalP:          f32as64(r.TypicalP),
		FrequencyPenalty:  f32as64(r.FrequencyPenalty),
		PresencePenalty:   f32as64(r.PresencePenalty),
		RepetitionPenalty: f32as64(r.RepetitionPenalty),
		RepeatPenalty:     f32as64(r.RepeatPenalty),
		TopK:              r.TopK,
		Seed:              r.Seed,
		MaxTokens:         r.MaxTokens,
	}.apply(options)
	if r.EnablePLD != nil {
		options["enable_pld"] = *r.EnablePLD
	}
	if r.EnableMTP != nil {
		options["enable_mtp"] = *r.EnableMTP
	}
	if r.EnableDrafter != nil {
		options["enable_drafter"] = *r.EnableDrafter
	}
	if err := putLogitBias(options, r.LogitBias); err != nil {
		return api.GenerateRequest{}, err
	}

	var logprobs bool
	var topLogprobs int
	if r.Logprobs != nil {
		logprobs = true
		topLogprobs = *r.Logprobs
	}

	return api.GenerateRequest{
		Model:           r.Model,
		Prompt:          r.Prompt,
		Options:         options,
		Stream:          &r.Stream,
		Suffix:          r.Suffix,
		Logprobs:        logprobs,
		TopLogprobs:     topLogprobs,
		DebugRenderOnly: r.DebugRenderOnly,
		EnablePLD:       r.EnablePLD,
		EnableMTP:       r.EnableMTP,
		EnableDrafter:   r.EnableDrafter,
	}, nil
}

// ImageGenerationRequest is an OpenAI-compatible image generation request.
type ImageGenerationRequest struct {
	Model          string         `json:"model"`
	Prompt         string         `json:"prompt"`
	N              int            `json:"n,omitempty"`
	Size           string         `json:"size,omitempty"`
	ResponseFormat string         `json:"response_format,omitempty"`
	Seed           *int64         `json:"seed,omitempty"`
	Options        map[string]any `json:"options,omitempty"`
}

// ImageGenerationResponse is an OpenAI-compatible image generation response.
type ImageGenerationResponse struct {
	Created int64            `json:"created"`
	Data    []ImageURLOrData `json:"data"`
}

// ImageURLOrData contains either a URL or base64-encoded image data.
type ImageURLOrData struct {
	URL     string `json:"url,omitempty"`
	B64JSON string `json:"b64_json,omitempty"`
}

// FromImageGenerationRequest converts an OpenAI image generation request to an Ollama GenerateRequest.
func FromImageGenerationRequest(r ImageGenerationRequest) (api.GenerateRequest, error) {
	if err := rejectUnsupportedImageOpenAI(r.N, "", r.ResponseFormat, nil); err != nil {
		return api.GenerateRequest{}, err
	}
	req := api.GenerateRequest{
		Model:   r.Model,
		Prompt:  r.Prompt,
		Options: r.Options,
	}
	// Parse size if provided (e.g., "1024x768")
	if r.Size != "" {
		var w, h int32
		if _, err := fmt.Sscanf(r.Size, "%dx%d", &w, &h); err == nil {
			req.Width = w
			req.Height = h
		}
	}
	if r.Seed != nil {
		if req.Options == nil {
			req.Options = map[string]any{}
		}
		req.Options["seed"] = *r.Seed
	}
	return req, nil
}

// ToImageGenerationResponse converts an Ollama GenerateResponse to an OpenAI ImageGenerationResponse.
func ToImageGenerationResponse(resp api.GenerateResponse) ImageGenerationResponse {
	var data []ImageURLOrData
	if resp.Image != "" {
		data = []ImageURLOrData{{B64JSON: resp.Image}}
	}
	return ImageGenerationResponse{
		Created: resp.CreatedAt.Unix(),
		Data:    data,
	}
}

// SpeechCreateRequest is an OpenAI-compatible POST /v1/audio/speech body.
type SpeechCreateRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice,omitempty"`
	ResponseFormat string   `json:"response_format,omitempty"` // mp3, opus, aac, flac, wav, pcm
	Speed          *float64 `json:"speed,omitempty"`
	// Emotion is a zerollama extension for expressive engines (Chatterbox / Orpheus).
	// Upstream OpenAI ignores unknown fields; remote-tts forwards it when set.
	Emotion string `json:"emotion,omitempty"`
	// Instructions is the Music 3 caption (SGLang-Omni speech field). Ignored by Piper.
	// Why a TTS field: Omni's /v1/audio/speech uses input=lyrics, instructions=caption;
	// we match that JSON so clients do not learn MiniMax cloud /v1/music_generation.
	Instructions string `json:"instructions,omitempty"`
	// Seed is a Music 3 / Omni extension.
	Seed *int64 `json:"seed,omitempty"`
	// MaxNewTokens caps AR frames (~25 fps). Music jobs may also set duration.
	MaxNewTokens *int `json:"max_new_tokens,omitempty"`
	// Duration seconds for Music 3 (lab). Ignored by Piper.
	Duration *float64 `json:"duration,omitempty"`
	// Steps is DiT Euler steps for Music 3 (default 30).
	Steps *int `json:"steps,omitempty"`
}

// TranscriptionResponse is the response format for /v1/audio/transcriptions.
type TranscriptionResponse struct {
	Text string `json:"text"`
}

// TranscriptionVerboseResponse is a subset of the OpenAI verbose_json transcription shape.
// Segment timestamps are omitted when the backend does not provide them.
type TranscriptionVerboseResponse struct {
	Task     string                 `json:"task"`
	Language string                 `json:"language,omitempty"`
	Duration float64                `json:"duration"`
	Text     string                 `json:"text"`
	Segments []TranscriptionSegment `json:"segments"`
}

// TranscriptionSegment mirrors OpenAI segment fields used by clients.
type TranscriptionSegment struct {
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// TranscriptionResult builds JSON/text/verbose payloads for a transcription.
func TranscriptionResult(text, responseFormat, language string) (contentType string, body []byte, err error) {
	text = strings.TrimSpace(text)
	switch responseFormat {
	case "text":
		return "text/plain", []byte(text), nil
	case "verbose_json":
		verbose := TranscriptionVerboseResponse{
			Task:     "transcribe",
			Language: language,
			Duration: 0,
			Text:     text,
			Segments: []TranscriptionSegment{
				{ID: 0, Start: 0, End: 0, Text: text},
			},
		}
		b, err := json.Marshal(verbose)
		return "application/json", b, err
	default:
		// json, srt, vtt — only json is fully supported; others fall back to simple JSON text field.
		if responseFormat != "" && responseFormat != "json" {
			slog.Debug("transcription response_format not fully implemented; returning json text field", "format", responseFormat)
		}
		b, err := json.Marshal(TranscriptionResponse{Text: text})
		return "application/json", b, err
	}
}

// TranscriptionRequest holds parsed fields from the multipart form.
type TranscriptionRequest struct {
	Model          string
	AudioData      []byte
	ResponseFormat string // "json", "text", "verbose_json"
	Language       string
	Prompt         string
}

// FromTranscriptionRequest converts a transcription request into a ChatRequest
// by wrapping the audio with a system prompt for transcription.
func FromTranscriptionRequest(r TranscriptionRequest) (*api.ChatRequest, error) {
	systemPrompt := "Transcribe the following audio exactly as spoken. Output only the transcription text, nothing else."
	if r.Language != "" {
		systemPrompt += " The audio is in " + r.Language + "."
	}
	if r.Prompt != "" {
		systemPrompt += " Context: " + r.Prompt
	}

	stream := true
	return &api.ChatRequest{
		Model: r.Model,
		Messages: []api.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: "Transcribe this audio.", Images: []api.ImageData{r.AudioData}},
		},
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
		},
	}, nil
}

// ImageEditRequest is an OpenAI-compatible image edit request.
type ImageEditRequest struct {
	Model          string         `json:"model"`
	Prompt         string         `json:"prompt"`
	Image          string         `json:"image"`          // Base64-encoded image data
	Mask           string         `json:"mask,omitempty"` // OpenAI inpaint mask — not supported
	N              *int           `json:"n,omitempty"`
	Size           string         `json:"size,omitempty"` // e.g., "1024x1024"
	ResponseFormat string         `json:"response_format,omitempty"`
	Stream         *bool          `json:"stream,omitempty"`
	Seed           *int64         `json:"seed,omitempty"`
	Options        map[string]any `json:"options,omitempty"`
}

func rejectUnsupportedImageOpenAI(n int, mask, responseFormat string, stream *bool) error {
	if n > 1 {
		return fmt.Errorf("n=%d is not supported (n>1 is 400; use n=1)", n)
	}
	if strings.TrimSpace(mask) != "" {
		return fmt.Errorf("mask is not supported")
	}
	rf := strings.ToLower(strings.TrimSpace(responseFormat))
	if rf != "" && rf != "b64_json" {
		return fmt.Errorf("response_format %q is not supported (omit or b64_json)", responseFormat)
	}
	if stream != nil && *stream {
		return fmt.Errorf("stream:true is not supported on image generation/edits")
	}
	return nil
}

// FromImageEditRequest converts an OpenAI image edit request to an Ollama GenerateRequest.
func FromImageEditRequest(r ImageEditRequest) (api.GenerateRequest, error) {
	n := 0
	if r.N != nil {
		n = *r.N
	}
	if err := rejectUnsupportedImageOpenAI(n, r.Mask, r.ResponseFormat, r.Stream); err != nil {
		return api.GenerateRequest{}, err
	}
	req := api.GenerateRequest{
		Model:   r.Model,
		Prompt:  r.Prompt,
		Options: r.Options,
	}

	// Decode the input image
	if r.Image != "" {
		imgData, err := decodeImageURL(r.Image)
		if err != nil {
			return api.GenerateRequest{}, fmt.Errorf("invalid image: %w", err)
		}
		req.Images = append(req.Images, imgData)
	}

	// Parse size if provided (e.g., "1024x768")
	if r.Size != "" {
		var w, h int32
		if _, err := fmt.Sscanf(r.Size, "%dx%d", &w, &h); err == nil {
			req.Width = w
			req.Height = h
		}
	}

	if r.Seed != nil {
		if req.Options == nil {
			req.Options = map[string]any{}
		}
		req.Options["seed"] = *r.Seed
	}

	return req, nil
}

// VideoCreateRequest is the JSON body for POST /v1/videos (subset of OpenAI Videos API).
// We use options.frames/steps instead of OpenAI "seconds" because Wan is frame-based.
type VideoCreateRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Size    string         `json:"size,omitempty"`
	Seconds string         `json:"seconds,omitempty"` // accepted for compatibility; frames come from options
	Seed    *int64         `json:"seed,omitempty"`
	Options map[string]any `json:"options,omitempty"`
}

// Video is an OpenAI-compatible async video job descriptor.
// IDs may be Python job ids or defer-* placeholders; polling must use the id from POST 202.
type Video struct {
	ID        string      `json:"id"`
	Object    string      `json:"object"`
	CreatedAt int64       `json:"created_at"`
	Status    string      `json:"status"`
	Model     string      `json:"model,omitempty"`
	Progress  float64     `json:"progress,omitempty"`
	Size      string      `json:"size,omitempty"`
	Error     *VideoError `json:"error,omitempty"`
}

// VideoError mirrors OpenAI error objects on failed video jobs.
type VideoError struct {
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
}

// VideoFromSubmit builds the initial 202 response for a newly submitted video job.
// When queued=true the job was deferred (inference was busy); status is "pending"
// to signal the client it is waiting behind inference.  Otherwise status is "queued"
// meaning it entered the Python training job queue and will start when a GPU slot is free.
func VideoFromSubmit(jobID, modelName, size string, queued bool, createdAt int64) Video {
	status := "queued"
	if queued {
		status = "pending"
	}
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}
	return Video{
		ID:        jobID,
		Object:    "video",
		CreatedAt: createdAt,
		Status:    status,
		Model:     modelName,
		Progress:  0,
		Size:      size,
	}
}

// AudioGeneration is an async Music 3 job (POST /v1/audio/generations).
type AudioGeneration struct {
	ID        string      `json:"id"`
	Object    string      `json:"object"`
	CreatedAt int64       `json:"created_at"`
	Status    string      `json:"status"`
	Model     string      `json:"model,omitempty"`
	Progress  float64     `json:"progress,omitempty"`
	Error     *VideoError `json:"error,omitempty"`
}

// AudioGenerationFromSubmit builds the 202 body for a newly queued music job.
func AudioGenerationFromSubmit(jobID, modelName string, queued bool, createdAt int64) AudioGeneration {
	v := VideoFromSubmit(jobID, modelName, "", queued, createdAt)
	return AudioGeneration{
		ID:        v.ID,
		Object:    "audio.generation",
		CreatedAt: v.CreatedAt,
		Status:    v.Status,
		Model:     v.Model,
		Progress:  v.Progress,
	}
}

// AudioGenerationFromTrainingJob maps a training job JSON object to AudioGeneration.
func AudioGenerationFromTrainingJob(jobJSON json.RawMessage) (AudioGeneration, error) {
	v, err := VideoFromTrainingJob(jobJSON)
	if err != nil {
		return AudioGeneration{}, err
	}
	return AudioGeneration{
		ID:        v.ID,
		Object:    "audio.generation",
		CreatedAt: v.CreatedAt,
		Status:    v.Status,
		Model:     v.Model,
		Progress:  v.Progress,
		Error:     v.Error,
	}, nil
}

// VideoFromTrainingJob maps embedded training job JSON to an OpenAI Video.
// The job object uses the bootstrap wire shape: jobId, status, progress,
// progressMessage, resultJson (JSON-encoded string), error, submittedAt,
// videoModel, videoSize.
func VideoFromTrainingJob(jobJSON json.RawMessage) (Video, error) {
	var job struct {
		JobID           string  `json:"jobId"`
		Status          string  `json:"status"`
		Progress        float64 `json:"progress"`
		ProgressMessage string  `json:"progressMessage"`
		Error           string  `json:"error"`
		SubmittedAt     string  `json:"submittedAt"`
		VideoModel      string  `json:"videoModel"`
		VideoSize       string  `json:"videoSize"`
	}
	if err := json.Unmarshal(jobJSON, &job); err != nil {
		return Video{}, err
	}
	v := Video{
		ID:        job.JobID,
		Object:    "video",
		CreatedAt: videoJobCreatedAt(job.SubmittedAt),
		Status:    mapTrainingStatusToVideo(job.Status),
		Progress:  job.Progress,
		Model:     job.VideoModel,
		Size:      job.VideoSize,
	}
	switch job.Status {
	case "failed":
		if job.Error != "" {
			v.Error = &VideoError{Message: job.Error, Code: "video_generation_failed"}
		}
	case "cancelled":
		if job.Error != "" {
			v.Error = &VideoError{Message: job.Error, Code: "video_generation_cancelled"}
		}
	}
	return v, nil
}

func videoJobCreatedAt(submittedAt string) int64 {
	submittedAt = strings.TrimSpace(submittedAt)
	if submittedAt == "" {
		return time.Now().Unix()
	}
	for _, layout := range []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05.999999",
		"2006-01-02T15:04:05",
	} {
		if t, err := time.Parse(layout, submittedAt); err == nil {
			return t.Unix()
		}
	}
	return time.Now().Unix()
}

// mapTrainingStatusToVideo bridges internal training/defer states to OpenAI vocabulary.
// "promoted" → queued: defer job entered Python queue but has not started generate.py yet.
// cancelled is not failed—clients may retry or inspect cancel reason separately.
func mapTrainingStatusToVideo(trainingStatus string) string {
	switch trainingStatus {
	case "pending":
		return "pending"
	case "promoted":
		return "queued"
	case "running":
		return "in_progress"
	case "completed":
		return "completed"
	case "failed":
		return "failed"
	case "cancelled":
		return "cancelled"
	default:
		return "queued"
	}
}
