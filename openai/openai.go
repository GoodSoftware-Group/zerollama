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
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

var finishReasonToolCalls = "tool_calls"

type Error struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   any     `json:"param"`
	Code    *string `json:"code"`
}

type ErrorResponse struct {
	Error Error `json:"error"`
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
	Index        int             `json:"index"`
	Message      Message         `json:"message"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs `json:"logprobs,omitempty"`
}

type ChunkChoice struct {
	Index        int             `json:"index"`
	Delta        Message         `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs `json:"logprobs,omitempty"`
}

type CompleteChunkChoice struct {
	Text         string          `json:"text"`
	Index        int             `json:"index"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs `json:"logprobs,omitempty"`
}

type PromptTokensDetails struct {
	CachedTokens        *int `json:"cached_tokens,omitempty"`
	CreatedCacheTokens  *int `json:"created_cache_tokens,omitempty"`
	ImageTokens         *int `json:"image_tokens,omitempty"`
	AudioTokens         *int `json:"audio_tokens,omitempty"`
	VideoTokens         *int `json:"video_tokens,omitempty"`
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
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

type ResponseFormat struct {
	Type       string      `json:"type"`
	JsonSchema *JsonSchema `json:"json_schema,omitempty"`
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
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Stream           bool            `json:"stream"`
	StreamOptions    *StreamOptions  `json:"stream_options"`
	MaxTokens        *int            `json:"max_tokens"`
	Seed             *int            `json:"seed"`
	Stop             any             `json:"stop"`
	Temperature      *float64        `json:"temperature"`
	FrequencyPenalty *float64        `json:"frequency_penalty"`
	PresencePenalty  *float64        `json:"presence_penalty"`
	TopP             *float64        `json:"top_p"`
	ResponseFormat   *ResponseFormat `json:"response_format"`
	Tools            []api.Tool      `json:"tools"`
	// ToolChoice gates tools for this turn ("none" | "auto" | "required" | object).
	// "none" omits tools from the underlying chat request (minefield trap 78).
	ToolChoice      any        `json:"tool_choice,omitempty"`
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
	// EnableThinking is a common harness alias (vLLM/SGLang). Mapped to Think in FromChatRequest.
	// Prefer think / reasoning_effort on this stack; accepted so thinking-off arms are not silent no-ops.
	EnableThinking *bool `json:"enable_thinking,omitempty"`
	// ChatTemplateKwargs carries template knobs (enable_thinking, reasoning_effort).
	// Unknown nested keys are rejected (minefield traps 07 + 77).
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
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
	Model            string         `json:"model"`
	Prompt           string         `json:"prompt"`
	FrequencyPenalty float32        `json:"frequency_penalty"`
	MaxTokens        *int           `json:"max_tokens"`
	PresencePenalty  float32        `json:"presence_penalty"`
	Seed             *int           `json:"seed"`
	Stop             any            `json:"stop"`
	Stream           bool           `json:"stream"`
	StreamOptions    *StreamOptions `json:"stream_options"`
	Temperature      *float32       `json:"temperature"`
	TopP             float32        `json:"top_p"`
	Suffix           string         `json:"suffix"`
	Logprobs         *int           `json:"logprobs"`
	DebugRenderOnly  bool           `json:"_debug_render_only"`
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
	Id      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
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
	return usageFromMetrics(r.Metrics)
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
	// OpenAI-shaped breakdown; zeros omitted so clients see a sparse object only when useful.
	if m.ImageTokens == 0 && m.VideoTokens == 0 && m.AudioTokens == 0 && m.CachedPromptTokens == 0 && m.CacheCreationTokens == 0 {
		return nil
	}
	d := &PromptTokensDetails{}
	if m.CachedPromptTokens > 0 {
		v := m.CachedPromptTokens
		d.CachedTokens = &v
	}
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
		toolCalls[i].ID = tc.ID
		toolCalls[i].Type = "function"
		toolCalls[i].Function.Name = tc.Function.Name
		toolCalls[i].Index = tc.Function.Index

		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			slog.Error("could not marshall function arguments to json", "error", err)
			continue
		}

		toolCalls[i].Function.Arguments = string(args)
	}
	return toolCalls
}

// ToChatCompletion converts an api.ChatResponse to ChatCompletion
func ToChatCompletion(id string, r api.ChatResponse) ChatCompletion {
	toolCalls := ToToolCalls(r.Message.ToolCalls)

	var logprobs *ChoiceLogprobs
	if len(r.Logprobs) > 0 {
		logprobs = &ChoiceLogprobs{Content: r.Logprobs}
	}

	return ChatCompletion{
		Id:                id,
		Object:            "chat.completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []Choice{{
			Index:   0,
			Message: Message{Role: r.Message.Role, Content: r.Message.Content, ToolCalls: toolCalls, Reasoning: r.Message.Thinking},
			FinishReason: func(reason string) *string {
				if len(toolCalls) > 0 {
					reason = "tool_calls"
				}
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
			Logprobs: logprobs,
		}},
		Usage:     ToUsage(r),
		Sglext:    SglExtFromMetrics(r.Metrics),
		DebugInfo: r.DebugInfo,
	}
}

func toChunk(id string, r api.ChatResponse, toolCallSent bool) ChatCompletionChunk {
	toolCalls := ToToolCalls(r.Message.ToolCalls)

	var logprobs *ChoiceLogprobs
	if len(r.Logprobs) > 0 {
		logprobs = &ChoiceLogprobs{Content: r.Logprobs}
	}

	return ChatCompletionChunk{
		Id:                id,
		Object:            "chat.completion.chunk",
		Created:           time.Now().Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []ChunkChoice{{
			Index: 0,
			Delta: Message{Role: "assistant", Content: r.Message.Content, ToolCalls: toolCalls, Reasoning: r.Message.Thinking},
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					if toolCallSent || len(toolCalls) > 0 {
						return &finishReasonToolCalls
					}
					return &reason
				}
				return nil
			}(r.DoneReason),
			Logprobs: logprobs,
		}},
	}
}

// ToChunks converts an api.ChatResponse to one or more ChatCompletionChunk values.
func ToChunks(id string, r api.ChatResponse, toolCallSent bool) []ChatCompletionChunk {
	hasMixedResponse := r.Message.Thinking != "" && (r.Message.Content != "" || len(r.Message.ToolCalls) > 0)
	if !hasMixedResponse {
		return []ChatCompletionChunk{toChunk(id, r, toolCallSent)}
	}

	reasoningChunk := toChunk(id, r, toolCallSent)
	// The logprobs here might include tokens not in this chunk because we now split between thinking and content/tool calls.
	reasoningChunk.Choices[0].Delta.Content = ""
	reasoningChunk.Choices[0].Delta.ToolCalls = nil
	reasoningChunk.Choices[0].FinishReason = nil

	contentOrToolCallsChunk := toChunk(id, r, toolCallSent)
	// Keep both split chunks on the same timestamp since they represent one logical emission.
	contentOrToolCallsChunk.Created = reasoningChunk.Created
	contentOrToolCallsChunk.Choices[0].Delta.Reasoning = ""
	contentOrToolCallsChunk.Choices[0].Logprobs = nil

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

		data = append(data, Model{
			Id:      id,
			Object:  "model",
			Created: m.ModifiedAt.Unix(),
			OwnedBy: model.ParseName(id).Namespace,
		})
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
	return Model{
		Id:      m,
		Object:  "model",
		Created: r.ModifiedAt.Unix(),
		OwnedBy: model.ParseName(m).Namespace,
	}
}

// FromChatRequest converts a ChatCompletionRequest to api.ChatRequest.
// Callers without a request scope use [context.Background] for remote video_url fetches;
// HTTP middleware should use [FromChatRequestWithContext] so clients can cancel downloads.
func FromChatRequest(r ChatCompletionRequest) (*api.ChatRequest, error) {
	return FromChatRequestWithContext(context.Background(), r)
}

// FromChatRequestWithContext converts a ChatCompletionRequest to api.ChatRequest.
// Context is threaded to remote video_url GETs so disconnect aborts work; data: URIs ignore it.
func FromChatRequestWithContext(ctx context.Context, r ChatCompletionRequest) (*api.ChatRequest, error) {
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
			contentJoined := strings.Join(textParts, "\n")
			messages = append(messages, api.Message{
				Role:       msg.Role,
				Content:    contentJoined,
				Images:     images,
				AudioClips: audioClips,
				Videos:     videos,
			})
			if len(msg.ToolCalls) > 0 {
				toolCalls, err := FromCompletionToolCall(msg.ToolCalls)
				if err != nil {
					return nil, err
				}
				messages[len(messages)-1].ToolCalls = toolCalls
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

	if r.MaxTokens != nil {
		options["num_predict"] = *r.MaxTokens
	}

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	} else {
		options["temperature"] = 1.0
	}

	if r.Seed != nil {
		options["seed"] = *r.Seed
	}

	if r.FrequencyPenalty != nil {
		options["frequency_penalty"] = *r.FrequencyPenalty
	}

	if r.PresencePenalty != nil {
		options["presence_penalty"] = *r.PresencePenalty
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	} else {
		options["top_p"] = 1.0
	}

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

	var format json.RawMessage
	if r.ResponseFormat != nil {
		switch strings.ToLower(strings.TrimSpace(r.ResponseFormat.Type)) {
		// Support the old "json_object" type for OpenAI compatibility
		case "json_object":
			format = json.RawMessage(`"json"`)
		case "json_schema":
			if r.ResponseFormat.JsonSchema != nil {
				format = r.ResponseFormat.JsonSchema.Schema
			}
		}
	}

	var think *api.ThinkValue
	var effort string
	// OpenAI reasoning_* / enable_thinking / chat_template_kwargs are soft on
	// non-thinking models (ThinkFromAlias). Native /api/chat think remains strict.
	thinkFromAlias := false

	if r.Reasoning != nil {
		effort = r.Reasoning.Effort
		thinkFromAlias = true
	} else if r.ReasoningEffort != nil {
		effort = *r.ReasoningEffort
		thinkFromAlias = true
	}

	if effort != "" {
		if !slices.Contains([]string{"high", "medium", "low", "none"}, effort) {
			return nil, fmt.Errorf("invalid reasoning value: '%s' (must be \"high\", \"medium\", \"low\", or \"none\")", effort)
		}

		if effort == "none" {
			think = &api.ThinkValue{Value: false}
		} else {
			think = &api.ThinkValue{Value: effort}
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

	tools := r.Tools
	if toolChoiceMeansNone(r.ToolChoice) {
		// Trap 78: tool_choice "none" must actually gate the turn (omit tools).
		tools = nil
	}

	return &api.ChatRequest{
		Model:           r.Model,
		Messages:        messages,
		Format:          format,
		Options:         options,
		Stream:          &r.Stream,
		Tools:           tools,
		Think:           think,
		ThinkFromAlias:  thinkFromAlias,
		Logprobs:        r.Logprobs != nil && *r.Logprobs,
		TopLogprobs:     r.TopLogprobs,
		DebugRenderOnly: r.DebugRenderOnly,
		KeepAlive:       chatKeepAliveFromRequest(r, options),
	}, nil
}

// toolChoiceMeansNone reports whether tool_choice is the string "none"
// (OpenAI Chat Completions / Responses). Object forms are never "none".
func toolChoiceMeansNone(v any) bool {
	switch t := v.(type) {
	case string:
		return strings.EqualFold(strings.TrimSpace(t), "none")
	case json.RawMessage:
		var s string
		if err := json.Unmarshal(t, &s); err == nil {
			return strings.EqualFold(strings.TrimSpace(s), "none")
		}
	}
	return false
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
		err := json.Unmarshal([]byte(tc.Function.Arguments), &apiToolCalls[i].Function.Arguments)
		if err != nil {
			return nil, errors.New("invalid tool call arguments")
		}
	}

	return apiToolCalls, nil
}

// FromCompleteRequest converts a CompletionRequest to api.GenerateRequest
func FromCompleteRequest(r CompletionRequest) (api.GenerateRequest, error) {
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

	if r.MaxTokens != nil {
		options["num_predict"] = *r.MaxTokens
	}

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	} else {
		options["temperature"] = 1.0
	}

	if r.Seed != nil {
		options["seed"] = *r.Seed
	}

	options["frequency_penalty"] = r.FrequencyPenalty

	options["presence_penalty"] = r.PresencePenalty

	if r.TopP != 0.0 {
		options["top_p"] = r.TopP
	} else {
		options["top_p"] = 1.0
	}

	var logprobs bool
	var topLogprobs int
	if r.Logprobs != nil && *r.Logprobs > 0 {
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
func FromImageGenerationRequest(r ImageGenerationRequest) api.GenerateRequest {
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
	return req
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
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Image   string         `json:"image"`          // Base64-encoded image data
	Size    string         `json:"size,omitempty"` // e.g., "1024x1024"
	Seed    *int64         `json:"seed,omitempty"`
	Options map[string]any `json:"options,omitempty"`
}

// FromImageEditRequest converts an OpenAI image edit request to an Ollama GenerateRequest.
func FromImageEditRequest(r ImageEditRequest) (api.GenerateRequest, error) {
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
