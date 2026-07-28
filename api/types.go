package api

import (
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/types/model"
)

// StatusError is an error with an HTTP status code and message.
type StatusError struct {
	StatusCode   int
	Status       string
	ErrorMessage string `json:"error"`
}

func (e StatusError) Error() string {
	switch {
	case e.Status != "" && e.ErrorMessage != "":
		return fmt.Sprintf("%s: %s", e.Status, e.ErrorMessage)
	case e.Status != "":
		return e.Status
	case e.ErrorMessage != "":
		return e.ErrorMessage
	default:
		// this should not happen
		return "something went wrong, please see the ollama server logs for details"
	}
}

type AuthorizationError struct {
	StatusCode int
	Status     string
	SigninURL  string `json:"signin_url"`
}

func (e AuthorizationError) Error() string {
	if e.Status != "" {
		return e.Status
	}
	return "something went wrong, please see the ollama server logs for details"
}

// ImageData represents the raw binary data of an image file.
type ImageData []byte

// VideoData is a raw video container (e.g. MP4/WebM). Bytes are staged separately from Images
// because the vision runner consumes raster frames, not containers—sampling runs in the server
// before inference so limits and errors stay centralized (see docs/video-understanding.md).
type VideoData []byte

// AudioData is a raw audio clip (e.g. OpenAI input_audio). Runners still receive these via
// the flat Images list until a dedicated audio projector path exists.
type AudioData []byte

// VideoSpan records how many raster frames came from one original video blob after expansion.
// When non-empty, Images are ordered as: any still images first, then frames for each Videos[] entry in order.
// See docs/video-understanding.md and model/renderers/qwen3vl.go.
type VideoSpan struct {
	FrameCount int `json:"frame_count"`
	// GridTHW is optional SGLang/Qwen-style [T,H,W] patch grid for this clip (preprocessed clients).
	// When set, token estimates use T×H×W / spatial_merge² instead of frame_count × tokens_per_image.
	GridTHW []int `json:"grid_thw,omitempty"`
	// GridTHWExplicit marks client-origin grid (observability). Server ffmpeg estimates
	// also set GridTHW and are forwarded to M-RoPE ViT (GridTHWExplicit=false).
	GridTHWExplicit bool `json:"-"`
}

// PrecomputedEmbedding is SGLang format=precomputed_embedding (ViT output rows, not raw pixels).
// Use with padded_input_ids on the same message; one item per modality list when preprocessed.
type PrecomputedEmbedding struct {
	Format  string      `json:"format,omitempty"`
	Feature [][]float32 `json:"feature,omitempty"`
	// GridTHW is optional [T,H,W] patch grid (required on ollama-engine precomputed ingest).
	GridTHW []int `json:"grid_thw,omitempty"`
}

const PrecomputedEmbeddingFormat = "precomputed_embedding"

// ProcessorOutput is SGLang format=processor_output (HF pixel_values + grid, no raw PNG).
type ProcessorOutput struct {
	Format       string    `json:"format,omitempty"`
	PixelValues  []float32 `json:"pixel_values,omitempty"`
	ImageGridTHW []int     `json:"image_grid_thw,omitempty"`
	// GridTHW alias for clients that send grid_thw instead of image_grid_thw.
	GridTHW []int `json:"grid_thw,omitempty"`
}

const ProcessorOutputFormat = "processor_output"

// UnmarshalJSON accepts SGLang/HF tensor shapes for pixel_values and image_grid_thw.
func (p *ProcessorOutput) UnmarshalJSON(b []byte) error {
	var aux struct {
		Format         string          `json:"format,omitempty"`
		PixelValuesRaw json.RawMessage `json:"pixel_values"`
		ImageGridRaw   json.RawMessage `json:"image_grid_thw"`
		GridTHWRaw     json.RawMessage `json:"grid_thw"`
	}
	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}
	p.Format = aux.Format
	if len(aux.PixelValuesRaw) > 0 {
		pv, err := FlattenJSONFloats(aux.PixelValuesRaw)
		if err != nil {
			return err
		}
		p.PixelValues = pv
	}
	if len(aux.ImageGridRaw) > 0 || len(aux.GridTHWRaw) > 0 {
		thw, err := ParseGridTHW(aux.ImageGridRaw, aux.GridTHWRaw)
		if err != nil {
			return err
		}
		p.ImageGridTHW = thw
	}
	return nil
}

// GridTHWForProcessor returns [T,H,W] for runner ingest.
func (p ProcessorOutput) GridTHWForProcessor() []int {
	if len(p.ImageGridTHW) == 3 {
		return p.ImageGridTHW
	}
	return p.GridTHW
}

// GenerateRequest describes a request sent by [Client.Generate]. While you
// have to specify the Model and Prompt fields, all the other fields have
// reasonable defaults for basic uses.
type GenerateRequest struct {
	// Model is the model name; it should be a name familiar to Ollama from
	// the library at https://ollama.com/library
	Model string `json:"model"`

	// Prompt is the textual prompt to send to the model.
	Prompt string `json:"prompt"`

	// Suffix is the text that comes after the inserted text.
	Suffix string `json:"suffix"`

	// System overrides the model's default system message/prompt.
	System string `json:"system"`

	// Template overrides the model's default prompt template.
	Template string `json:"template"`

	// Context is the context parameter returned from a previous call to
	// [Client.Generate]. It can be used to keep a short conversational memory.
	Context []int `json:"context,omitempty"`

	// Stream specifies whether the response is streaming; it is true by default.
	Stream *bool `json:"stream,omitempty"`

	// Raw set to true means that no formatting will be applied to the prompt.
	Raw bool `json:"raw,omitempty"`

	// Format specifies the format to return a response in.
	Format json.RawMessage `json:"format,omitempty"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Images is an optional list of raw image bytes accompanying this
	// request, for multimodal models.
	Images []ImageData `json:"images,omitempty"`

	// Options lists model-specific options. For example, temperature can be
	// set through this field, if the model supports it.
	Options map[string]any `json:"options"`

	// Think controls whether thinking/reasoning models will think before
	// responding. Can be a boolean (true/false) or a string ("high", "medium", "low")
	// for supported models. Needs to be a pointer so we can distinguish between false
	// (request that thinking _not_ be used) and unset (use the old behavior
	// before this option was introduced)
	Think *ThinkValue `json:"think,omitempty"`

	// Truncate is a boolean that, when set to true, truncates the chat history messages
	// if the rendered prompt exceeds the context length limit.
	Truncate *bool `json:"truncate,omitempty"`

	// Shift is a boolean that, when set to true, shifts the chat history
	// when hitting the context length limit instead of erroring.
	Shift *bool `json:"shift,omitempty"`

	// DebugRenderOnly is a debug option that, when set to true, returns the rendered
	// template instead of calling the model.
	DebugRenderOnly bool `json:"_debug_render_only,omitempty"`

	// Logprobs specifies whether to return log probabilities of the output tokens.
	Logprobs bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number of most likely tokens to return at each token position,
	// each with an associated log probability. Only applies when Logprobs is true.
	// Valid values are 0-20. Default is 0 (only return the selected token's logprob).
	TopLogprobs int `json:"top_logprobs,omitempty"`

	// Experimental: Image generation fields (may change or be removed)

	// Width is the width of the generated image in pixels.
	// Only used for image generation models.
	Width int32 `json:"width,omitempty"`

	// Height is the height of the generated image in pixels.
	// Only used for image generation models.
	Height int32 `json:"height,omitempty"`

	// AspectRatio selects a preset shape when width/height are omitted: 16:9, 9:16, 3:2, 2:3, 1:1.
	// Only used for image generation models.
	AspectRatio string `json:"aspect_ratio,omitempty"`

	// Steps is the number of diffusion steps for image generation.
	// Only used for image generation models.
	Steps int32 `json:"steps,omitempty"`

	// EnableThinking is a harness alias. Mapped onto Think when Think is unset.
	EnableThinking *bool `json:"enable_thinking,omitempty"`

	// ChatTemplateKwargs carries template knobs. Unknown nested keys are rejected (trap 07/77).
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

// ChatRequest describes a request sent by [Client.Chat].
type ChatRequest struct {
	// Model is the model name, as in [GenerateRequest].
	Model string `json:"model"`

	// Messages is the messages of the chat - can be used to keep a chat memory.
	Messages []Message `json:"messages"`

	// Stream enables streaming of returned responses; true by default.
	Stream *bool `json:"stream,omitempty"`

	// Format is the format to return the response in (e.g. "json").
	Format json.RawMessage `json:"format,omitempty"`

	// KeepAlive controls how long the model will stay loaded into memory
	// following the request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Tools is an optional list of tools the model has access to.
	Tools `json:"tools,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`

	// Think controls whether thinking/reasoning models will think before
	// responding. Can be a boolean (true/false) or a string ("high", "medium", "low")
	// for supported models.
	Think *ThinkValue `json:"think,omitempty"`

	// Truncate is a boolean that, when set to true, truncates the chat history messages
	// if the rendered prompt exceeds the context length limit.
	Truncate *bool `json:"truncate,omitempty"`

	// Shift is a boolean that, when set to true, shifts the chat history
	// when hitting the context length limit instead of erroring.
	Shift *bool `json:"shift,omitempty"`

	// DebugRenderOnly is a debug option that, when set to true, returns the rendered
	// template instead of calling the model.
	DebugRenderOnly bool `json:"_debug_render_only,omitempty"`

	// Logprobs specifies whether to return log probabilities of the output tokens.
	Logprobs bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number of most likely tokens to return at each token position,
	// each with an associated log probability. Only applies when Logprobs is true.
	// Valid values are 0-20. Default is 0 (only return the selected token's logprob).
	TopLogprobs int `json:"top_logprobs,omitempty"`

	// EnableThinking is a harness alias (vLLM/SGLang). Mapped onto Think when Think is unset.
	EnableThinking *bool `json:"enable_thinking,omitempty"`

	// ChatTemplateKwargs carries template knobs. Unknown nested keys are rejected (trap 07/77).
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`

	// ThinkFromAlias is true when Think was derived from enable_thinking /
	// chat_template_kwargs rather than an explicit think / reasoning_effort field.
	// Non-thinking models ignore alias-only Think instead of HTTP 400 (minefield ceiling probes).
	ThinkFromAlias bool `json:"-"`
}

type Tools []Tool

func (t Tools) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

func (t Tool) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// Message is a single message in a chat sequence. The message contains the
// role ("system", "user", or "assistant"), the content and optional multimodal
// payloads (images, raw video bytes).
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Thinking contains the text that was inside thinking tags in the
	// original model output when ChatRequest.Think is enabled.
	Thinking string      `json:"thinking,omitempty"`
	Images   []ImageData `json:"images,omitempty"`
	// AudioClips holds raw audio blobs (OpenAI input_audio). Appended to Images at prompt time.
	AudioClips []AudioData `json:"audio_clips,omitempty"`
	// Videos holds undecoded video blobs (e.g. OpenAI video_url or /api/chat).
	// They are expanded to frames on Images before inference—see docs/video-understanding.md.
	Videos []VideoData `json:"videos,omitempty"`
	// VideoSpans is set when Videos were expanded: one entry per original blob, frame counts in order.
	// Why: templates still receive a flat Images list; spans preserve provenance for renderers that
	// must distinguish video-derived frames from unrelated stills (optional; see model/renderers).
	VideoSpans []VideoSpan `json:"video_spans,omitempty"`
	// PaddedInputIDs is optional SGLang-style pretokenized layout for pre-expanded multimodal turns.
	// Accepted for preflight and usage estimates; native render still uses images until wired.
	PaddedInputIDs []int `json:"padded_input_ids,omitempty"`
	// PrecomputedEmbeddings holds SGLang precomputed_embedding payloads (ViT rows). Populated from
	// images[] objects or precomputed_embeddings on the message. Mutually exclusive with raw Images bytes.
	PrecomputedEmbeddings []PrecomputedEmbedding `json:"precomputed_embeddings,omitempty"`
	// ProcessorOutputs holds SGLang processor_output payloads (pixel_values + grid). Populated from
	// images[] objects or processor_outputs on the message.
	ProcessorOutputs []ProcessorOutput `json:"processor_outputs,omitempty"`
	ToolCalls        []ToolCall        `json:"tool_calls,omitempty"`
	ToolName         string            `json:"tool_name,omitempty"`
	ToolCallID       string            `json:"tool_call_id,omitempty"`
}

func (m *Message) UnmarshalJSON(b []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}

	var imagesRaw json.RawMessage
	if v, ok := raw["images"]; ok {
		imagesRaw = v
		delete(raw, "images")
	}

	rest, err := json.Marshal(raw)
	if err != nil {
		return err
	}
	type messageAlias Message
	var aux messageAlias
	if err := json.Unmarshal(rest, &aux); err != nil {
		return err
	}
	*m = Message(aux)
	m.Role = strings.ToLower(m.Role)

	if len(imagesRaw) == 0 {
		return nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(imagesRaw, &items); err != nil {
		// tolerate a single image payload not wrapped in an array
		items = []json.RawMessage{imagesRaw}
	}

	m.Images = nil
	for _, item := range items {
		if len(item) == 0 {
			continue
		}
		if item[0] == '{' {
			var probe struct {
				Format string `json:"format"`
			}
			if err := json.Unmarshal(item, &probe); err != nil {
				return err
			}
			switch probe.Format {
			case ProcessorOutputFormat:
				var po ProcessorOutput
				if err := json.Unmarshal(item, &po); err != nil {
					return err
				}
				if len(po.PixelValues) == 0 {
					return fmt.Errorf("processor_output requires pixel_values")
				}
				if po.Format == "" {
					po.Format = ProcessorOutputFormat
				}
				m.ProcessorOutputs = append(m.ProcessorOutputs, po)
				continue
			case PrecomputedEmbeddingFormat:
				fallthrough
			case "":
				var generic map[string]json.RawMessage
				if err := json.Unmarshal(item, &generic); err != nil {
					return err
				}
				if _, ok := generic["pixel_values"]; ok && probe.Format == "" {
					var po ProcessorOutput
					if err := json.Unmarshal(item, &po); err != nil {
						return err
					}
					if len(po.PixelValues) == 0 {
						return fmt.Errorf("processor_output requires pixel_values")
					}
					po.Format = ProcessorOutputFormat
					m.ProcessorOutputs = append(m.ProcessorOutputs, po)
					continue
				}
				var pe PrecomputedEmbedding
				if err := json.Unmarshal(item, &pe); err != nil {
					return err
				}
				if pe.Format != "" && pe.Format != PrecomputedEmbeddingFormat {
					return fmt.Errorf("unsupported image format %q", pe.Format)
				}
				if len(pe.Feature) == 0 {
					return fmt.Errorf("precomputed_embedding requires non-empty feature")
				}
				if pe.Format == "" {
					pe.Format = PrecomputedEmbeddingFormat
				}
				m.PrecomputedEmbeddings = append(m.PrecomputedEmbeddings, pe)
				continue
			default:
				return fmt.Errorf("unsupported image format %q", probe.Format)
			}
		}
		var img ImageData
		if err := json.Unmarshal(item, &img); err != nil {
			return err
		}
		if len(img) > 0 {
			m.Images = append(m.Images, img)
		}
	}
	return nil
}

type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Index     int                       `json:"index"`
	Name      string                    `json:"name"`
	Arguments ToolCallFunctionArguments `json:"arguments"`
}

// ToolCallFunctionArguments holds tool call arguments in insertion order.
type ToolCallFunctionArguments struct {
	om *orderedmap.Map[string, any]
}

// NewToolCallFunctionArguments creates a new empty ToolCallFunctionArguments.
func NewToolCallFunctionArguments() ToolCallFunctionArguments {
	return ToolCallFunctionArguments{om: orderedmap.New[string, any]()}
}

// Get retrieves a value by key.
func (t *ToolCallFunctionArguments) Get(key string) (any, bool) {
	if t == nil || t.om == nil {
		return nil, false
	}
	return t.om.Get(key)
}

// Set sets a key-value pair, preserving insertion order.
func (t *ToolCallFunctionArguments) Set(key string, value any) {
	if t == nil {
		return
	}
	if t.om == nil {
		t.om = orderedmap.New[string, any]()
	}
	t.om.Set(key, value)
}

// Len returns the number of arguments.
func (t *ToolCallFunctionArguments) Len() int {
	if t == nil || t.om == nil {
		return 0
	}
	return t.om.Len()
}

// All returns an iterator over all key-value pairs in insertion order.
func (t *ToolCallFunctionArguments) All() iter.Seq2[string, any] {
	if t == nil || t.om == nil {
		return func(yield func(string, any) bool) {}
	}
	return t.om.All()
}

// ToMap returns a regular map (order not preserved).
func (t *ToolCallFunctionArguments) ToMap() map[string]any {
	if t == nil || t.om == nil {
		return nil
	}
	return t.om.ToMap()
}

func (t *ToolCallFunctionArguments) String() string {
	if t == nil || t.om == nil {
		return "{}"
	}
	bts, _ := json.Marshal(t.om)
	return string(bts)
}

func (t *ToolCallFunctionArguments) UnmarshalJSON(data []byte) error {
	t.om = orderedmap.New[string, any]()
	return json.Unmarshal(data, t.om)
}

func (t ToolCallFunctionArguments) MarshalJSON() ([]byte, error) {
	if t.om == nil {
		return []byte("{}"), nil
	}
	return json.Marshal(t.om)
}

type Tool struct {
	Type     string       `json:"type"`
	Items    any          `json:"items,omitempty"`
	Function ToolFunction `json:"function"`
}

// PropertyType can be either a string or an array of strings
type PropertyType []string

// UnmarshalJSON implements the json.Unmarshaler interface
func (pt *PropertyType) UnmarshalJSON(data []byte) error {
	// Try to unmarshal as a string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*pt = []string{s}
		return nil
	}

	// If that fails, try to unmarshal as an array of strings
	var a []string
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*pt = a
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (pt PropertyType) MarshalJSON() ([]byte, error) {
	if len(pt) == 1 {
		// If there's only one type, marshal as a string
		return json.Marshal(pt[0])
	}
	// Otherwise marshal as an array
	return json.Marshal([]string(pt))
}

// String returns a string representation of the PropertyType
func (pt PropertyType) String() string {
	if len(pt) == 0 {
		return ""
	}
	if len(pt) == 1 {
		return pt[0]
	}
	return fmt.Sprintf("%v", []string(pt))
}

// ToolPropertiesMap holds tool properties in insertion order.
type ToolPropertiesMap struct {
	om *orderedmap.Map[string, ToolProperty]
}

// NewToolPropertiesMap creates a new empty ToolPropertiesMap.
func NewToolPropertiesMap() *ToolPropertiesMap {
	return &ToolPropertiesMap{om: orderedmap.New[string, ToolProperty]()}
}

// Get retrieves a property by name.
func (t *ToolPropertiesMap) Get(key string) (ToolProperty, bool) {
	if t == nil || t.om == nil {
		return ToolProperty{}, false
	}
	return t.om.Get(key)
}

// Set sets a property, preserving insertion order.
func (t *ToolPropertiesMap) Set(key string, value ToolProperty) {
	if t == nil {
		return
	}
	if t.om == nil {
		t.om = orderedmap.New[string, ToolProperty]()
	}
	t.om.Set(key, value)
}

// Len returns the number of properties.
func (t *ToolPropertiesMap) Len() int {
	if t == nil || t.om == nil {
		return 0
	}
	return t.om.Len()
}

// All returns an iterator over all properties in insertion order.
func (t *ToolPropertiesMap) All() iter.Seq2[string, ToolProperty] {
	if t == nil || t.om == nil {
		return func(yield func(string, ToolProperty) bool) {}
	}
	return t.om.All()
}

// ToMap returns a regular map (order not preserved).
func (t *ToolPropertiesMap) ToMap() map[string]ToolProperty {
	if t == nil || t.om == nil {
		return nil
	}
	return t.om.ToMap()
}

func (t ToolPropertiesMap) MarshalJSON() ([]byte, error) {
	if t.om == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.om)
}

func (t *ToolPropertiesMap) UnmarshalJSON(data []byte) error {
	t.om = orderedmap.New[string, ToolProperty]()
	return json.Unmarshal(data, t.om)
}

type ToolProperty struct {
	AnyOf       []ToolProperty     `json:"anyOf,omitempty"`
	Type        PropertyType       `json:"type,omitempty"`
	Items       any                `json:"items,omitempty"`
	Description string             `json:"description,omitempty"`
	Enum        []any              `json:"enum,omitempty"`
	Properties  *ToolPropertiesMap `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
}

// ToTypeScriptType converts a ToolProperty to a TypeScript type string
func (tp ToolProperty) ToTypeScriptType() string {
	if len(tp.AnyOf) > 0 {
		var types []string
		for _, anyOf := range tp.AnyOf {
			types = append(types, anyOf.ToTypeScriptType())
		}
		return strings.Join(types, " | ")
	}

	if len(tp.Type) == 0 {
		return "any"
	}

	if len(tp.Type) == 1 {
		return mapToTypeScriptType(tp.Type[0])
	}

	var types []string
	for _, t := range tp.Type {
		types = append(types, mapToTypeScriptType(t))
	}
	return strings.Join(types, " | ")
}

// mapToTypeScriptType maps JSON Schema types to TypeScript types
func mapToTypeScriptType(jsonType string) string {
	switch jsonType {
	case "string":
		return "string"
	case "number", "integer":
		return "number"
	case "boolean":
		return "boolean"
	case "array":
		return "any[]"
	case "object":
		return "Record<string, any>"
	case "null":
		return "null"
	default:
		return "any"
	}
}

type ToolFunctionParameters struct {
	Type       string             `json:"type"`
	Defs       any                `json:"$defs,omitempty"`
	Items      any                `json:"items,omitempty"`
	Required   []string           `json:"required,omitempty"`
	Properties *ToolPropertiesMap `json:"properties"`
}

func (t *ToolFunctionParameters) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  ToolFunctionParameters `json:"parameters"`
}

func (t *ToolFunction) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// TokenLogprob represents log probability information for a single token alternative.
type TokenLogprob struct {
	// Token is the text representation of the token.
	Token string `json:"token"`

	// Logprob is the log probability of this token.
	Logprob float64 `json:"logprob"`

	// Bytes contains the raw byte representation of the token
	Bytes []int `json:"bytes,omitempty"`
}

// Logprob contains log probability information for a generated token.
type Logprob struct {
	TokenLogprob

	// TopLogprobs contains the most likely tokens and their log probabilities
	// at this position, if requested via TopLogprobs parameter.
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

// ChatResponse is the response returned by [Client.Chat]. Its fields are
// similar to [GenerateResponse].
type ChatResponse struct {
	// Model is the model name that generated the response.
	Model string `json:"model"`

	// RemoteModel is the name of the upstream model that generated the response.
	RemoteModel string `json:"remote_model,omitempty"`

	// RemoteHost is the URL of the upstream Ollama host that generated the response.
	RemoteHost string `json:"remote_host,omitempty"`

	// CreatedAt is the timestamp of the response.
	CreatedAt time.Time `json:"created_at"`

	// Message contains the message or part of a message from the model.
	Message Message `json:"message"`

	// Done specifies if the response is complete.
	Done bool `json:"done"`

	// DoneReason is the reason the model stopped generating text.
	DoneReason string `json:"done_reason,omitempty"`

	// PromptTruncated is true when input was shortened to fit num_ctx
	// (chatPrompt tail-trim and/or runner token trim / runtime context-shift detect).
	// Why on API: logs showed truncation while clients got silent 200 responses;
	// agents had to infer overflow from prompt_eval_count ≈ num_ctx.
	PromptTruncated bool `json:"prompt_truncated,omitempty"`
	// OriginalPromptTokens is the token count before prompt truncation (pre-drop size).
	// Prefer the largest known count (chatPrompt over runner) so megaprompts report ~44k not ~8k.
	OriginalPromptTokens int `json:"original_prompt_tokens,omitempty"`
	// MessagesTruncated is true when older chat messages were dropped to fit context.
	MessagesTruncated bool `json:"messages_truncated,omitempty"`
	// MessagesDropped is how many leading chat messages were removed.
	MessagesDropped int `json:"messages_dropped,omitempty"`

	GgmlNumCtx *GgmlNumCtx `json:"ggml_num_ctx,omitempty"`

	// Streaming progress (done=false, empty message): accepted, queued, loading, generating.
	Status     string `json:"status,omitempty"`
	Position   int    `json:"position,omitempty"`
	QueueDepth int    `json:"queue_depth,omitempty"`
	Detail     string `json:"detail,omitempty"`

	DebugInfo *DebugInfo `json:"_debug_info,omitempty"`

	// Logprobs contains log probability information for the generated tokens,
	// if requested via the Logprobs parameter.
	Logprobs []Logprob `json:"logprobs,omitempty"`

	Metrics
}

// DebugInfo contains debug information for template rendering
type DebugInfo struct {
	RenderedTemplate string `json:"rendered_template"`
	ImageCount       int    `json:"image_count,omitempty"`
	// PaddedInputIDsLen is latest-user pretokenized layout length when client sent padded_input_ids.
	PaddedInputIDsLen int `json:"padded_input_ids_len,omitempty"`
	// PaddedLayoutConsume is "deferred" when layout is acknowledged but native render still uses images.
	PaddedLayoutConsume string `json:"padded_layout_consume,omitempty"`
}

type Metrics struct {
	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
	// Multimodal token estimates (vision preflight heuristic; OpenAI usage.prompt_tokens_details).
	ImageTokens int `json:"image_tokens,omitempty"`
	VideoTokens int `json:"video_tokens,omitempty"`
	AudioTokens int `json:"audio_tokens,omitempty"`
	// CachedPromptTokens is prefix KV reused from llama-server cache_n / L3 cache_prompt.
	// Why separate from PromptEvalCount: OpenAI reports cached prefix in prompt_tokens_details.cached_tokens.
	CachedPromptTokens int `json:"cached_prompt_tokens,omitempty"`
	// CachedTokensHost is prefix KV served from host-tier cache (HiCache); 0 on native-only paths.
	CachedTokensHost int `json:"cached_tokens_host,omitempty"`
	// CachedTokensStorage is prefix KV served from L3 storage backend when wired.
	CachedTokensStorage        int    `json:"cached_tokens_storage,omitempty"`
	CachedTokensStorageBackend string `json:"cached_tokens_storage_backend,omitempty"`
	// CacheCreationTokens is prefix KV newly written this turn (vLLM #48535 /
	// Anthropic cache_creation_input_tokens). Distinct from CachedPromptTokens
	// (read hits at schedule time).
	CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
}

// Options specified in [GenerateRequest].  If you add a new option here, also
// add it to the API docs.
type Options struct {
	Runner

	// Predict options used at runtime
	NumKeep          int      `json:"num_keep,omitempty"`
	Seed             int      `json:"seed,omitempty"`
	NumPredict       int      `json:"num_predict,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	MinP             float32  `json:"min_p,omitempty"`
	TypicalP         float32  `json:"typical_p,omitempty"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	Stop             []string `json:"stop,omitempty"`
}

// Runner options which must be set when the model is loaded into memory
type Runner struct {
	NumCtx          int   `json:"num_ctx,omitempty"`
	NumBatch        int   `json:"num_batch,omitempty"`
	NumGPU          int   `json:"num_gpu,omitempty"`
	MainGPU         int   `json:"main_gpu,omitempty"`
	UseMMap         *bool `json:"use_mmap,omitempty"`
	NumThread       int   `json:"num_thread,omitempty"`
	DraftNumPredict int   `json:"draft_num_predict,omitempty"`

	// SpecType selects llama-server speculative decoding (e.g. ngram-simple, draft-mtp, draft-eagle3).
	SpecType string `json:"spec_type,omitempty"`
	// N-gram speculative tuning (--spec-ngram-simple-*).
	SpecNgramSizeN   int `json:"spec_ngram_size_n,omitempty"`
	SpecNgramSizeM   int `json:"spec_ngram_size_m,omitempty"`
	SpecNgramMinHits int `json:"spec_ngram_min_hits,omitempty"`

	// KvCacheType sets KV cache quantization for this load (e.g. f16, q8_0, q4_0).
	// Overrides OLLAMA_KV_CACHE_TYPE when set. Different requests may use different types.
	KvCacheType string `json:"kv_cache_type,omitempty"`
	// GgmlClampNumCtx lowers num_ctx to the VRAM-safe maximum for this request when true.
	GgmlClampNumCtx *bool `json:"ggml_clamp_num_ctx,omitempty"`
	// GgmlAutoKVQuant downgrades KV cache type (f16→q8_0→q4_0) when the load would exceed memory.
	GgmlAutoKVQuant *bool `json:"ggml_auto_kv_quant,omitempty"`

	// Flash-MoE (anemll-flash-llama.cpp): streamed routed experts via sidecar directory.
	MoeMode             string `json:"moe_mode,omitempty"`
	MoeSidecar          string `json:"moe_sidecar,omitempty"`
	MoeSlotBank         int    `json:"moe_slot_bank,omitempty"`
	MoeTopK             int    `json:"moe_topk,omitempty"`
	MoePrefetchTemporal *bool  `json:"moe_prefetch_temporal,omitempty"`
}

// KvCacheTypeEffective returns the KV cache type for load/estimate: request option, then
// OLLAMA_KV_CACHE_TYPE server default, then f16.
func (o Options) KvCacheTypeEffective() string {
	if kt := strings.ToLower(strings.TrimSpace(o.KvCacheType)); kt != "" {
		return kt
	}
	if kv := envconfig.KvCacheType(); kv != "" {
		return strings.ToLower(kv)
	}
	return "f16"
}

// GgmlClampNumCtxEnabled is true when this request opts in to VRAM-based num_ctx clamping.
func (o Options) GgmlClampNumCtxEnabled() bool {
	return o.GgmlClampNumCtx != nil && *o.GgmlClampNumCtx
}

// GgmlAutoKVQuantEnabled is true when this request opts in to automatic KV cache downgrade.
func (o Options) GgmlAutoKVQuantEnabled() bool {
	return o.GgmlAutoKVQuant != nil && *o.GgmlAutoKVQuant
}

// EmbedRequest is the request passed to [Client.Embed].
type EmbedRequest struct {
	// Model is the model name.
	Model string `json:"model"`

	// Input is the input to embed.
	Input any `json:"input"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Truncate truncates the input to fit the model's max sequence length.
	Truncate *bool `json:"truncate,omitempty"`

	// Dimensions truncates the output embedding to the specified dimension.
	Dimensions int `json:"dimensions,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbedResponse is the response from [Client.Embed].
type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`

	TotalDuration   time.Duration `json:"total_duration,omitempty"`
	LoadDuration    time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
}

// ScoreRequest scores fixed candidate continuations against a shared prompt.
// Why: agent routing / classifier models without full generation (LocalAI Score RPC pattern).
type ScoreRequest struct {
	// Model is the model name.
	Model string `json:"model"`

	// Prompt is the shared prefix; candidate strings are scored as continuations.
	Prompt string `json:"prompt"`

	// Candidates are the continuation strings to score (at least one required).
	Candidates []string `json:"candidates"`

	// LengthNormalize divides joint log-prob by candidate token count when true.
	LengthNormalize bool `json:"length_normalize,omitempty"`

	// IncludeTokenLogprobs returns per-token logprobs for each candidate.
	IncludeTokenLogprobs bool `json:"include_token_logprobs,omitempty"`

	// KeepAlive controls how long the model stays loaded after this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// CandidateScore is one scored continuation.
type CandidateScore struct {
	Candidate               string         `json:"candidate"`
	LogProb                 float64        `json:"log_prob"`
	LengthNormalizedLogProb float64        `json:"length_normalized_log_prob,omitempty"`
	NumTokens               int            `json:"num_tokens"`
	Tokens                  []TokenLogprob `json:"tokens,omitempty"`
}

// ScoreResponse is the response from POST /api/score.
type ScoreResponse struct {
	Model      string           `json:"model"`
	Candidates []CandidateScore `json:"candidates"`
}

// EmbeddingRequest is the request passed to [Client.Embeddings].
type EmbeddingRequest struct {
	// Model is the model name.
	Model string `json:"model"`

	// Prompt is the textual prompt to embed.
	Prompt string `json:"prompt"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbeddingResponse is the response from [Client.Embeddings].
type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

// CreateRequest is the request passed to [Client.Create].
type CreateRequest struct {
	// Model is the model name to create.
	Model string `json:"model"`

	// Stream specifies whether the response is streaming; it is true by default.
	Stream *bool `json:"stream,omitempty"`

	// Quantize is the quantization format for the model; leave blank to not change the quantization level.
	Quantize string `json:"quantize,omitempty"`

	// From is the name of the model or file to use as the source.
	From string `json:"from,omitempty"`

	// RemoteHost is the URL of the upstream ollama API for the model (if any).
	RemoteHost string `json:"remote_host,omitempty"`

	// Files is a map of files include when creating the model.
	Files map[string]string `json:"files,omitempty"`

	// Adapters is a map of LoRA adapters to include when creating the model.
	Adapters map[string]string `json:"adapters,omitempty"`

	// DraftFiles is a map of draft model files to include when creating the model.
	DraftFiles map[string]string `json:"draft_files,omitempty"`

	// DraftQuantize is the quantization format for draft model tensors.
	DraftQuantize string `json:"draft_quantize,omitempty"`

	// Template is the template used when constructing a request to the model.
	Template string `json:"template,omitempty"`

	// License is a string or list of strings for licenses.
	License any `json:"license,omitempty"`

	// System is the system prompt for the model.
	System string `json:"system,omitempty"`

	// Parameters is a map of hyper-parameters which are applied to the model.
	Parameters map[string]any `json:"parameters,omitempty"`

	// Messages is a list of messages added to the model before chat and generation requests.
	Messages []Message `json:"messages,omitempty"`

	Renderer string `json:"renderer,omitempty"`
	Parser   string `json:"parser,omitempty"`

	// Requires is the minimum version of Ollama required by the model.
	Requires string `json:"requires,omitempty"`

	// Info is a map of additional information for the model
	Info map[string]any `json:"info,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
	// Deprecated: use Quantize instead
	Quantization string `json:"quantization,omitempty"`
}

// DeleteRequest is the request passed to [Client.Delete].
type DeleteRequest struct {
	Model string `json:"model"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowRequest is the request passed to [Client.Show].
type ShowRequest struct {
	Model  string `json:"model"`
	System string `json:"system"`

	// Template is deprecated
	Template string `json:"template"`
	Verbose  bool   `json:"verbose"`

	Options map[string]any `json:"options"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowResponse is the response returned from [Client.Show].
type ShowResponse struct {
	License       string             `json:"license,omitempty"`
	Modelfile     string             `json:"modelfile,omitempty"`
	Parameters    string             `json:"parameters,omitempty"`
	Template      string             `json:"template,omitempty"`
	System        string             `json:"system,omitempty"`
	Renderer      string             `json:"renderer,omitempty"`
	Parser        string             `json:"parser,omitempty"`
	Details       ModelDetails       `json:"details,omitempty"`
	Messages      []Message          `json:"messages,omitempty"`
	RemoteModel   string             `json:"remote_model,omitempty"`
	RemoteHost    string             `json:"remote_host,omitempty"`
	ModelInfo     map[string]any     `json:"model_info"`
	ProjectorInfo map[string]any     `json:"projector_info,omitempty"`
	Tensors       []Tensor           `json:"tensors,omitempty"`
	Capabilities  []model.Capability `json:"capabilities,omitempty"`
	ModifiedAt    time.Time          `json:"modified_at,omitempty"`
	Requires      string             `json:"requires,omitempty"`
	// GgmlNumCtx is VRAM suggest for ggml models (M12); see scheduling-vram-policy.md.
	GgmlNumCtx *GgmlNumCtx `json:"ggml_num_ctx,omitempty"`
}

// CopyRequest is the request passed to [Client.Copy].
type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

// RepairRequest is the request for POST /api/repair.
type RepairRequest struct {
	Model  string   `json:"model,omitempty"`
	Models []string `json:"models,omitempty"`
	All    bool     `json:"all,omitempty"`
	Write  bool     `json:"write,omitempty"`
}

// RepairChange is one proposed manifest metadata update.
type RepairChange struct {
	Field string `json:"field"`
	From  string `json:"from,omitempty"`
	To    string `json:"to,omitempty"`
}

// RepairResult summarizes repair for one model tag.
type RepairResult struct {
	Name    string         `json:"name"`
	Skipped bool           `json:"skipped,omitempty"`
	Reason  string         `json:"reason,omitempty"`
	Changes []RepairChange `json:"changes,omitempty"`
	Written bool           `json:"written,omitempty"`
}

// RepairResponse is the response from POST /api/repair.
type RepairResponse struct {
	Results []RepairResult `json:"results"`
}

// PullRequest is the request passed to [Client.Pull].
type PullRequest struct {
	Model    string `json:"model"`
	Source   string `json:"source,omitempty"`   // huggingface:// URI; Model is the local tag name
	Insecure bool   `json:"insecure,omitempty"` // Deprecated: ignored
	Username string `json:"username"`           // Deprecated: ignored
	Password string `json:"password"`           // Deprecated: ignored
	Stream   *bool  `json:"stream,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ProgressResponse is the response passed to progress functions like
// [PullProgressFunc] and [PushProgressFunc].
type ProgressResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// PushRequest is the request passed to [Client.Push].
type PushRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure,omitempty"`
	Username string `json:"username"`
	Password string `json:"password"`
	Stream   *bool  `json:"stream,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ListResponse is the response from [Client.List].
type ListResponse struct {
	Models []ListModelResponse `json:"models"`
}

// ProcessResponse is the response from [Client.Process].
type ProcessResponse struct {
	Models []ProcessModelResponse `json:"models"`
	// Pending is the ggml scheduler prompt queue depth (waiting for a runner).
	// Includes requests for models not yet listed under Models (cold load / eviction).
	Pending int `json:"pending"`
}

// ListModelResponse is a single model description in [ListResponse].
type ListModelResponse struct {
	Name         string             `json:"name"`
	Model        string             `json:"model"`
	RemoteModel  string             `json:"remote_model,omitempty"`
	RemoteHost   string             `json:"remote_host,omitempty"`
	ModifiedAt   time.Time          `json:"modified_at"`
	Size         int64              `json:"size"`
	Digest       string             `json:"digest"`
	Details      ModelDetails       `json:"details,omitempty"`
	Capabilities []model.Capability `json:"capabilities,omitempty"`
}

// ProcessModelResponse is a single model description in [ProcessResponse].
type ProcessModelResponse struct {
	Name           string                `json:"name"`
	Model          string                `json:"model"`
	Size           int64                 `json:"size"`
	Digest         string                `json:"digest"`
	Details        ModelDetails          `json:"details,omitempty"`
	ExpiresAt      time.Time             `json:"expires_at"`
	SizeVRAM       int64                 `json:"size_vram"`
	ContextLength  int                   `json:"context_length"`
	Pending        int                   `json:"pending,omitempty"` // queued prompts waiting on this model
	LoadedMetadata *LoadedModelMetadata  `json:"loaded_metadata,omitempty"`
	Zerollama      *ProcessZerollamaInfo `json:"zerollama,omitempty"`
}

// ProcessZerollamaInfo is zerollama-specific /api/ps metadata for loaded runners.
type ProcessZerollamaInfo struct {
	Sessions []ProcessSessionInfo `json:"sessions,omitempty"`
}

// ProcessSessionInfo describes a hot or in-flight prompt_cache_key session on a runner.
type ProcessSessionInfo struct {
	SessionKey    string    `json:"session_key,omitempty"`
	SessionClass  string    `json:"session_class,omitempty"`
	SessionGroup  string    `json:"session_group,omitempty"`
	SessionParent string    `json:"session_parent,omitempty"`
	ProjectID     string    `json:"project_id,omitempty"`
	ProjectName   string    `json:"project_name,omitempty"`
	CacheScope    string    `json:"cache_scope,omitempty"`
	CacheLevel    string    `json:"cache_level,omitempty"`
	Fulfillment   string    `json:"fulfillment,omitempty"`
	Inflight      int       `json:"inflight,omitempty"`
	HotUntil      time.Time `json:"hot_until,omitempty"`
}

// LoadedModelMetadata is probed from the runner after load (ground truth vs manifest).
// Why: manifest num_ctx and parser often drift; fleet and /api/ps expose effective
// values without re-reading weights on every request. See docs/localai-borrowings.md.
type LoadedModelMetadata struct {
	NumCtx             int    `json:"num_ctx"`
	TrainContextLength int    `json:"train_context_length,omitempty"`
	ManifestNumCtx     int    `json:"manifest_num_ctx,omitempty"`
	NumParallel        int    `json:"num_parallel,omitempty"`
	NumGPU             int    `json:"num_gpu,omitempty"`
	Backend            string `json:"backend,omitempty"`
	Parser             string `json:"parser,omitempty"`
	SupportsThinking   bool   `json:"supports_thinking,omitempty"`
	SupportsTools      bool   `json:"supports_tools,omitempty"`
	HasChatTemplate    bool   `json:"has_chat_template,omitempty"`
	// GPULayersOffloaded / GPULayersTotal are parsed from llama-server
	// "offloaded N/M layers to GPU" (minefield trap 97). Zero total means unknown.
	GPULayersOffloaded int       `json:"gpu_layers_offloaded,omitempty"`
	GPULayersTotal     int       `json:"gpu_layers_total,omitempty"`
	ProbedAt           time.Time `json:"probed_at"`
}

type TokenResponse struct {
	Token string `json:"token"`
}

type CloudStatus struct {
	Disabled bool   `json:"disabled"`
	Source   string `json:"source"`
}

// GgmlStatus is the ggml scheduler snapshot on GET /api/status (fleet polling).
type GgmlStatus struct {
	Pending            int                     `json:"pending"`
	Active             int                     `json:"active"`
	Loaded             int                     `json:"loaded"`
	LoadsPaused        bool                    `json:"loads_paused"`
	Loading            bool                    `json:"loading"`
	AssignHolds        int                     `json:"assign_holds,omitempty"` // F5 soft holds (not yet consumed by chat/generate)
	LoadedModels       []string                `json:"loaded_models,omitempty"`
	LoadedModelDetails []GgmlLoadedModelStatus `json:"loaded_model_details,omitempty"`
}

// GgmlLoadedModelStatus is per-resident-runner metadata for fleet routing and ops.
type GgmlLoadedModelStatus struct {
	Name string `json:"name"`
	LoadedModelMetadata
}

// RuntimeStatus is the Python runtime sidecar snapshot on GET /api/status.
// Queue fields are omitted when enabled is true but available is false (probe failed).
type RuntimeStatus struct {
	Enabled     bool   `json:"enabled"`
	Available   bool   `json:"available"`
	Waiting     *int   `json:"waiting,omitempty"`
	Running     *int   `json:"running,omitempty"`
	LlamaLoaded *bool  `json:"llama_loaded,omitempty"`
	State       string `json:"state,omitempty"`
	// Radix mirrors Python /health.kv_resume.prefix_block_pool (L3-R8 / LA13).
	Radix *RadixMirrorStatus `json:"radix,omitempty"`
}

// RadixMirrorStatus is a control-plane mirror of Python Radix / prefix block pool health.
// Why: fleet assign and /api/status must see L3 residency without curling :8081.
type RadixMirrorStatus struct {
	Enabled           bool `json:"enabled"`
	EntryCount        int  `json:"entry_count"`
	SlotCount         int  `json:"slot_count,omitempty"`
	MultiHolderBlocks int  `json:"multi_holder_blocks,omitempty"`
	BlobDigestBlocks  int  `json:"blob_digest_blocks,omitempty"`
	// BlockHashes is a capped newest-first sample from Python prefix_block_pool (L3-R9 / LA13).
	BlockHashes []string `json:"block_hashes,omitempty"`
	// BlobDigests is a capped sample of content-addressed slot digests (L3-R11 / peer pull).
	BlobDigests    []string `json:"blob_digests,omitempty"`
	RadixShare     bool     `json:"radix_share"`
	SeqCpMode      string   `json:"seq_cp_mode,omitempty"`
	KvUnified      bool     `json:"kv_unified,omitempty"`
	LmcacheBlobs   bool     `json:"lmcache_blobs,omitempty"`
	L3R6MetadataOK *bool    `json:"l3_r6_metadata_ok,omitempty"`
}

// TrainingQueuePolicy summarizes T6 training submit gates on GET /api/status.
type TrainingQueuePolicy struct {
	WaitInferenceIdle          bool   `json:"wait_inference_idle"`
	WaitGgmlLoaded             bool   `json:"wait_ggml_loaded"`
	WaitFailClosed             bool   `json:"wait_fail_closed"`
	QueueOnBusy                bool   `json:"queue_on_busy"`
	AllowedWindow              string `json:"allowed_window,omitempty"`
	AllowedWindowEnabled       bool   `json:"allowed_window_enabled"`
	AllowedWindowMisconfigured bool   `json:"allowed_window_misconfigured,omitempty"`
	CrossQueueFifo             bool   `json:"cross_queue_fifo"`
	DeferWaiting               int    `json:"defer_waiting,omitempty"`
	DeferTracked               int    `json:"defer_tracked,omitempty"`
}

// TrainingStatus is the training-side snapshot on GET /api/status.
type TrainingStatus struct {
	QueuePolicy TrainingQueuePolicy `json:"queue_policy"`
}

// InferenceConfigStatus advertises effective scheduler knobs on GET /api/status
// so clients (Orient Inventory / Decide) need not guess env on the host.
type InferenceConfigStatus struct {
	NumParallel           uint   `json:"num_parallel"`
	NumParallelAuto       bool   `json:"num_parallel_auto,omitempty"`
	MaxLoadedModels       uint   `json:"max_loaded_models"`
	MaxLoadedConfigured   uint   `json:"max_loaded_configured"`
	MaxQueue              uint   `json:"max_queue"`
	KeepAlive             string `json:"keep_alive"`
	LoadTimeout           string `json:"load_timeout"`
	RuntimeMaxQueue       *uint  `json:"runtime_max_queue,omitempty"`
	SameModelMultiCopy    bool   `json:"same_model_multi_copy"`
	ResidencyOwner        string `json:"residency_owner"`
	NumParallelMeansSlots bool   `json:"num_parallel_means_slots"`
}

// InferenceStatus summarizes local inference load for fleet management polling.
type InferenceStatus struct {
	Ggml     GgmlStatus            `json:"ggml"`
	Runtime  RuntimeStatus         `json:"runtime"`
	Backend  BackendPolicy         `json:"backend"`
	Config   InferenceConfigStatus `json:"config"`
	Pins     []PinStatus           `json:"pins,omitempty"`
	Training *TrainingStatus       `json:"training,omitempty"`
}

// CanLoadRequest is the body for POST /api/can-load (capacity dry-run).
// Why a dedicated endpoint: loopback /internal/vram-estimate was not a public product API.
type CanLoadRequest struct {
	Model   string         `json:"model"`
	Options map[string]any `json:"options,omitempty"`
}

// CanLoadQueueSnapshot is queue pressure included in CanLoadResponse.
type CanLoadQueueSnapshot struct {
	GgmlPending     int  `json:"ggml_pending"`
	GgmlMaxQueue    uint `json:"ggml_max_queue"`
	RuntimeWaiting  int  `json:"runtime_waiting,omitempty"`
	RuntimeMaxQueue uint `json:"runtime_max_queue,omitempty"`
}

// CanLoadResponse is the dry-run result for POST /api/can-load (always HTTP 200).
//
// Why fields: can_load vs needs_eviction are orthogonal (admit-by-swap vs thrash-free);
// confidence separates runtime VRAM math from ggml heuristics; already_loaded is
// path-matched (not "any llama warm") so single-resident runtimes do not lie.
type CanLoadResponse struct {
	Model              string               `json:"model"`
	Backend            string               `json:"backend"`
	CanLoad            bool                 `json:"can_load"`
	Confidence         string               `json:"confidence"` // exact | heuristic
	AlreadyLoaded      bool                 `json:"already_loaded"`
	NeedsEviction      bool                 `json:"needs_eviction"`
	EvictionReason     string               `json:"eviction_reason,omitempty"`
	Busy               bool                 `json:"busy"`
	BusyReason         string               `json:"busy_reason,omitempty"`
	LoadsPaused        bool                 `json:"loads_paused"`
	Queue              CanLoadQueueSnapshot `json:"queue"`
	Warm               ProcessResponse      `json:"warm"`
	VramEstimate       map[string]any       `json:"vram_estimate,omitempty"`
	VramBudget         map[string]any       `json:"vram_budget,omitempty"`
	SuggestedMaxNumCtx *int                 `json:"suggested_max_num_ctx,omitempty"`
	MaxLoadedModels    uint                 `json:"max_loaded_models"`
	LoadedCount        int                  `json:"loaded_count"`
	Notes              string               `json:"notes,omitempty"`
}

// PinRequest is the body for POST /api/pin (session eviction lease; does not load).
// WHY no load: reserve residency intent for Orient/Decide without GetRunner cost.
type PinRequest struct {
	Models     []string `json:"models"`
	TTLSeconds *int     `json:"ttl_seconds,omitempty"`
	ProjectID  string   `json:"project_id,omitempty"`
}

// PinResponse is returned by POST /api/pin.
// CanPin false + error notes when multi-runtime GGUF or budget exceeded (fail closed).
type PinResponse struct {
	PinID      string    `json:"pin_id"`
	ExpiresAt  time.Time `json:"expires_at"`
	Models     []string  `json:"models"`
	ProjectID  string    `json:"project_id,omitempty"`
	CoResident bool      `json:"co_resident"`
	CanPin     bool      `json:"can_pin"`
	Notes      string    `json:"notes,omitempty"`
}

// PinStatus is one active pin on GET /api/status → inference.pins.
type PinStatus struct {
	PinID      string    `json:"pin_id"`
	Models     []string  `json:"models"`
	ExpiresAt  time.Time `json:"expires_at"`
	ProjectID  string    `json:"project_id,omitempty"`
	CoResident bool      `json:"co_resident"`
	Notes      string    `json:"notes,omitempty"`
}

// ProposeLoadRequest is the body for POST /api/propose-load.
type ProposeLoadRequest struct {
	Models []CanLoadRequest `json:"models"`
}

// ProposeLoadPlan aggregates batch can-load into an honest co-residency plan.
// WHY serialize_required: Python holds one GGUF — never imply two runtime models stay warm.
type ProposeLoadPlan struct {
	FitsWithoutEviction bool     `json:"fits_without_eviction"`
	CoResident          bool     `json:"co_resident"`
	SerializeRequired   bool     `json:"serialize_required"`
	LoadOrder           []string `json:"load_order"`
	EvictCandidates     []string `json:"evict_candidates,omitempty"`
	Confidence          string   `json:"confidence"` // exact | heuristic | mixed
	Notes               string   `json:"notes,omitempty"`
}

// ProposeLoadResponse is the dry-run multi-model plan (never loads).
type ProposeLoadResponse struct {
	Models []CanLoadResponse `json:"models"`
	Plan   ProposeLoadPlan   `json:"plan"`
	Warm   ProcessResponse   `json:"warm"`
}

// BackendPolicy describes Phase 16/17 local GGUF routing for GET /api/status.
type BackendPolicy struct {
	Edge            bool   `json:"edge"`
	EdgeBuild       bool   `json:"edge_build"`
	GgmlLinked      bool   `json:"ggml_linked"`
	LlamaServer     string `json:"llama_server"`              // off | auto | explicit
	SpecAutoRoute   bool   `json:"spec_auto_route,omitempty"` // Darwin: spec tags → llama-server when binary present
	LlamaCppHarness bool   `json:"llama_cpp_harness,omitempty"`
	RuntimeChat     string `json:"runtime_chat"` // on | off
	GgufPath        string `json:"gguf_path"`    // ggml | llama-server | runtime | mixed
}

// StatusResponse is the response from GET /api/status.
type StatusResponse struct {
	Cloud     CloudStatus     `json:"cloud"`
	Inference InferenceStatus `json:"inference"`
}

// GenerateResponse is the response passed into [GenerateResponseFunc].
type GenerateResponse struct {
	// Model is the model name that generated the response.
	Model string `json:"model"`

	// RemoteModel is the name of the upstream model that generated the response.
	RemoteModel string `json:"remote_model,omitempty"`

	// RemoteHost is the URL of the upstream Ollama host that generated the response.
	RemoteHost string `json:"remote_host,omitempty"`

	// CreatedAt is the timestamp of the response.
	CreatedAt time.Time `json:"created_at"`

	// Response is the textual response itself.
	Response string `json:"response"`

	// Thinking contains the text that was inside thinking tags in the
	// original model output when ChatRequest.Think is enabled.
	Thinking string `json:"thinking,omitempty"`

	// Done specifies if the response is complete.
	Done bool `json:"done"`

	// DoneReason is the reason the model stopped generating text.
	DoneReason string `json:"done_reason,omitempty"`

	// PromptTruncated is true when input was shortened to fit num_ctx
	// (chatPrompt tail-trim and/or runner token trim / runtime context-shift detect).
	// Why on API: logs showed truncation while clients got silent 200 responses;
	// agents had to infer overflow from prompt_eval_count ≈ num_ctx.
	PromptTruncated bool `json:"prompt_truncated,omitempty"`
	// OriginalPromptTokens is the token count before prompt truncation (pre-drop size).
	// Prefer the largest known count (chatPrompt over runner) so megaprompts report ~44k not ~8k.
	OriginalPromptTokens int `json:"original_prompt_tokens,omitempty"`
	// MessagesTruncated is true when older chat messages were dropped to fit context.
	MessagesTruncated bool `json:"messages_truncated,omitempty"`
	// MessagesDropped is how many leading chat messages were removed.
	MessagesDropped int `json:"messages_dropped,omitempty"`

	GgmlNumCtx *GgmlNumCtx `json:"ggml_num_ctx,omitempty"`

	// Streaming progress (done=false, empty response): accepted, queued, loading, generating.
	Status     string `json:"status,omitempty"`
	Position   int    `json:"position,omitempty"`
	QueueDepth int    `json:"queue_depth,omitempty"`
	Detail     string `json:"detail,omitempty"`

	// Context is an encoding of the conversation used in this response; this
	// can be sent in the next request to keep a conversational memory.
	Context []int `json:"context,omitempty"`

	Metrics

	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	DebugInfo *DebugInfo `json:"_debug_info,omitempty"`

	// Logprobs contains log probability information for the generated tokens,
	// if requested via the Logprobs parameter.
	Logprobs []Logprob `json:"logprobs,omitempty"`

	// Experimental: Image generation fields (may change or be removed)

	// Image contains a base64-encoded generated image.
	// Only present for image generation models.
	Image string `json:"image,omitempty"`

	// Completed is the number of completed steps in image generation.
	// Only present for image generation models during streaming.
	Completed int64 `json:"completed,omitempty"`

	// Total is the total number of steps for image generation.
	// Only present for image generation models during streaming.
	Total int64 `json:"total,omitempty"`
}

// GgmlNumCtx surfaces VRAM-aware num_ctx suggest/clamp on the ggml scheduler path (M12).
// Field semantics differ by endpoint — see docs/scheduling-vram-policy.md.
type GgmlNumCtx struct {
	// SuggestedMaxNumCtx is the largest num_ctx estimate that fits free VRAM (show + load).
	SuggestedMaxNumCtx int `json:"suggested_max_num_ctx,omitempty"`
	// MergedNumCtx is the merged server/manifest default when it exceeds SuggestedMaxNumCtx (show only).
	// Why separate from NumCtx: NumCtx on load means effective after clamp, not "requested default".
	MergedNumCtx      int  `json:"merged_num_ctx,omitempty"`
	NumCtxClamped     bool `json:"num_ctx_clamped,omitempty"`
	NumCtxClampedFrom int  `json:"num_ctx_clamped_from,omitempty"`
	// NumCtx is effective context after opt-in clamp (chat/generate when clamped).
	NumCtx int `json:"num_ctx,omitempty"`

	// EstimatedLoadBytes is the estimated memory needed for the requested num_ctx (weights + KV + graph).
	EstimatedLoadBytes uint64 `json:"estimated_load_bytes,omitempty"`
	// AvailableBytes is the free device memory at schedule time.
	AvailableBytes uint64 `json:"available_bytes,omitempty"`
	// ExceedsAvailable is true when the estimated load would exceed available memory.
	ExceedsAvailable bool `json:"exceeds_available,omitempty"`
	// SuggestedKVCacheType is a KV quantization type (e.g. "q8_0") that would reduce
	// KV memory enough to fit the requested context. Set kv_cache_type or ggml_auto_kv_quant on the request.
	SuggestedKVCacheType string `json:"suggested_kv_cache_type,omitempty"`
	// KVCacheTypeDowngraded is set when ggml_auto_kv_quant was true on the request and the
	// scheduler automatically applied a quantized KV cache to fit context in memory.
	KVCacheTypeDowngraded bool   `json:"kv_cache_type_downgraded,omitempty"`
	KVCacheTypeFrom       string `json:"kv_cache_type_from,omitempty"`
	KVCacheType           string `json:"kv_cache_type,omitempty"`
}

// ModelDetails provides details about a model.
type ModelDetails struct {
	ParentModel          string   `json:"parent_model"`
	Format               string   `json:"format"`
	Family               string   `json:"family"`
	Families             []string `json:"families"`
	ParameterSize        string   `json:"parameter_size"`
	QuantizationLevel    string   `json:"quantization_level"`
	ContextLength        int      `json:"context_length,omitempty"`
	EmbeddingLength      int      `json:"embedding_length,omitempty"`
	ArchitectureType     string   `json:"architecture_type,omitempty"`
	ParameterCount       uint64   `json:"parameter_count,omitempty"`
	ActiveParameterCount uint64   `json:"active_parameter_count,omitempty"`
	ExpertCount          uint32   `json:"expert_count,omitempty"`
	ExpertUsedCount      uint32   `json:"expert_used_count,omitempty"`
}

// UserResponse provides information about a user.
type UserResponse struct {
	ID        uuid.UUID `json:"id"`
	Email     string    `json:"email"`
	Name      string    `json:"name"`
	Bio       string    `json:"bio,omitempty"`
	AvatarURL string    `json:"avatarurl,omitempty"`
	FirstName string    `json:"firstname,omitempty"`
	LastName  string    `json:"lastname,omitempty"`
	Plan      string    `json:"plan,omitempty"`
}

// Tensor describes the metadata for a given tensor.
type Tensor struct {
	Name  string   `json:"name"`
	Type  string   `json:"type"`
	Shape []uint64 `json:"shape"`
}

func (m *Metrics) Summary() {
	if m.TotalDuration > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", m.TotalDuration)
	}

	if m.LoadDuration > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", m.LoadDuration)
	}

	if m.PromptEvalCount > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", m.PromptEvalCount)
	}

	if m.PromptEvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", m.PromptEvalDuration)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n", float64(m.PromptEvalCount)/m.PromptEvalDuration.Seconds())
	}

	if m.EvalCount > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", m.EvalCount)
	}

	if m.EvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", m.EvalDuration)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(m.EvalCount)/m.EvalDuration.Seconds())
	}
}

func coerceInt(val any) (int64, bool) {
	switch t := val.(type) {
	case int:
		return int64(t), true
	case int32:
		return int64(t), true
	case int64:
		return t, true
	case float64:
		// JSON unmarshals numbers as float64.
		return int64(t), true
	case float32:
		return int64(t), true
	default:
		return 0, false
	}
}

func coerceFloat64(val any) (float64, bool) {
	switch t := val.(type) {
	case float64:
		return t, true
	case float32:
		return float64(t), true
	case int:
		return float64(t), true
	case int64:
		return float64(t), true
	default:
		return 0, false
	}
}

func coerceStringSlice(val any) ([]string, error) {
	switch t := val.(type) {
	case []string:
		return t, nil
	case []any:
		slice := make([]string, len(t))
		for i, item := range t {
			str, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("option must be of an array of strings")
			}
			slice[i] = str
		}
		return slice, nil
	default:
		return nil, fmt.Errorf("option must be of type array")
	}
}

func (opts *Options) FromMap(m map[string]any) error {
	valueOpts := reflect.ValueOf(opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts).Elem()   // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	for key, val := range m {
		opt, ok := jsonOpts[key]
		if !ok {
			// Suppress noise for known pass-through keys handled elsewhere
			// (e.g. eliza metadata used by EnsureAgentPromptCacheKey,
			// prompt_cache_key / cache_seed used by L3 slot bridge).
			switch key {
			case "eliza", "prompt_cache_key", "cache_seed":
			default:
				slog.Warn("invalid option provided", "option", key)
			}
			continue
		}

		field := valueOpts.FieldByName(opt.Name)
		if field.IsValid() && field.CanSet() {
			if val == nil {
				continue
			}

			switch field.Kind() {
			case reflect.Int:
				n, ok := coerceInt(val)
				if !ok {
					return fmt.Errorf("option %q must be of type integer", key)
				}
				field.SetInt(n)
			case reflect.Bool:
				val, ok := val.(bool)
				if !ok {
					return fmt.Errorf("option %q must be of type boolean", key)
				}
				field.SetBool(val)
			case reflect.Float32:
				f, ok := coerceFloat64(val)
				if !ok {
					return fmt.Errorf("option %q must be of type float32", key)
				}
				field.SetFloat(f)
			case reflect.String:
				val, ok := val.(string)
				if !ok {
					return fmt.Errorf("option %q must be of type string", key)
				}
				field.SetString(val)
			case reflect.Slice:
				slice, err := coerceStringSlice(val)
				if err != nil {
					return fmt.Errorf("option %q must be of type array", key)
				}
				field.Set(reflect.ValueOf(slice))
			case reflect.Pointer:
				var b bool
				if field.Type() == reflect.TypeOf(&b) {
					val, ok := val.(bool)
					if !ok {
						return fmt.Errorf("option %q must be of type boolean", key)
					}
					field.Set(reflect.ValueOf(&val))
				} else {
					return fmt.Errorf("unknown type loading config params: %v %v", field.Kind(), field.Type())
				}
			default:
				return fmt.Errorf("unknown type loading config params: %v", field.Kind())
			}
		}
	}

	return nil
}

// DefaultOptions is the default set of options for [GenerateRequest]; these
// values are used unless the user specifies other values explicitly.
func DefaultOptions() Options {
	return Options{
		// options set on request to runner
		NumPredict: -1,

		// set a minimal num_keep to avoid issues on context shifts
		NumKeep:          4,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.9,
		TypicalP:         1.0,
		RepeatLastN:      64,
		RepeatPenalty:    1.1,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		Seed:             -1,

		Runner: Runner{
			// options set when the model is loaded
			NumCtx:          int(envconfig.ContextLength()),
			NumBatch:        512,
			NumGPU:          -1, // -1 here indicates that NumGPU should be set dynamically
			NumThread:       0,  // let the runtime decide
			DraftNumPredict: 4,
			UseMMap:         nil,
		},
	}
}

// ThinkValue represents a value that can be a boolean or a string ("high", "medium", "low")
type ThinkValue struct {
	// Value can be a bool or string
	Value interface{}
}

// IsValid checks if the ThinkValue is valid
func (t *ThinkValue) IsValid() bool {
	if t == nil || t.Value == nil {
		return true // nil is valid (means not set)
	}

	switch v := t.Value.(type) {
	case bool:
		return true
	case string:
		return v == "high" || v == "medium" || v == "low"
	default:
		return false
	}
}

// IsBool returns true if the value is a boolean
func (t *ThinkValue) IsBool() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(bool)
	return ok
}

// IsString returns true if the value is a string
func (t *ThinkValue) IsString() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(string)
	return ok
}

// Bool returns the value as a bool (true if enabled in any way)
func (t *ThinkValue) Bool() bool {
	if t == nil || t.Value == nil {
		return false
	}

	switch v := t.Value.(type) {
	case bool:
		return v
	case string:
		// Any string value ("high", "medium", "low") means thinking is enabled
		return v == "high" || v == "medium" || v == "low"
	default:
		return false
	}
}

// String returns the value as a string
func (t *ThinkValue) String() string {
	if t == nil || t.Value == nil {
		return ""
	}

	switch v := t.Value.(type) {
	case string:
		return v
	case bool:
		if v {
			return "medium" // Default level when just true
		}
		return ""
	default:
		return ""
	}
}

// UnmarshalJSON implements json.Unmarshaler
func (t *ThinkValue) UnmarshalJSON(data []byte) error {
	// Try to unmarshal as bool first
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		t.Value = b
		return nil
	}

	// Try to unmarshal as string
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		// Validate string values
		if s != "high" && s != "medium" && s != "low" {
			return fmt.Errorf("invalid think value: %q (must be \"high\", \"medium\", \"low\", true, or false)", s)
		}
		t.Value = s
		return nil
	}

	return fmt.Errorf("think must be a boolean or string (\"high\", \"medium\", \"low\", true, or false)")
}

// MarshalJSON implements json.Marshaler
func (t *ThinkValue) MarshalJSON() ([]byte, error) {
	if t == nil || t.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.Value)
}

type Duration struct {
	time.Duration
}

func (d Duration) MarshalJSON() ([]byte, error) {
	if d.Duration < 0 {
		return []byte("-1"), nil
	}
	return []byte("\"" + d.Duration.String() + "\""), nil
}

func (d *Duration) UnmarshalJSON(b []byte) (err error) {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	d.Duration = 5 * time.Minute

	switch t := v.(type) {
	case float64:
		if t < 0 {
			d.Duration = time.Duration(math.MaxInt64)
		} else {
			d.Duration = time.Duration(t * float64(time.Second))
		}
	case string:
		d.Duration, err = time.ParseDuration(t)
		if err != nil {
			return err
		}
		if d.Duration < 0 {
			d.Duration = time.Duration(math.MaxInt64)
		}
	default:
		return fmt.Errorf("Unsupported type: '%s'", reflect.TypeOf(v))
	}

	return nil
}

// FormatParams converts specified parameter options to their correct types
func FormatParams(params map[string][]string) (map[string]any, error) {
	opts := Options{}
	valueOpts := reflect.ValueOf(&opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts)           // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	out := make(map[string]any)
	// iterate params and set values based on json struct tags
	for key, vals := range params {
		if opt, ok := jsonOpts[key]; !ok {
			return nil, fmt.Errorf("unknown parameter '%s'", key)
		} else {
			field := valueOpts.FieldByName(opt.Name)
			if field.IsValid() && field.CanSet() {
				switch field.Kind() {
				case reflect.Float32:
					floatVal, err := strconv.ParseFloat(vals[0], 32)
					if err != nil {
						return nil, fmt.Errorf("invalid float value %s", vals)
					}

					out[key] = float32(floatVal)
				case reflect.Int:
					intVal, err := strconv.ParseInt(vals[0], 10, 64)
					if err != nil {
						return nil, fmt.Errorf("invalid int value %s", vals)
					}

					out[key] = intVal
				case reflect.Bool:
					boolVal, err := strconv.ParseBool(vals[0])
					if err != nil {
						return nil, fmt.Errorf("invalid bool value %s", vals)
					}

					out[key] = boolVal
				case reflect.String:
					out[key] = vals[0]
				case reflect.Slice:
					// TODO: only string slices are supported right now
					out[key] = vals
				case reflect.Pointer:
					var b bool
					if field.Type() == reflect.TypeOf(&b) {
						boolVal, err := strconv.ParseBool(vals[0])
						if err != nil {
							return nil, fmt.Errorf("invalid bool value %s", vals)
						}
						out[key] = &boolVal
					} else {
						return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
					}
				default:
					return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
				}
			}
		}
	}

	return out, nil
}
