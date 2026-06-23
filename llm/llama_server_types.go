package llm

import (
	"encoding/json"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

const (
	llamaServerStreamInitialBufferSize = 64 * 1024
	llamaServerStreamMaxBufferSize     = 8 * format.MegaByte
)

// LlamaServerConfig controls optional llama-server subprocess behavior (upstream shape).
type LlamaServerConfig struct {
	DisableJinja   bool
	ContextShift   bool
	EnableMTP      bool
	DraftModelPath string
	SpecType       string
	NgramSizeN     int
	NgramSizeM     int
	NgramMinHits   int
}

type MediaKind string

const (
	MediaKindUnknown MediaKind = ""
	MediaKindImage   MediaKind = "image"
	MediaKindAudio   MediaKind = "audio"
)

type MediaData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
	Kind MediaKind
}

type Message struct {
	Role       string
	Content    string
	Thinking   string
	Media      []MediaData
	ToolCalls  []api.ToolCall
	ToolName   string
	ToolCallID string
}

func MessageFromAPI(msg api.Message) Message {
	media := make([]MediaData, len(msg.Images))
	for i, data := range msg.Images {
		media[i] = NewMediaData(i, data)
	}

	return Message{
		Role:       msg.Role,
		Content:    msg.Content,
		Thinking:   msg.Thinking,
		Media:      media,
		ToolCalls:  msg.ToolCalls,
		ToolName:   msg.ToolName,
		ToolCallID: msg.ToolCallID,
	}
}

type ChatRequest struct {
	Messages []api.Message
	Tools    api.Tools
	Format   json.RawMessage
	Options  *api.Options
	Think    *api.ThinkValue
	Shift    bool

	Logprobs    bool
	TopLogprobs int
}

type ChatResponse struct {
	Message            api.Message   `json:"message"`
	DoneReason         DoneReason    `json:"done_reason"`
	Done               bool          `json:"done"`
	PromptEvalCount       int           `json:"prompt_eval_count"`
	PromptEvalCachedCount int           `json:"prompt_eval_cached_count,omitempty"`
	PromptEvalDuration    time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
	Logprobs           []Logprob     `json:"logprobs,omitempty"`
}
