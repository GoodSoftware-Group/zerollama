package llm

import (
	"context"
	"encoding/json"
	"io"
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
	// ASR forces llama-server for Parakeet / Nemotron-transcribe GGUFs. The
	// encoder graph is mtmd inside llama-server, including on Darwin where
	// plain GGUF stays on the ggml Metal runner.
	ASR bool
	// Audio forces llama-server for mtmd file-audio projectors (Voxtral Small,
	// Qwen2.5-Omni, Ultravox, Qwen audio, Gemma 4 audio, LFM2 audio, Phi-4).
	Audio bool
	// Realtime forces llama-server for a streaming audio encoder
	// (voxtral_rt, lfm3a) and for GET /v1/realtime. A ggml runner has no socket.
	Realtime bool
	// ReasoningOff passes --reasoning off. LFM3-Omni's template thinks, and
	// the checkpoint is not trained to.
	ReasoningOff bool
	// SingleGPU passes --split-mode none so a realtime model occupies one GPU.
	// Offline audio that needs both cards (Voxtral Small Q8, Qwen3-Omni Q8,
	// Ultravox 70B Q4) stays on layer split unless main_gpu is set.
	SingleGPU bool
	// QwenVL asks llama-server for the Qwen vision image-token floor. Qwen3-Omni's
	// text architecture is not qwen3vl; the merger lives on the mmproj.
	QwenVL bool
	// Speech forces llama-server for a gen-audio projector. POST /tts is there.
	Speech bool
}

// LlamaASRTranscriber is implemented by the llama-server subprocess runner.
type LlamaASRTranscriber interface {
	// stream asks llama-server for text/event-stream. The caller closes the body.
	TranscribeAudio(ctx context.Context, audio []byte, filename, language, prompt, temperature, maxTokens string, stream bool) (contentType string, body io.ReadCloser, err error)
}

// SpeechSample is optional acoustic sampling for llama-server POST /tts.
// Nil pointers are omitted. The server then keeps top_k 0, top_p 0, and repeat_penalty 1.05.
type SpeechSample struct {
	TopK          *int
	TopP          *float64
	RepeatPenalty *float64
}

// LlamaSpeaker is implemented by the llama-server subprocess runner.
// Speak posts text to POST /tts. The reader is wav or pcm and stays open
// until the caller closes it. stream sends chunks as they are synthesized.
type LlamaSpeaker interface {
	Speak(ctx context.Context, text, voice, format, lang string, speakerRef []byte, stream bool, seed *int64, maxPredict *int, sample SpeechSample) (contentType string, audio io.ReadCloser, err error)
}

type realtimeLoadKey struct{}

// WithRealtimeLoad marks ctx so the scheduler loads this model on llama-server.
// The realtime WebSocket is only implemented there.
func WithRealtimeLoad(ctx context.Context) context.Context {
	return context.WithValue(ctx, realtimeLoadKey{}, true)
}

// RealtimeLoadFromContext reports whether ctx was marked with WithRealtimeLoad.
func RealtimeLoadFromContext(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	v, _ := ctx.Value(realtimeLoadKey{}).(bool)
	return v
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
	// Images first, then audio clips. Native llama-server chat builds content
	// parts from Media only; leaving AudioClips off dropped input_audio for
	// Voxtral, Ultravox, Qwen-Omni, Phi-4, and LFM3.
	media := make([]MediaData, 0, len(msg.Images)+len(msg.AudioClips))
	for _, data := range msg.Images {
		media = append(media, NewMediaData(len(media), data))
	}
	for _, data := range msg.AudioClips {
		clip := NewMediaData(len(media), data)
		clip.Kind = MediaKindAudio
		media = append(media, clip)
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
	Message               api.Message   `json:"message"`
	DoneReason            DoneReason    `json:"done_reason"`
	Done                  bool          `json:"done"`
	PromptEvalCount       int           `json:"prompt_eval_count"`
	PromptEvalCachedCount int           `json:"prompt_eval_cached_count,omitempty"`
	PromptEvalDuration    time.Duration `json:"prompt_eval_duration"`
	EvalCount             int           `json:"eval_count"`
	EvalDuration          time.Duration `json:"eval_duration"`
	Logprobs              []Logprob     `json:"logprobs,omitempty"`
}
