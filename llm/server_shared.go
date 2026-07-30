// server_shared.go holds llm package types and helpers used by both default and edge builds.
//
// WHY a separate file (no build tag): edge builds still need LoadModel (pure Go GGUF header decode),
// CompletionRequest/Response wire types, and ggmlRunnerRequired — but must not compile server.go's
// llama.cpp CGO subprocess path. Splitting shared surface here keeps one API for server/, discover/,
// and llama_server.go without duplicating structs across build tags.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

type filteredEnv []string

func (e filteredEnv) LogValue() slog.Value {
	var attrs []slog.Attr
	for _, env := range e {
		if key, value, ok := strings.Cut(env, "="); ok {
			if filteredEnvLogKey(key) {
				attrs = append(attrs, slog.String(key, filteredEnvLogValue(key, value)))
			}
		}
	}
	return slog.GroupValue(attrs...)
}

func filteredEnvLogKey(key string) bool {
	return strings.HasPrefix(key, "CUDA_") ||
		strings.HasPrefix(key, "ROCR_") ||
		strings.HasPrefix(key, "ROCM_") ||
		strings.HasPrefix(key, "HIP_") ||
		strings.HasPrefix(key, "HSA_") ||
		strings.HasPrefix(key, "GGML_") ||
		slices.Contains([]string{
			"PATH",
			"LD_LIBRARY_PATH",
			"DYLD_LIBRARY_PATH",
		}, key)
}

func filteredEnvLogValue(key, value string) string {
	for _, token := range []string{"API", "KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "CREDENTIAL", "AUTH"} {
		if strings.Contains(strings.ToUpper(key), token) {
			return "[redacted]"
		}
	}
	return value
}

// LlamaServer is implemented by ggml subprocess runners and llama-server subprocess runners.
type LlamaServer interface {
	ModelPath() string
	Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error)
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	ApplyChatTemplate(ctx context.Context, req ChatRequest) (string, error)
	Embedding(ctx context.Context, input string) ([]float32, int, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	MemorySize() (total, vram uint64)
	VRAMByGPU(id ml.DeviceID) uint64
	Pid() int
	GetPort() int
	GetDeviceInfos(ctx context.Context) []ml.DeviceInfo
	HasExited() bool
	ContextLength() int
}

// LoadModel will load a model from disk. The model must be in the GGML format.
func LoadModel(model string, maxArraySize int) (*ggml.GGML, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, err := ggml.Decode(f, maxArraySize)
	return ggml, err
}

// LoadModelMetadata reads GGUF headers without tokenizer vocab bodies or a full tensor weight walk.
func LoadModelMetadata(model string) (*ggml.GGML, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return ggml.DecodeMetadata(f)
}

func useOllamaEngine(f *ggml.GGML) bool {
	if !envconfig.GgmlRunnerLinked() {
		return false
	}
	return pickOllamaEngine(f.KV().OllamaEngineRequired())
}

func pickOllamaEngine(ollamaRequired bool) bool {
	return ollamaRequired
}

// ErrGgmlRunnerUnlinked is returned when an edge build cannot load GGUF without llama-server.
var ErrGgmlRunnerUnlinked = errors.New("ggml runner not linked in edge build")

func ggmlRunnerRequired(projectors []string, config LlamaServerConfig) error {
	if envconfig.GgmlRunnerLinked() || useLlamaServerBackendForModel(projectors, config) {
		return nil
	}
	return fmt.Errorf("%w; use llama-server (ZEROLLAMA_LLAMA_SERVER=1/auto, --edge, or --llama-server-backend)", ErrGgmlRunnerUnlinked)
}

type LoadOperation int

const (
	LoadOperationFit LoadOperation = iota
	LoadOperationAlloc
	LoadOperationCommit
	LoadOperationClose
)

func (o LoadOperation) String() string {
	switch o {
	case LoadOperationFit:
		return "fit"
	case LoadOperationAlloc:
		return "alloc"
	case LoadOperationCommit:
		return "commit"
	case LoadOperationClose:
		return "close"
	default:
		return "unknown"
	}
}

type LoadRequest struct {
	Operation LoadOperation

	LoraPath       []string
	Parallel       int
	BatchSize      int
	FlashAttention ml.FlashAttentionType
	KvSize         int
	KvCacheType    string
	NumThreads     int
	GPULayers      ml.GPULayersList
	MultiUserCache bool

	ProjectorPath string
	MainGPU       int
	UseMmap       bool
}

type LoadResponse struct {
	Success bool
	Memory  ml.BackendMemory
}

var ErrLoadRequiredFull = errors.New("unable to load full model on GPU")

type ServerStatus int

const (
	ServerStatusReady ServerStatus = iota
	ServerStatusNoSlotsAvailable
	ServerStatusLaunched
	ServerStatusLoadingModel
	ServerStatusNotResponding
	ServerStatusError
)

func (s ServerStatus) String() string {
	switch s {
	case ServerStatusReady:
		return "llm server ready"
	case ServerStatusNoSlotsAvailable:
		return "llm busy - no slots available"
	case ServerStatusLaunched:
		return "llm server launched"
	case ServerStatusLoadingModel:
		return "llm server loading model"
	case ServerStatusNotResponding:
		return "llm server not responding"
	default:
		return "llm server error"
	}
}

type ServerStatusResponse struct {
	Status   ServerStatus `json:"status"`
	Progress float32      `json:"progress"`
}

const maxBufferSize = 512 * format.KiloByte

type ImageData struct {
	Data    []byte `json:"data"`
	ID      int    `json:"id"`
	GridTHW []int  `json:"grid_thw,omitempty"`
	// PrecomputedFeature is SGLang precomputed_embedding rows ([vision_tokens][hidden]).
	// When set, runners skip ViT encode and inject these embeds at padded layout slots.
	PrecomputedFeature [][]float32 `json:"precomputed_feature,omitempty"`
	// ProcessorPixelValues is SGLang processor_output pixel_values (HF patch tensor, flat).
	ProcessorPixelValues []float32 `json:"processor_pixel_values,omitempty"`
}

// HasPrecomputedEmbedding reports whether this attachment carries client-supplied ViT rows.
func (img ImageData) HasPrecomputedEmbedding() bool {
	return len(img.PrecomputedFeature) > 0
}

// HasProcessorOutput reports whether this attachment carries HF processor pixel_values.
func (img ImageData) HasProcessorOutput() bool {
	return len(img.ProcessorPixelValues) > 0
}

type CompletionRequest struct {
	Prompt  string
	Format  json.RawMessage
	Images  []ImageData
	Media   []MediaData
	Options *api.Options

	PromptTokens        []int  `json:"prompt_tokens,omitempty"`
	PaddedLayoutConsume string `json:"padded_layout_consume,omitempty"`
	PromptCacheKey      string `json:"prompt_cache_key,omitempty"`
	// CacheReset forces a miss under the same PromptCacheKey for this request.
	CacheReset bool `json:"cache_reset,omitempty"`
	// SessionViTOverlay enables SGLang-style per-thread ViT embed pinning (see modality.SessionViTOverlayEnabled).
	SessionViTOverlay   bool `json:"session_vit_overlay,omitempty"`
	Gemma4PaddedMedia   Gemma4PaddedMediaSchedule `json:"gemma4_padded_media,omitempty"`

	Grammar         string
	Shift           bool
	Truncate        bool
	PreservedTokens []string
	ToolCallTag     string
	LeadingBOS      string

	Logprobs    bool
	TopLogprobs int

	Width       int32  `json:"width,omitempty"`
	Height      int32  `json:"height,omitempty"`
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Steps       int32  `json:"steps,omitempty"`
	Seed        int64  `json:"seed,omitempty"`
}

type DoneReason int

const (
	DoneReasonStop DoneReason = iota
	DoneReasonLength
	DoneReasonConnectionClosed
)

func (d DoneReason) String() string {
	switch d {
	case DoneReasonLength:
		return "length"
	case DoneReasonStop:
		return "stop"
	default:
		return ""
	}
}

type TokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

type Logprob struct {
	TokenLogprob
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

type CompletionResponse struct {
	Content               string        `json:"content"`
	DoneReason            DoneReason    `json:"done_reason"`
	Done                  bool          `json:"done"`
	PrefillProcessed      int           `json:"prefill_processed,omitempty"`
	PrefillTotal          int           `json:"prefill_total,omitempty"`
	PromptEvalCount       int           `json:"prompt_eval_count"`
	PromptEvalCachedCount int           `json:"prompt_eval_cached_count,omitempty"`
	// HiCache-shaped tiers (SGLang sglext); device is PromptEvalCachedCount.
	PromptEvalCachedHost           int    `json:"prompt_eval_cached_host,omitempty"`
	PromptEvalCachedStorage        int    `json:"prompt_eval_cached_storage,omitempty"`
	PromptEvalCachedStorageBackend string `json:"cached_tokens_storage_backend,omitempty"`
	// PromptEvalCacheCreationCount — tokens newly written to prefix cache this turn.
	PromptEvalCacheCreationCount int `json:"prompt_eval_cache_creation_count,omitempty"`
	PromptEvalDuration             time.Duration `json:"prompt_eval_duration"`
	EvalCount                      int           `json:"eval_count"`
	EvalDuration                   time.Duration `json:"eval_duration"`

	// Why: clients need an explicit overflow signal; logs alone left silent 200s.
	PromptTruncated      bool `json:"prompt_truncated,omitempty"`
	OriginalPromptTokens int  `json:"original_prompt_tokens,omitempty"`

	Logprobs []Logprob `json:"logprobs,omitempty"`

	// Tokens is the sampled prompt+generated id list when the runner provides
	// it (MLX Done chunk). Prefer over re-tokenizing response text (F0686).
	Tokens []int `json:"tokens,omitempty"`

	// PreemptedReason explains done_reason=preempted (M15f soft mid-stream preempt).
	PreemptedReason string `json:"preempted_reason,omitempty"`

	Image      string `json:"image,omitempty"`
	Step       int    `json:"step,omitempty"`
	TotalSteps int    `json:"total_steps,omitempty"`
}

func (c *CompletionResponse) UnmarshalJSON(data []byte) error {
	type alias CompletionResponse
	aux := struct {
		alias
		CachedPromptTokens         int    `json:"cached_prompt_tokens,omitempty"`
		CachedTokensHost           int    `json:"cached_tokens_host,omitempty"`
		CachedTokensStorage        int    `json:"cached_tokens_storage,omitempty"`
		CachedTokensStorageBackend string `json:"cached_tokens_storage_backend,omitempty"`
		CacheCreationTokens        int    `json:"cache_creation_tokens,omitempty"`
		CreatedCacheTokens         int    `json:"created_cache_tokens,omitempty"`
	}{}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*c = CompletionResponse(aux.alias)
	if c.PromptEvalCachedCount == 0 && aux.CachedPromptTokens > 0 {
		c.PromptEvalCachedCount = aux.CachedPromptTokens
	}
	if c.PromptEvalCachedHost == 0 && aux.CachedTokensHost > 0 {
		c.PromptEvalCachedHost = aux.CachedTokensHost
	}
	if c.PromptEvalCachedStorage == 0 && aux.CachedTokensStorage > 0 {
		c.PromptEvalCachedStorage = aux.CachedTokensStorage
	}
	if c.PromptEvalCachedStorageBackend == "" && aux.CachedTokensStorageBackend != "" {
		c.PromptEvalCachedStorageBackend = aux.CachedTokensStorageBackend
	}
	if c.PromptEvalCacheCreationCount == 0 {
		if aux.CacheCreationTokens > 0 {
			c.PromptEvalCacheCreationCount = aux.CacheCreationTokens
		} else if aux.CreatedCacheTokens > 0 {
			c.PromptEvalCacheCreationCount = aux.CreatedCacheTokens
		}
	}
	return nil
}

type EmbeddingRequest struct {
	Content string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding       []float32 `json:"embedding"`
	PromptEvalCount int       `json:"prompt_eval_count"`
}

var grammarJSON = `
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
         string ":" ws value
    ("," ws string ":" ws value)*
  )? ws "}" 
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? ws "]" 
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" 
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? 
# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`
