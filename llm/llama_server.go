// llama_server.go wraps the llama-server binary as a subprocess
//
// Ollama uses two chat paths with llama-server. Models with explicit Ollama
// renderers/parsers, Harmony handling, MLX, or an enabled Go TEMPLATE layer
// still render prompts in Go and call /completion. Other GGUF chat models use
// llama-server's chat_template handling through /v1/chat/completions.
//
// For structured output, JSON schemas are passed directly to llama-server via
// its json_schema field (avoiding the CGO SchemaToGrammar dependency). Raw BNF
// grammars are passed via the grammar field.
//
// llama-server auto-detects GPU layers (-ngl), thread count (-t), and flash
// attention (--flash-attn).
package llm

import (
	"bufio"
	"bytes"
	"context"
	crand "crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/image/webp"
	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// DefaultEmbeddingNumBatch is the default NumBatch used for embedding models
// when neither the model nor the request specifies num_batch.
const (
	DefaultEmbeddingNumBatch             = 2048
	openEndedGenerationContextMultiplier = 10
)

const (
	llamaArgFitTargetEnv = "LLAMA_ARG_FIT_TARGET"
	bytesPerMiB          = 1 << 20

	// mmprojOffloadHeadroom leaves 1 GiB for backend buffers beyond projector weights.
	mmprojOffloadHeadroom = 1 << 30
)

// DefaultEmbeddingNumBatchForContext caps the embedding batch default to the
// active context length before it is passed to llama-server.
func DefaultEmbeddingNumBatchForContext(numCtx int) int {
	if numCtx > 0 {
		return min(DefaultEmbeddingNumBatch, numCtx)
	}
	return DefaultEmbeddingNumBatch
}

// WithDefaultEmbeddingNumBatch applies the llama-server embedding batch
// default to a copy of opts.
func WithDefaultEmbeddingNumBatch(opts api.Options) api.Options {
	opts.NumBatch = DefaultEmbeddingNumBatchForContext(opts.NumCtx)
	return opts
}

func boundedNumPredict(numPredict, numCtx int) int {
	if numCtx <= 0 {
		return numPredict
	}
	// Ollama's default num_predict=-1 means "generate until a stop condition".
	// llama-server still needs a finite request budget, so keep open-ended
	// generations bounded while allowing several full context windows.
	limit := openEndedGenerationContextMultiplier * numCtx
	if numPredict < 0 || numPredict > limit {
		return limit
	}
	return numPredict
}

// llamaServerRunner wraps an upstream llama-server process and implements the LlamaServer interface.
// It communicates with llama-server over HTTP.
type llamaServerRunner struct {
	port               int
	cmd                *exec.Cmd
	done               chan struct{}
	doneErr            error
	client             *http.Client
	memoryMu           sync.RWMutex
	memTotal           uint64 // actual total buffer size parsed from llama-server logs (bytes)
	memGPU             uint64 // actual GPU buffer size parsed from llama-server logs (bytes)
	memModelFileBacked uint64 // model weight bytes mirroring on-disk file (mmap + device copies)
	memCPUMappedModel  uint64 // mmap-backed CPU model buffers (e.g. CPU_Mapped)
	gpuLayers          uint64 // model layers loaded on GPU, parsed from llama-server logs
	gpuLayerOverflow   int    // number of GPU-selected layers partially overflowed to CPU
	status             *StatusWriter
	options            api.Options
	modelPath          string
	// mediaMarker must match the LLAMA_MEDIA_MARKER value passed to llama-server.
	// llama.cpp randomizes this by default; Ollama renders stable [img-N] markers
	// and rewrites them before forwarding the request.
	mediaMarker string

	// Per-device VRAM tracking, populated from llama-server log parsing.
	// Keys are device names from llama-server output (e.g., "CUDA0", "ROCm0", "MTL0").
	vramByDevice map[string]uint64

	// System-reported free VRAM per device at model load time, parsed from
	// "using device CUDA0 ... - 15221 MiB free" log lines. This reflects
	// real system state including external VRAM consumers (on platforms where
	// the GPU driver reports accurately). Keys match vramByDevice (e.g., "CUDA0").
	systemFreeAtLoad map[string]uint64

	// gpus is the list of GPU devices assigned to this runner at creation time,
	// used to map DeviceIDs to device names for VRAMByGPU lookups.
	gpus []ml.DeviceInfo

	ggml          *ggml.GGML
	totalLayers   uint64 // maximum offloadable model layers
	loadStart     time.Time
	loadActivity  atomic.Int64
	loadTracking  atomic.Bool
	rawEmbeddings bool

	sem *semaphore.Weighted

	launch                  llamaServerLaunchConfig
	output                  *memoryParsingWriter
	mmprojOffloadOOMRetried bool

	// Recorded at spawn time so a later rebuild of llama-server (same path,
	// newer mtime) can be detected without restarting zerollama serve itself.
	serverBinPath    string
	serverBinModTime time.Time
}

type llamaServerLaunchConfig struct {
	modelPath            string
	modelArch            string
	projectors           []string
	mmprojMemory         uint64
	modelLayers          uint64
	adapters             []string
	opts                 api.Options
	numParallel          int
	kvCacheType          string
	ubatchSize           int     // 0 → same as opts.NumBatch (L1 may set ub≠b)
	forceFlashAttn       bool    // L1 profile flash_attn when OLLAMA_FLASH_ATTENTION unset
	draftPMin            float64 // L1 draft_p_min → --spec-draft-p-min when speculative on
	gpuProfileID         string
	embedding            bool
	config               LlamaServerConfig
	gpus                 []ml.DeviceInfo
	gpuLibs              []string
	extraEnvs            map[string]string
	forceNoMMProjOffload bool
	ggufKV               ggml.KV // raw GGUF KV for feature detection
}

func newLlamaServerHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DisableKeepAlives: true,
			Proxy:             nil,
		},
	}
}

var defaultLlamaServerHTTPClient = newLlamaServerHTTPClient()

func (s *llamaServerRunner) httpClient() *http.Client {
	if s.client != nil {
		return s.client
	}
	return defaultLlamaServerHTTPClient
}

func (s *llamaServerRunner) ModelPath() string {
	return s.modelPath
}

func (s *llamaServerRunner) Pid() int {
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return 0
}

func (s *llamaServerRunner) GetPort() int {
	return s.port
}

func (s *llamaServerRunner) HasExited() bool {
	return s.cmd != nil && s.cmd.ProcessState != nil && s.cmd.ProcessState.ExitCode() >= 0
}

// BinaryStale reports whether the llama-server binary this runner was spawned
// from has since been rebuilt in place (same resolved path, newer mtime) —
// e.g. via ./scripts/build/build_llama_server.sh or ./scripts/build/build_zerollama_mac.sh
// while this runner was still alive. The scheduler uses this to force a
// reload instead of silently keeping a stale subprocess running forever.
func (s *llamaServerRunner) BinaryStale() bool {
	if s.serverBinPath == "" || s.serverBinModTime.IsZero() {
		return false
	}
	exe, err := FindLlamaServer()
	if err != nil || exe != s.serverBinPath {
		// Path resolution changed (e.g. env override added/removed) — let the
		// normal reload paths handle that; don't force an unload here.
		return false
	}
	st, err := os.Stat(exe)
	if err != nil {
		return false
	}
	return st.ModTime().After(s.serverBinModTime)
}

func (s *llamaServerRunner) llamaServerMediaMarker() string {
	if s.mediaMarker != "" {
		return s.mediaMarker
	}
	return "<__media__>"
}

func newLlamaServerMediaMarker() string {
	var b [16]byte
	if _, err := crand.Read(b[:]); err == nil {
		return fmt.Sprintf("<__ollama_media_%x__>", b)
	}

	return fmt.Sprintf("<__ollama_media_%d_%d__>", time.Now().UnixNano(), rand.Int63())
}

func (s *llamaServerRunner) completionPrompt(prompt, leadingBOS string) string {
	if s.tokenizerAddsBOS() {
		if leadingBOS != "" && strings.HasPrefix(prompt, leadingBOS) {
			return strings.TrimPrefix(prompt, leadingBOS)
		}

		if strings.HasPrefix(prompt, "<bos>") {
			return strings.TrimPrefix(prompt, "<bos>")
		}
	}

	return prompt
}

func (s *llamaServerRunner) tokenizerAddsBOS() bool {
	if s.ggml == nil {
		return false
	}

	kv := s.ggml.KV()

	if kv.String("tokenizer.ggml.pre") == "lfm2" {
		return true
	}

	// llama.cpp forces add_bos on for Gemma4 at load time, even for GGUFs
	// whose tokenizer.ggml.add_bos_token metadata is explicitly false. Some
	// GGUFs omit tokenizer.ggml.pre and are still treated as Gemma4 from
	// tokenizer.ggml.model.
	if kv.String("tokenizer.ggml.pre") == "gemma4" || kv.String("tokenizer.ggml.model") == "gemma4" {
		return true
	}

	return kv.Bool("tokenizer.ggml.add_bos_token")
}

func (s *llamaServerRunner) gemma4SoftTokens(ctx context.Context) (Gemma4SoftTokens, error) {
	imageSlot, err := s.gemma4SlotToken(ctx, gemma4ImagePlaceholder)
	if err != nil {
		return Gemma4SoftTokens{}, err
	}
	videoSlot, err := s.gemma4SlotToken(ctx, gemma4VideoPlaceholder)
	if err != nil {
		return Gemma4SoftTokens{}, err
	}
	audioSlot, err := s.gemma4SlotToken(ctx, gemma4AudioPlaceholder)
	if err != nil {
		return Gemma4SoftTokens{}, err
	}
	return Gemma4SoftTokens{Image: imageSlot, Video: videoSlot, Audio: audioSlot}, nil
}

func (s *llamaServerRunner) gemma4SlotToken(ctx context.Context, placeholder string) (int, error) {
	parseSpecial := true
	toks, err := s.tokenize(ctx, placeholder, false, &parseSpecial)
	if err != nil {
		return 0, fmt.Errorf("tokenize gemma4 placeholder %q: %w", placeholder, err)
	}
	if len(toks) == 0 {
		return 0, fmt.Errorf("gemma4 placeholder %q tokenized to empty ids", placeholder)
	}
	return toks[len(toks)-1], nil
}

func (s *llamaServerRunner) placeholderToken(ctx context.Context, placeholder string) (int, error) {
	parseSpecial := true
	toks, err := s.tokenize(ctx, placeholder, false, &parseSpecial)
	if err != nil {
		return 0, err
	}
	if len(toks) == 0 {
		return 0, nil
	}
	return toks[len(toks)-1], nil
}

func (s *llamaServerRunner) lfm2VisionTokens(ctx context.Context) (Lfm2VisionTokens, error) {
	slots := Lfm2VisionTokens{Image: 396}
	if id, err := s.placeholderToken(ctx, "<image>"); err == nil && id != 0 {
		slots.Image = id
	}
	start, err := s.placeholderToken(ctx, "<|image_start|>")
	if err != nil {
		return Lfm2VisionTokens{}, err
	}
	end, err := s.placeholderToken(ctx, "<|image_end|>")
	if err != nil {
		return Lfm2VisionTokens{}, err
	}
	slots.Start = start
	slots.End = end
	slots.UseBlock = start != 0 && end != 0
	return slots, nil
}

func (s *llamaServerRunner) glmocrVisionTokens(ctx context.Context) (Lfm2VisionTokens, error) {
	slots := Lfm2VisionTokens{Image: 59280, Start: 59256, End: 59257}
	if id, err := s.placeholderToken(ctx, "<|image_start|>"); err == nil && id != 0 {
		slots.Start = id
	}
	if id, err := s.placeholderToken(ctx, "<|image_end|>"); err == nil && id != 0 {
		slots.End = id
	}
	if id, err := s.placeholderToken(ctx, "<|image|>"); err == nil && id != 0 {
		slots.Image = id
	}
	slots.UseBlock = slots.Start != 0 && slots.End != 0
	return slots, nil
}

func (s *llamaServerRunner) mistral3VisionTokens(ctx context.Context) (Mistral3VisionTokens, error) {
	slots := Mistral3VisionTokens{Img: 10, Break: 12, End: 13}
	if id, err := s.placeholderToken(ctx, "[IMG]"); err == nil && id != 0 {
		slots.Img = id
	}
	if id, err := s.placeholderToken(ctx, "[IMG_BREAK]"); err == nil && id != 0 {
		slots.Break = id
	}
	if id, err := s.placeholderToken(ctx, "[IMG_END]"); err == nil && id != 0 {
		slots.End = id
	}
	return slots, nil
}

func (s *llamaServerRunner) deepseekOcrVisionTokens(ctx context.Context) (DeepseekOcrVisionTokens, error) {
	slots := DeepseekOcrVisionTokens{Image: 128815}
	if id, err := s.placeholderToken(ctx, "<image>"); err == nil && id != 0 {
		slots.Image = id
	}
	return slots, nil
}

func (s *llamaServerRunner) paddedInjectDetokenizeFallback(ctx context.Context, tokens []int, media []MediaData, reason string, attrs ...any) (string, error) {
	if len(media) == 0 {
		return "", fmt.Errorf("padded inject layout mismatch")
	}
	args := append([]any{"media", len(media), "prompt_tokens", len(tokens)}, attrs...)
	slog.Warn(reason, args...)
	return s.Detokenize(ctx, tokens)
}

func (s *llamaServerRunner) completionPromptForRequest(ctx context.Context, req CompletionRequest) (prompt any, truncated bool, originalTokens int, err error) {
	media := completionMediaFromRequest(req)
	for _, img := range req.Images {
		if img.HasPrecomputedEmbedding() {
			return nil, false, 0, api.StatusError{
				StatusCode:   http.StatusBadRequest,
				ErrorMessage: "precomputed_embedding is not supported on llama-server (multimodal_data is base64 rasters only); use ggml llamarunner or ollama-engine for Qwen3-VL",
			}
		}
		if img.HasProcessorOutput() {
			return nil, false, 0, api.StatusError{
				StatusCode:   http.StatusBadRequest,
				ErrorMessage: "processor_output is not supported on llama-server (multimodal_data is base64 rasters only); use ollama-engine for Qwen3-VL",
			}
		}
	}

	if len(req.PromptTokens) > 0 {
		tokens := req.PromptTokens
		// WHY truncate with media: pretokenized multimodal layouts can exceed num_ctx;
		// skipping truncate when images were attached left oversized prompts on llama-server.
		// truncateCompletionTokens keeps vision_start…vision_end blocks intact.
		if req.Truncate && s.options.NumCtx > 1 {
			fullPromptLimit := s.options.NumCtx - 1
			if len(tokens) > fullPromptLimit {
				if !s.launch.config.ContextShift {
					return nil, false, 0, api.StatusError{
						StatusCode:   http.StatusBadRequest,
						ErrorMessage: "the prompt is longer than the context length currently available to the model; shorten the prompt, adjust the context length in settings, or use a model with a longer context length",
					}
				}
				nKeep := req.Options.NumKeep
				if nKeep < 0 {
					nKeep = len(tokens)
				}
				if s.tokenizerAddsBOS() {
					nKeep++
				}
				nKeep = min(nKeep, fullPromptLimit)
				limit := contextShiftPromptLimit(s.options.NumCtx, nKeep)
				tokens, truncated, originalTokens = truncateCompletionTokens(tokens, limit, nKeep)
				if truncated {
					slog.Warn("truncating pretokenized prompt", "limit", limit, "prompt", originalTokens, "keep", nKeep, "new", len(tokens), "media", len(media))
				}
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeQwen3VLHFRunner {
			if qwen3VLPromptHasVisionBlocks(tokens) {
				promptStr, mediaCount, err := buildLlamaServerPaddedMultimodalPrompt(ctx, s.Detokenize, tokens, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				// WHY detokenize fallback: client sent rasters but pretokenized ids lack
				// vision_start — token-only path would drop multimodal_data entirely.
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids runner_inject without vision blocks; using detokenized prompt + media markers")
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeGemma4ImgRunner {
			slots, err := s.gemma4SoftTokens(ctx)
			if err != nil {
				return nil, false, 0, err
			}
			if gemma4PromptHasSoftSlots(tokens, slots) {
				promptStr, mediaCount, err := buildLlamaServerGemma4PaddedMultimodalPrompt(ctx, s.Detokenize, tokens, slots, req.Gemma4PaddedMedia, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids gemma4 inject without multimodal soft tokens; using detokenized prompt + media markers",
					"image_slot", slots.Image, "video_slot", slots.Video, "audio_slot", slots.Audio)
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeMllamaImgRunner {
			slot := mllamaImageSlotTokenDefault
			if id, err := s.placeholderToken(ctx, "<|image|>"); err == nil && id != 0 {
				slot = id
			}
			if promptHasSlotToken(tokens, slot) {
				promptStr, mediaCount, err := buildLlamaServerSlotPaddedMultimodalPrompt(ctx, s.Detokenize, tokens, slot, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids mllama inject without image slot; using detokenized prompt + media markers",
					"image_slot", slot)
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeGemma3ImgRunner {
			slot := gemma3ImageSlotTokenDefault
			if id, err := s.placeholderToken(ctx, "<start_of_image>"); err == nil && id != 0 {
				slot = id
			}
			if promptHasSlotToken(tokens, slot) {
				promptStr, mediaCount, err := buildLlamaServerSlotPaddedMultimodalPrompt(ctx, s.Detokenize, tokens, slot, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids gemma3 inject without start_of_image slot; using detokenized prompt + media markers",
					"image_slot", slot)
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeLlama4ImgRunner {
			if llama4PromptHasVisionBlocks(tokens) {
				promptStr, mediaCount, err := buildLlamaServerLlama4PaddedMultimodalPrompt(ctx, s.Detokenize, tokens, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids llama4 inject without image blocks; using detokenized prompt + media markers")
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeLfm2ImgRunner || req.PaddedLayoutConsume == PaddedLayoutConsumeGlmocrImgRunner {
			var slots Lfm2VisionTokens
			var err error
			if req.PaddedLayoutConsume == PaddedLayoutConsumeGlmocrImgRunner {
				slots, err = s.glmocrVisionTokens(ctx)
			} else {
				slots, err = s.lfm2VisionTokens(ctx)
			}
			if err != nil {
				return nil, false, 0, err
			}
			if lfm2PromptHasVisionSlots(tokens, slots) {
				promptStr, mediaCount, err := buildLlamaServerLfm2PaddedMultimodalPrompt(ctx, s.Detokenize, tokens, slots, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids lfm2/glmocr inject without vision slots; using detokenized prompt + media markers",
					"use_block", slots.UseBlock, "image", slots.Image, "start", slots.Start, "end", slots.End)
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeMistral3ImgRunner {
			slots, err := s.mistral3VisionTokens(ctx)
			if err != nil {
				return nil, false, 0, err
			}
			if mistral3PromptHasVisionBlocks(tokens, slots) {
				promptStr, mediaCount, err := buildLlamaServerMistral3PaddedMultimodalPrompt(ctx, s.Detokenize, tokens, slots, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids mistral3 inject without IMG blocks; using detokenized prompt + media markers",
					"img", slots.Img, "end", slots.End)
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		if req.PaddedLayoutConsume == PaddedLayoutConsumeDeepseekOcrImgRunner {
			slots, err := s.deepseekOcrVisionTokens(ctx)
			if err != nil {
				return nil, false, 0, err
			}
			if deepseekOcrPromptHasImageRuns(tokens, slots.Image) {
				promptStr, mediaCount, err := buildLlamaServerDeepseekOcrPaddedMultimodalPrompt(ctx, s.Detokenize, tokens, slots.Image, s.llamaServerMediaMarker())
				if err != nil {
					return nil, false, 0, err
				}
				return llamaServerPaddedInjectPrompt{PromptString: promptStr, MediaCount: mediaCount}, truncated, originalTokens, nil
			}
			if len(media) > 0 {
				promptStr, err := s.paddedInjectDetokenizeFallback(ctx, tokens, media,
					"padded_input_ids deepseekocr inject without image runs; using detokenized prompt + media markers",
					"image", slots.Image)
				if err != nil {
					return nil, false, 0, err
				}
				return promptStr, truncated, originalTokens, nil
			}
		}

		return tokens, truncated, originalTokens, nil
	}

	promptVal := s.completionPrompt(req.Prompt, req.LeadingBOS)
	if !req.Truncate || len(media) > 0 || s.options.NumCtx <= 1 || len(promptVal) < s.options.NumCtx {
		return promptVal, false, 0, nil
	}

	tokens, err := s.tokenize(ctx, promptVal, true, nil)
	if err != nil {
		return nil, false, 0, err
	}

	// llama-server rejects prompts that fill the entire slot context, while the
	// old runner could accept exactly num_ctx prompt tokens. Keep one token of
	// headroom so token-level truncation preserves old behavior as closely as
	// llama-server allows.
	fullPromptLimit := s.options.NumCtx - 1
	if len(tokens) <= fullPromptLimit {
		return promptVal, false, 0, nil
	}

	if !s.launch.config.ContextShift {
		return nil, false, 0, api.StatusError{
			StatusCode:   http.StatusBadRequest,
			ErrorMessage: "the prompt is longer than the context length currently available to the model; shorten the prompt, adjust the context length in settings, or use a model with a longer context length",
		}
	}

	nKeep := req.Options.NumKeep
	if nKeep < 0 {
		nKeep = len(tokens)
	}
	if s.tokenizerAddsBOS() {
		nKeep++
	}
	nKeep = min(nKeep, fullPromptLimit)

	limit := contextShiftPromptLimit(s.options.NumCtx, nKeep)
	discard := len(tokens) - limit
	truncatedTokens := make([]int, 0, limit)
	truncatedTokens = append(truncatedTokens, tokens[:nKeep]...)
	truncatedTokens = append(truncatedTokens, tokens[nKeep+discard:]...)

	slog.Warn("truncating input prompt", "limit", limit, "prompt", len(tokens), "keep", nKeep, "new", len(truncatedTokens))
	return truncatedTokens, true, len(tokens), nil
}

func contextShiftPromptLimit(numCtx, numKeep int) int {
	if numCtx <= 1 {
		return 0
	}

	numKeep = max(0, min(numKeep, numCtx-1))

	// Match the old runners' first context shift: preserve num_keep, then free
	// roughly half of the remaining context before generation needs the slot.
	return numCtx - max((numCtx-numKeep)/2, 1)
}

func (s *llamaServerRunner) ContextLength() int {
	return s.options.NumCtx
}

// GrowNumCtx asks llama-server to enlarge KV in place (POST /kv/grow).
// Fail → caller reloads.
func (s *llamaServerRunner) GrowNumCtx(ctx context.Context, n int) error {
	if n <= 0 {
		return fmt.Errorf("GrowNumCtx: n_ctx must be > 0")
	}
	if n <= s.options.NumCtx {
		return nil
	}
	body, err := json.Marshal(map[string]int{"n_ctx": n})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d/kv/grow", s.port), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("kv grow: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("kv grow read: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("kv grow HTTP %d: %s", resp.StatusCode, s.statusErrorMessage(raw))
	}
	var out struct {
		OK       bool `json:"ok"`
		NCtx     int  `json:"n_ctx"`
		NCtxSeq  int  `json:"n_ctx_seq"`
		NCtxFrom int  `json:"n_ctx_from"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return fmt.Errorf("kv grow unmarshal: %w", err)
	}
	if out.NCtxSeq > 0 {
		s.options.NumCtx = out.NCtxSeq
	} else {
		s.options.NumCtx = n
	}
	slog.Info("in-place kv grow", "from", out.NCtxFrom, "n_ctx_seq", s.options.NumCtx, "n_ctx", out.NCtx)
	return nil
}

// ShrinkNumCtx packs live KV then cuts the buffer (POST /kv/shrink).
// Fails if live tokens do not fit. Not used automatically on smaller num_ctx.
func (s *llamaServerRunner) ShrinkNumCtx(ctx context.Context, n int) error {
	if n <= 0 {
		return fmt.Errorf("ShrinkNumCtx: n_ctx must be > 0")
	}
	if n >= s.options.NumCtx {
		return nil
	}
	body, err := json.Marshal(map[string]int{"n_ctx": n})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d/kv/shrink", s.port), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("kv shrink: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("kv shrink read: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("kv shrink HTTP %d: %s", resp.StatusCode, s.statusErrorMessage(raw))
	}
	var out struct {
		OK       bool `json:"ok"`
		NCtx     int  `json:"n_ctx"`
		NCtxSeq  int  `json:"n_ctx_seq"`
		NCtxFrom int  `json:"n_ctx_from"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return fmt.Errorf("kv shrink unmarshal: %w", err)
	}
	if out.NCtxSeq > 0 {
		s.options.NumCtx = out.NCtxSeq
	} else {
		s.options.NumCtx = n
	}
	slog.Info("in-place kv shrink", "from", out.NCtxFrom, "n_ctx_seq", s.options.NumCtx, "n_ctx", out.NCtx)
	return nil
}

// FindLlamaServer locates the llama-server binary in lib/ollama/.
// There is a single binary that dynamically loads GPU backends at runtime.
//
// Discovery order: LLAMA_SERVER_BIN → unified ../llama.cpp build → Flash-MoE → packaged layouts.
// WHY unified second: one elizaOS @ LLAMA_CPP_COMMIT tree; avoid stale eliza-llama.cpp siblings.
func FindLlamaServer() (string, error) {
	if override := strings.TrimSpace(os.Getenv("LLAMA_SERVER_BIN")); override != "" {
		if abs, err := filepath.Abs(override); err == nil {
			override = abs
		}
		if isUsableLlamaServerBin(override) {
			return override, nil
		}
		return "", fmt.Errorf("llama-server binary not found at LLAMA_SERVER_BIN=%s", override)
	}
	if path, ok := unifiedLlamaServerBinExists(); ok {
		return path, nil
	}
	if preferFlashMoELlamaServer() {
		if bin, err := FindFlashMoELlamaServer(); err == nil {
			return bin, nil
		}
	}
	path, candidates, err := findLlamaCppBinary("llama-server", defaultLlamaCppBinarySearch())
	if err != nil {
		return "", fmt.Errorf("llama-server binary not found (checked: %s). Run 'cmake -S llama/server --preset cpu && cmake --build --preset cpu' first", strings.Join(candidates, ", "))
	}
	return path, nil
}

// startLlamaServer spawns the upstream llama-server process with appropriate CLI flags.
func startLlamaServer(launch llamaServerLaunchConfig, out io.Writer) (cmd *exec.Cmd, port int, err error) {
	exe, err := FindLlamaServer()
	if err != nil {
		return nil, 0, err
	}

	// Allocate a port
	port = 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		var l *net.TCPListener
		if l, err = net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		slog.Debug("ResolveTCPAddr failed, using random port")
		port = rand.Intn(65535-49152) + 49152
	}

	// Build CLI flags — minimal set, let llama-server auto-detect the rest
	params := []string{
		"--model", launch.modelPath,
		"--port", strconv.Itoa(port),
		"--host", "127.0.0.1",
		"--no-webui",
		"--offline",
		"-c", strconv.Itoa(launch.opts.NumCtx * launch.numParallel),
		"-np", strconv.Itoa(launch.numParallel),
	}
	params = appendLlamaServerLogArgs(params)
	params = appendJinjaArgs(params, launch.config)

	params = appendMMProjArgs(params, launch)
	params = appendSpeculativeArgs(params, exe, launch.config, launch.opts)
	if launch.draftPMin > 0 {
		// Only emit when speculative decode is active (profile draft_p_min).
		for i := 0; i+1 < len(params); i++ {
			if params[i] == "--spec-type" && params[i+1] != "" && params[i+1] != "none" {
				params = append(params, "--spec-draft-p-min", strconv.FormatFloat(launch.draftPMin, 'f', -1, 64))
				break
			}
		}
	}

	params = append(params, qwenVLServerArgs(launch.modelArch)...)

	// LoRA adapters
	for _, adapter := range launch.adapters {
		params = append(params, "--lora", adapter)
	}

	// UseMmap
	if launch.opts.UseMMap != nil && !*launch.opts.UseMMap {
		params = append(params, "--no-mmap")
	}

	// KV cache type
	if launch.kvCacheType != "" {
		params = append(params, "--cache-type-k", launch.kvCacheType, "--cache-type-v", launch.kvCacheType)
	}

	params = appendFlashAttentionArgsForLaunch(params, launch)

	params = appendBatchArgsWithUBatch(params, launch.opts, launch.embedding, launch.numParallel, launch.ubatchSize)

	// GPU layer offloading — only pass an exact count when the caller asked for
	// a real partial/full pin. Default (-1) and Modelfile/eliza sentinel 999
	// ("all layers") omit -ngl so llama-server --fit can reduce layers to free
	// VRAM. Passing -ngl 999 pins full offload and aborts fit
	// ("n_gpu_layers already set by user"), which OOMs ~14 GiB IQ2 models on
	// 16 GiB cards (e.g. Nemotron-Super-49B virtuoso).
	if ngl, ok := llamaServerNGLArg(launch.opts.NumGPU); ok {
		params = append(params, "-ngl", ngl)
	}

	// Thread count — only pass if user explicitly set it.
	// Default behavior: let llama-server auto-detect.
	if launch.opts.NumThread > 0 {
		params = append(params, "-t", strconv.Itoa(launch.opts.NumThread))
	}

	params = appendMainGPUArgs(params, launch.opts)

	params = appendContextShiftArgs(params, launch.opts, launch.config.ContextShift)

	params = appendFlashMoEArgs(params, launch.opts)

	// Set up library paths for GPU backend discovery
	cmd = exec.Command(exe, params...)

	if out != nil {
		// os/exec serializes Write calls when stdout and stderr share a writer.
		cmd.Stdout = out
		cmd.Stderr = out
	}
	cmd.SysProcAttr = LlamaServerSysProcAttr
	ApplyParentDeath(cmd)
	SetupLlamaServerCommandEnv(cmd, exe, launch.gpuLibs, launch.extraEnvsForStart())

	slog.Info("starting llama-server", "cmd", cmd)
	slog.Debug("subprocess", "", filteredEnv(cmd.Env))

	if err = cmd.Start(); err != nil {
		return nil, 0, err
	}
	return cmd, port, nil
}

// SetupLlamaServerCommandEnv configures the environment for a llama-server
// subprocess so discovery and real model runners use the same library search
// paths and GPU backend selection.
func SetupLlamaServerCommandEnv(cmd *exec.Cmd, exe string, gpuLibs []string, extraEnvs map[string]string) {
	cmd.Env = os.Environ()

	envUpdates := make(map[string]string, len(extraEnvs)+2)
	for k, v := range extraEnvs {
		envUpdates[k] = v
	}

	libraryPaths := llamaServerLibraryPaths(exe, gpuLibs, envUpdates)
	pathEnv := llamaServerLibraryPathEnv()
	envUpdates[pathEnv] = strings.Join(libraryPaths, string(filepath.ListSeparator))

	applied := make(map[string]bool, len(envUpdates))
	for i := range cmd.Env {
		key, _, ok := strings.Cut(cmd.Env[i], "=")
		if !ok {
			continue
		}
		for updateKey, updateVal := range envUpdates {
			if strings.EqualFold(key, updateKey) {
				cmd.Env[i] = updateKey + "=" + updateVal
				applied[updateKey] = true
			}
		}
	}
	for key, val := range envUpdates {
		if !applied[key] {
			cmd.Env = append(cmd.Env, key+"="+val)
		}
	}
}

func llamaServerLibraryPathEnv() string {
	switch runtime.GOOS {
	case "windows":
		return "PATH"
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	default:
		return "LD_LIBRARY_PATH"
	}
}

func llamaServerLibraryPaths(exe string, gpuLibs []string, envUpdates map[string]string) []string {
	llamaDir := filepath.Dir(exe)
	seen := map[string]bool{}
	var libraryPaths []string
	addPath := func(path string) {
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		libraryPaths = append(libraryPaths, path)
	}

	// Library path ordering:
	// 1. llama-server's own directory — ggml-base, ggml-cpu, libllama
	// 2. GPU variant directories — cublas, cudart, backend DLL/.so
	// 3. User/system library path
	addPath(llamaDir)
	for _, dir := range gpuLibs {
		if dir == ml.LibOllamaPath || dir == llamaDir {
			continue
		}
		if envUpdates["GGML_BACKEND_PATH"] == "" {
			if backend := findLlamaServerGPUBackend(dir); backend != "" {
				envUpdates["GGML_BACKEND_PATH"] = backend
			}
		}
		addPath(dir)
	}
	if libraryPath, ok := os.LookupEnv(llamaServerLibraryPathEnv()); ok {
		for _, dir := range filepath.SplitList(libraryPath) {
			addPath(dir)
		}
	}
	return adjustPlatformLibraryPaths(libraryPaths, gpuLibs)
}

func findLlamaServerGPUBackend(dir string) string {
	patterns := []string{
		"libggml-*.so*",
		"libggml-*.dylib",
		"libggml-*.dll",
		"ggml-*.dll",
	}
	var candidates []string
	for _, pattern := range patterns {
		matches, _ := filepath.Glob(filepath.Join(dir, pattern))
		candidates = append(candidates, matches...)
	}
	slices.Sort(candidates)

	for _, match := range candidates {
		if isLlamaServerGPUBackend(match) {
			return match
		}
	}
	return ""
}

func isLlamaServerGPUBackend(path string) bool {
	name := strings.ToLower(filepath.Base(path))
	for _, prefix := range []string{
		"libggml-base",
		"ggml-base",
		"libggml-cpu",
		"ggml-cpu",
	} {
		if strings.HasPrefix(name, prefix) {
			return false
		}
	}
	return true
}

func embeddingBatchSize(opts api.Options, numParallel int) int {
	batchSize := opts.NumBatch
	if batchSize <= 0 {
		return 0
	}
	if opts.NumCtx > 0 {
		batchSize = min(batchSize, opts.NumCtx*max(numParallel, 1))
	}
	return batchSize
}

func appendLlamaServerLogArgs(params []string) []string {
	// Keep startup memory/offload lines visible for scheduler accounting.
	return append(params,
		"--log-verbosity", "4",
		"--no-log-prefix",
		"--no-log-timestamps",
	)
}

func appendBatchArgs(params []string, opts api.Options, embedding bool, numParallel int) []string {
	return appendBatchArgsWithUBatch(params, opts, embedding, numParallel, 0)
}

// appendBatchArgsWithUBatch sets -b/-ub. ubatch>0 allows L1 ubatch≠batch (e.g. 1024/256).
func appendBatchArgsWithUBatch(params []string, opts api.Options, embedding bool, numParallel, ubatch int) []string {
	if embedding {
		params = append(params, "--embedding")
		if batchSize := embeddingBatchSize(opts, numParallel); batchSize > 0 {
			params = append(params, "-b", strconv.Itoa(batchSize), "-ub", strconv.Itoa(batchSize))
		}
		return params
	}

	if opts.NumBatch > 0 {
		ub := opts.NumBatch
		if ubatch > 0 {
			ub = ubatch
		}
		params = append(params, "-b", strconv.Itoa(opts.NumBatch), "-ub", strconv.Itoa(ub))
	}
	return params
}

// LlamaServerFlashAttention resolves the flash-attention mode passed to llama-server.
func LlamaServerFlashAttention(gpus []ml.DeviceInfo) ml.FlashAttentionType {
	enabled := envconfig.FlashAttention(false)
	userSet := enabled == envconfig.FlashAttention(true)
	if userSet {
		if enabled {
			return ml.FlashAttentionEnabled
		}
		return ml.FlashAttentionDisabled
	}

	if !ml.FlashAttentionSupported(gpus) {
		return ml.FlashAttentionDisabled
	}
	return ml.FlashAttentionAuto
}

func appendFlashAttentionArgs(params []string, gpus []ml.DeviceInfo) []string {
	switch LlamaServerFlashAttention(gpus) {
	case ml.FlashAttentionEnabled:
		return append(params, "--flash-attn", "on")
	case ml.FlashAttentionDisabled:
		return append(params, "--flash-attn", "off")
	default:
		return append(params, "--flash-attn", "auto")
	}
}

func appendMainGPUArgs(params []string, opts api.Options) []string {
	if opts.MainGPU <= 0 {
		return params
	}

	return append(params, "--split-mode", "none", "--main-gpu", strconv.Itoa(opts.MainGPU))
}

// llamaServerNGLArg maps Ollama NumGPU to a llama-server -ngl value.
// ok=false means omit -ngl (auto / --fit). Matches runtime gpu_profiles:
// n_gpu_layers >= 999 → -1 (all / auto).
func llamaServerNGLArg(numGPU int) (ngl string, ok bool) {
	switch {
	case numGPU == 0:
		return "0", true
	case numGPU > 0 && numGPU < 999:
		return strconv.Itoa(numGPU), true
	default:
		// -1 default, or >=999 "all layers" Modelfile sentinel
		return "", false
	}
}

func appendMMProjArgs(params []string, launch llamaServerLaunchConfig) []string {
	if len(launch.projectors) == 0 {
		return params
	}

	// Upstream llama.cpp does not support --mmproj together with --spec-type draft-mtp
	// (CLIP load fails / generation corrupts). Prefer MTP text speed over vision for
	// these loads; vision requests should use a non-MTP tag.
	if llamaServerMTPActive(launch.config, launch.opts) {
		slog.Info("skipping mmproj with draft-mtp (unsupported together)",
			"model", launch.modelPath,
			"projector", launch.projectors[0],
			"spec_type", launch.config.SpecType,
		)
		return params
	}

	params = append(params, "--mmproj", launch.projectors[0])
	if disable, reason := launch.mmprojOffloadDisabled(); disable {
		slog.Info("disabling multimodal projector offload", "reason", reason, "model", launch.modelPath, "projector", launch.projectors[0])
		params = append(params, "--no-mmproj-offload")
	}

	return params
}

// llamaServerMTPActive reports whether launch args will enable draft-mtp.
// Requires draft_num_predict > 0 so we do not skip mmproj without actually enabling MTP.
func llamaServerMTPActive(config LlamaServerConfig, opts api.Options) bool {
	if opts.DraftNumPredict <= 0 {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(config.SpecType)) {
	case "draft-mtp", "mtp":
		return true
	case "":
		return config.EnableMTP || config.DraftModelPath != ""
	default:
		return false
	}
}

func (launch llamaServerLaunchConfig) mmprojOffloadDisabled() (bool, string) {
	if launch.forceNoMMProjOffload {
		return true, "startup-oom-retry"
	}
	// Inline multimodal GGUFs (gemma4 e2b/e4b, etc.) pass --mmproj as the same
	// model path. Offloading text + mtmd onto one 16GB card under ghost VRAM
	// SIGKILLs llama-server during bench warmups. Keep projector on CPU unless
	// free VRAM is clearly large or the operator opts in.
	if len(launch.projectors) > 0 && launch.projectors[0] == launch.modelPath {
		if !inlineMMProjGPUOffloadAllowed(launch.gpus) {
			return true, "inline-mmproj"
		}
	}
	return shouldDisableMMProjOffload(launch.opts, launch.gpus, launch.modelLayers, launch.mmprojMemory)
}

// inlineMMProjGPUOffloadAllowed reports whether an inline (same-file) mmproj may
// use GPU. Default off below 24 GiB free; override with ZEROLLAMA_INLINE_MMPROJ_OFFLOAD=1.
func inlineMMProjGPUOffloadAllowed(gpus []ml.DeviceInfo) bool {
	v := strings.TrimSpace(os.Getenv("ZEROLLAMA_INLINE_MMPROJ_OFFLOAD"))
	if strings.EqualFold(v, "1") || strings.EqualFold(v, "true") || strings.EqualFold(v, "on") {
		return true
	}
	if strings.EqualFold(v, "0") || strings.EqualFold(v, "false") || strings.EqualFold(v, "off") {
		return false
	}
	const minFree = uint64(24) << 30
	for _, gpu := range gpus {
		if gpu.FreeMemory >= minFree {
			return true
		}
	}
	return false
}

func shouldDisableMMProjOffload(opts api.Options, gpus []ml.DeviceInfo, modelLayers, mmprojMemory uint64) (bool, string) {
	if opts.NumGPU == 0 {
		return true, "cpu-only"
	}
	// Treat >=999 like -1 (all/auto); only real partial pins force projector CPU.
	if opts.NumGPU > 0 && opts.NumGPU < 999 && modelLayers > 0 && uint64(opts.NumGPU) < modelLayers {
		return true, "partial-text-offload"
	}

	requiredMemory := mmprojMemory + mmprojOffloadHeadroom

	for _, gpu := range gpus {
		memory := gpu.FreeMemory
		// FreeMemory==0 is common with ghost/driver accounting on 5080 hosts.
		// Falling back to TotalMemory wrongly enables mmproj GPU offload and
		// OOM-kills llama-server (signal: killed) once text weights fill VRAM.
		if memory == 0 {
			return true, "unknown-free-vram"
		}
		if gpu.TotalMemory > 0 && gpu.TotalMemory < memory {
			memory = gpu.TotalMemory
		}
		if memory > 0 && memory < requiredMemory {
			return true, "limited-vram"
		}
	}

	return false, ""
}

func (launch llamaServerLaunchConfig) extraEnvsForStart() map[string]string {
	pad, ok := launch.mmprojFitTargetMiB()
	if !ok {
		return launch.extraEnvs
	}

	if existing, ok := launch.extraEnvs[llamaArgFitTargetEnv]; ok {
		existingTarget, err := strconv.ParseUint(existing, 10, 64)
		if err != nil {
			slog.Warn("invalid llama-server fit target", "env", llamaArgFitTargetEnv, "value", existing, "error", err)
			return launch.extraEnvs
		}

		envs := cloneStringMap(launch.extraEnvs)
		envs[llamaArgFitTargetEnv] = strconv.FormatUint(existingTarget+pad, 10)
		return envs
	}

	if _, ok := os.LookupEnv(llamaArgFitTargetEnv); ok {
		// Preserve an inherited user override. SetupLlamaServerCommandEnv
		// will pass it through unless extraEnvs overrides it.
		return launch.extraEnvs
	}

	envs := cloneStringMap(launch.extraEnvs)
	envs[llamaArgFitTargetEnv] = strconv.FormatUint(pad, 10)
	return envs
}

func (launch llamaServerLaunchConfig) mmprojFitTargetMiB() (uint64, bool) {
	if len(launch.projectors) == 0 || launch.mmprojMemory == 0 {
		return 0, false
	}
	if disable, _ := launch.mmprojOffloadDisabled(); disable {
		return 0, false
	}

	requiredMemory := launch.mmprojMemory + mmprojOffloadHeadroom
	return (requiredMemory + bytesPerMiB - 1) / bytesPerMiB, true
}

// mmprojMemoryRequirement is a stopgap until fit accounts for mmproj memory directly.
func mmprojMemoryRequirement(modelPath string, f *ggml.GGML, projectors []string) (uint64, error) {
	if len(projectors) == 0 {
		return 0, nil
	}

	if projectors[0] == modelPath {
		if f == nil {
			return 0, errors.New("read inline mmproj metadata: missing model metadata")
		}
		var size uint64
		for _, prefix := range []string{"v.", "mm.", "a."} {
			for _, tensor := range f.Tensors().Items(prefix) {
				size += tensor.Size()
			}
		}
		if size == 0 {
			return 0, errors.New("read inline mmproj metadata: no projector tensors found")
		}
		return size, nil
	}

	file, err := os.Open(projectors[0])
	if err != nil {
		return 0, fmt.Errorf("read mmproj metadata %q: %w", projectors[0], err)
	}
	defer file.Close()

	projector, err := ggml.Decode(file, 1024)
	if err != nil {
		return 0, fmt.Errorf("read mmproj metadata %q: %w", projectors[0], err)
	}
	var size uint64
	for _, tensor := range projector.Tensors().Items() {
		size += tensor.Size()
	}
	if size == 0 {
		return 0, fmt.Errorf("read mmproj metadata %q: no projector tensors found", projectors[0])
	}
	return size, nil
}

func appendJinjaArgs(params []string, config LlamaServerConfig) []string {
	if config.DisableJinja {
		// Go-rendered chat paths send already-rendered prompts through completion
		// endpoints. Override any GGUF chat template so llama-server startup
		// does not parse an unused model template. llama-server still requires a
		// template name, so chatml is a startup-only placeholder and must not be
		// used for request routing.
		return append(params, "--no-jinja", "--chat-template", "chatml")
	}

	return params
}

func appendContextShiftArgs(params []string, opts api.Options, enabled bool) []string {
	if !enabled {
		return params
	}

	params = append(params, "--context-shift")
	if opts.NumKeep > 0 {
		params = append(params, "--keep", strconv.Itoa(opts.NumKeep))
	}

	return params
}

func isDraftMTPSpec(specType string) bool {
	switch strings.ToLower(strings.TrimSpace(specType)) {
	case "draft-mtp", "mtp":
		return true
	default:
		return false
	}
}

func appendSpeculativeArgs(params []string, serverBin string, config LlamaServerConfig, opts api.Options) []string {
	specType := strings.ToLower(strings.TrimSpace(config.SpecType))
	switch specType {
	case "none", "off", "disabled":
		return params
	case "ngram", "ngram-simple":
		sizeN := config.NgramSizeN
		if sizeN <= 0 {
			sizeN = 12
		}
		sizeM := config.NgramSizeM
		if sizeM <= 0 {
			sizeM = 48
		}
		minHits := config.NgramMinHits
		if minHits <= 0 {
			minHits = 1
		}
		return append(params,
			"--spec-type", "ngram-simple",
			"--spec-ngram-simple-size-n", strconv.Itoa(sizeN),
			"--spec-ngram-simple-size-m", strconv.Itoa(sizeM),
			"--spec-ngram-simple-min-hits", strconv.Itoa(minHits),
		)
	case "dflash", "draft-dflash":
		// CLI token differs by fork: ggml-org uses draft-dflash; eliza/c84
		// vendor builds still advertise plain "dflash".
		if config.DraftModelPath == "" {
			return params
		}
		nMax := opts.DraftNumPredict
		if nMax <= 0 {
			nMax = 4
		}
		cliType := resolveDFlashSpecType(serverBin)
		params = append(params,
			"--spec-type", cliType,
			"--spec-draft-model", config.DraftModelPath,
			"--spec-draft-n-max", strconv.Itoa(nMax),
		)
		params = appendSpecDraftBackendSamplingArg(params, serverBin)
		return appendSpecDmAdaptiveArg(params, serverBin)
	case "draft-eagle3", "eagle3":
		if config.DraftModelPath == "" {
			return params
		}
		nMax := opts.DraftNumPredict
		if nMax <= 0 {
			nMax = 4
		}
		params = append(params,
			"--spec-type", "draft-eagle3",
			"--spec-draft-model", config.DraftModelPath,
			"--spec-draft-n-max", strconv.Itoa(nMax),
		)
		return appendSpecDraftBackendSamplingArg(params, serverBin)
	case "draft-mtp", "mtp":
		return appendMTPDraftArgs(params, serverBin, config, opts)
	case "":
		if config.EnableMTP {
			return appendMTPDraftArgs(params, serverBin, config, opts)
		}
		return params
	default:
		slog.Warn("unknown llama-server spec type; ignoring", "spec_type", config.SpecType)
		return params
	}
}

func appendMTPDraftArgs(params []string, serverBin string, config LlamaServerConfig, opts api.Options) []string {
	if opts.DraftNumPredict <= 0 {
		return params
	}
	spec := strings.ToLower(strings.TrimSpace(config.SpecType))
	explicit := spec == "draft-mtp" || spec == "mtp"
	// Explicit Modelfile/request SpecType, auto-detected embedded MTP, or external draft GGUF.
	if !explicit && !config.EnableMTP && config.DraftModelPath == "" {
		return params
	}

	params = append(params, "--spec-type", "draft-mtp")
	params = append(params, "--spec-draft-n-max", strconv.Itoa(opts.DraftNumPredict))
	// WHY not appendSpecDraftBackendSamplingArg: see appendNoSpecDraftBackendSamplingArg's
	// doc comment. The ggml-org 5f55650a (b10199+1) pin's backend/GPU-side draft sampler
	// is broken for draft-mtp (near-0% acceptance, token salad); force CPU-side draft
	// sampling instead. Confirmed root cause + fix 2026-07-30 (superseded the earlier,
	// disproven --cache-ram 0 hypothesis for this same incident).
	params = appendNoSpecDraftBackendSamplingArg(params, serverBin)
	if config.DraftModelPath != "" {
		params = append(params, "--spec-draft-model", config.DraftModelPath)
	}
	return params
}

// HasMTPDraft reports whether a GGUF bundles embedded multi-token prediction weights.
func HasMTPDraft(f *ggml.GGML) bool {
	return hasMTPDraft(f)
}

func hasMTPDraft(f *ggml.GGML) bool {
	if f.KV().Uint("nextn_predict_layers") > 0 {
		return true
	}
	return hasLegacyQwenMTPDraft(f.KV().Architecture(), f.Tensors().Items("mtp."))
}

func hasLegacyQwenMTPDraft(arch string, tensors []*ggml.Tensor) bool {
	switch arch {
	case "qwen35", "qwen35moe":
		return len(tensors) > 0
	default:
		return false
	}
}

// DisableDraftMTPForArchitecture is true for Qwen3.5/3.6/3.8 hybrid Gated-Delta-Net +
// SWA GGUFs. llama.cpp draft-mtp desyncs after SWA/hybrid cache invalidation and
// emits multilingual token salad (qwen3.6 2026-07-30; qwen3.8:27b 2026-08-24).
// See llama_server_flags.go CAUTION. ngram/eagle3 are unaffected.
func DisableDraftMTPForArchitecture(arch string) bool {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "qwen35", "qwen35moe":
		return true
	default:
		return false
	}
}

// NewLlamaServerRunner creates a new llama-server runner that wraps the upstream llama-server binary.
func NewLlamaServerRunner(
	gpus []ml.DeviceInfo,
	modelPath string,
	f *ggml.GGML,
	adapters, projectors []string,
	opts api.Options,
	numParallel int,
	kvCacheType string,
	config LlamaServerConfig,
) (LlamaServer, error) {
	// Check if this is an embedding model
	arch := f.KV().Architecture()
	_, isEmbedding := f.KV()[fmt.Sprintf("%s.pooling_type", arch)]

	// Older Ollama-format GGUFs store vision tensors (v.*, mm.*) inline in
	// the main model file rather than in a separate projector layer. When
	// the arch has a llama/compat clip handler, we can point --mmproj at
	// the same file and the in-process shim translates the two views.
	//
	// If we auto-enable --mmproj for an arch whose clip handler doesn't
	// exist yet, upstream's clip loader sees un-translated Ollama tensors
	// and aborts model load. So gate on an explicit allowlist that mirrors
	// the compat layer's clip-side coverage in llama/compat/.
	compatClipArches := map[string]bool{
		"gemma3":          true,
		"gemma4":          true,
		"qwen35":          true,
		"qwen35moe":       true,
		"qwen25vl":        true,
		"qwen3vl":         true,
		"qwen3vlmoe":      true,
		"mistral3":        true,
		"deepseekocr":     true,
		"glmocr":          true,
		"llama4":          true,
		"nemotron_h_omni": true,
		// Add entries as llama/compat grows clip handlers.
	}
	if len(projectors) == 0 &&
		len(f.Tensors().Items("v.")) > 0 &&
		compatClipArches[arch] {
		projectors = []string{modelPath}
	}
	mmprojMemory, err := mmprojMemoryRequirement(modelPath, f, projectors)
	if err != nil {
		return nil, err
	}
	if config.DraftModelPath == "" && hasMTPDraft(f) {
		config.EnableMTP = true
	}
	if DisableDraftMTPForArchitecture(f.KV().Architecture()) {
		if config.EnableMTP || isDraftMTPSpec(config.SpecType) {
			slog.Warn("disabling draft-mtp for hybrid SWA/GDN architecture (cache desync / token salad)",
				"arch", f.KV().Architecture(),
				"spec_type", config.SpecType,
			)
		}
		config.EnableMTP = false
		if isDraftMTPSpec(config.SpecType) {
			config.SpecType = ""
		}
	}

	gpuLibs := ml.LibraryPaths(gpus)
	status := NewStatusWriter(os.Stderr)

	// memWriter wraps the status writer and parses buffer size lines from llama-server logs
	memWriter := &memoryParsingWriter{inner: status}

	mediaMarker := newLlamaServerMediaMarker()
	extraEnvs := ml.GetDevicesEnv(gpus)
	serverEnvs := make(map[string]string, len(extraEnvs)+2)
	for k, v := range extraEnvs {
		serverEnvs[k] = v
	}
	serverEnvs["LLAMA_MEDIA_MARKER"] = mediaMarker
	applyLlamaServerMXFP4CUDAEnv(serverEnvs, f)

	launch := llamaServerLaunchConfig{
		modelPath:    modelPath,
		modelArch:    arch,
		projectors:   slices.Clone(projectors),
		mmprojMemory: mmprojMemory,
		modelLayers:  f.KV().BlockCount() + 1,
		adapters:     slices.Clone(adapters),
		opts:         opts,
		numParallel:  numParallel,
		kvCacheType:  kvCacheType,
		embedding:    isEmbedding,
		config:       config,
		gpus:         slices.Clone(gpus),
		gpuLibs:      slices.Clone(gpuLibs),
		extraEnvs:    cloneStringMap(serverEnvs),
		ggufKV:       f.KV(),
	}
	// L1: apply runtime/configs/gpu/*.json (e.g. rtx-5080) so Phase 17 Go→llama-server
	// shares calibrated -b/-ub/KV/FA/np with the Python runtime path.
	if profile := SelectGpuProfile(gpus); profile != nil {
		ApplyGpuProfileToLaunch(&launch, profile)
		numParallel = launch.numParallel
		opts = launch.opts
		kvCacheType = launch.kvCacheType
	}

	s := &llamaServerRunner{
		client:           newLlamaServerHTTPClient(),
		status:           status,
		options:          opts,
		modelPath:        modelPath,
		mediaMarker:      mediaMarker,
		vramByDevice:     make(map[string]uint64),
		systemFreeAtLoad: make(map[string]uint64),
		gpus:             gpus,
		ggml:             f,
		totalLayers:      f.KV().BlockCount() + 1,
		rawEmbeddings:    legacyEmbeddingsWereRaw(f.KV()),
		sem:              semaphore.NewWeighted(int64(numParallel)),
		launch:           launch,
		output:           memWriter,
	}
	// Point the memory parsing writer at this runner so values are updated as logs stream in
	memWriter.runner = s

	if err := s.startProcess(); err != nil {
		msg := s.lastErrMsg()
		return nil, fmt.Errorf("error starting llama-server: %v %s", err, msg)
	}

	return s, nil
}

func cloneStringMap(src map[string]string) map[string]string {
	dst := make(map[string]string, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func legacyEmbeddingsWereRaw(kv ggml.KV) bool {
	arch := kv.Architecture()
	if _, ok := kv[fmt.Sprintf("%s.pooling_type", arch)]; !ok {
		return false
	}

	// Legacy /api/embeddings returned runner output, so preserve only old raw embed paths.
	switch arch {
	case "bert":
		if kv.String("tokenizer.ggml.model", "bert") != "bert" {
			return true
		}
		return !kv.Bool("normalize_embeddings", true)
	case "nomic-bert", "nomic-bert-moe":
		return !kv.Bool("normalize_embeddings", false)
	case "gemma3", "gemma-embedding", "qwen3":
		return false
	default:
		return false
	}
}

func (s *llamaServerRunner) startProcess() error {
	cmd, port, err := startLlamaServer(s.launch, s.output)
	if err != nil {
		return err
	}

	s.cmd = cmd
	s.port = port
	s.done = make(chan struct{})
	s.doneErr = nil
	s.loadStart = time.Now()
	s.startLoadTracking(s.loadStart)

	if exe, err := FindLlamaServer(); err == nil {
		s.serverBinPath = exe
		if st, err := os.Stat(exe); err == nil {
			s.serverBinModTime = st.ModTime()
		}
	}

	// Reap subprocess when it exits.
	go func(cmd *exec.Cmd, done chan struct{}) {
		err := cmd.Wait()
		s.doneErr = err
		if msg := s.lastErrMsg(); err != nil && msg != "" {
			slog.Error("llama-server terminated", "error", err, "exit", ExitStatusFromError(err))
			s.doneErr = errors.New(msg)
		}
		close(done)
	}(s.cmd, s.done)

	return nil
}

func qwenVLServerArgs(modelArch string) []string {
	switch modelArch {
	case "qwen2vl", "qwen25vl", "qwen3vl", "qwen3vlmoe":
		// Upstream mtmd warns that Qwen-VL needs at least 1024 image tokens for
		// correct grounding/counting behavior; the GGUF metadata default is too low.
		return []string{"--image-min-tokens", "1024"}
	default:
		return nil
	}
}

// Load waits for llama-server to finish loading the model. llama-server loads
// the model at startup and auto-detects GPU layers, so this just waits for
// health to report ready. The scheduler handles full-fit preflight for
// llama-server before this point.
func (s *llamaServerRunner) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	slog.Info("loading model via llama-server", "model", s.modelPath)

	if err := s.WaitUntilRunning(ctx); err != nil {
		retried, retryErr := s.retryWithMMProjCPUOffload(err)
		if retryErr != nil {
			return nil, retryErr
		}
		if !retried {
			return nil, err
		}
		if err := s.WaitUntilRunning(ctx); err != nil {
			return nil, fmt.Errorf("llama-server startup failed after projector CPU offload retry: %w", err)
		}
	}

	// Verify that buffer size parsing captured GPU allocations.
	// If parsing failed (e.g., llama-server log format changed), warn so the
	// issue is visible in logs when users report problems.
	if len(s.gpus) > 0 && !s.hasParsedVRAM() {
		slog.Warn("llama-server VRAM tracking: no per-device buffer sizes were parsed from "+
			"llama-server logs. VRAM accounting will be inaccurate. This may indicate a "+
			"change in llama-server's log format — check for 'buffer size' lines in the output.",
			"model", s.modelPath, "gpus", len(s.gpus))
	}

	if s.options.MainGPU >= 0 && s.options.MainGPU < len(gpus) {
		return []ml.DeviceID{gpus[s.options.MainGPU].DeviceID}, nil
	}

	// Return device IDs for all GPUs when llama-server manages layer placement itself.
	deviceIDs := make([]ml.DeviceID, len(gpus))
	for i, g := range gpus {
		deviceIDs[i] = g.DeviceID
	}

	return deviceIDs, nil
}

func (s *llamaServerRunner) retryWithMMProjCPUOffload(loadErr error) (bool, error) {
	if !s.shouldRetryMMProjCPUOffload(loadErr) {
		return false, nil
	}

	slog.Warn("llama-server startup failed with projector GPU offload; retrying with projector CPU offload", "model", s.modelPath, "error", loadErr)
	s.mmprojOffloadOOMRetried = true
	s.launch.forceNoMMProjOffload = true

	if err := s.stopProcess(); err != nil {
		return false, fmt.Errorf("llama-server startup failed before projector CPU offload retry: %w; error stopping failed process: %v", loadErr, err)
	}
	s.resetLoadAccounting()

	if err := s.startProcess(); err != nil {
		return false, fmt.Errorf("llama-server startup failed before projector CPU offload retry: %w; error starting retry: %v", loadErr, err)
	}
	return true, nil
}

func (s *llamaServerRunner) shouldRetryMMProjCPUOffload(err error) bool {
	if err == nil || s.mmprojOffloadOOMRetried || len(s.launch.projectors) == 0 {
		return false
	}
	// Kernel SIGKILL during load usually means CUDA OOM without a cudaMalloc
	// log line — treat like OOM so we retry with --no-mmproj-offload.
	if !IsOutOfMemory(err) && !isLlamaServerLikelyVRAMKill(err) {
		return false
	}
	// llama-server --fit can select a text-layer placement that fits before
	// mtmd/CLIP allocates the multimodal projector. Retry once with the
	// projector on CPU so the scheduler can keep the text model placement.
	disabled, _ := s.launch.mmprojOffloadDisabled()
	return !disabled
}

// isLlamaServerLikelyVRAMKill reports startup deaths that look like OOM kills
// (no cudaMalloc message). Used for mmproj CPU-offload retry only.
func isLlamaServerLikelyVRAMKill(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "signal: killed") ||
		strings.Contains(msg, "sys=9") ||
		strings.Contains(msg, "killed: 9")
}

func (s *llamaServerRunner) resetLoadAccounting() {
	s.memoryMu.Lock()
	defer s.memoryMu.Unlock()

	s.memTotal = 0
	s.memGPU = 0
	s.memModelFileBacked = 0
	s.memCPUMappedModel = 0
	s.gpuLayers = 0
	s.gpuLayerOverflow = 0
	for k := range s.vramByDevice {
		delete(s.vramByDevice, k)
	}
	for k := range s.systemFreeAtLoad {
		delete(s.systemFreeAtLoad, k)
	}
	if s.status != nil {
		s.status.SetLastError("")
	}
}

func (s *llamaServerRunner) hasParsedVRAM() bool {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()

	return len(s.vramByDevice) > 0
}

func (s *llamaServerRunner) startLoadTracking(t time.Time) {
	if s == nil {
		return
	}
	s.loadTracking.Store(true)
	s.noteLoadActivity(t)
}

func (s *llamaServerRunner) stopLoadTracking() {
	if s == nil {
		return
	}
	s.loadTracking.Store(false)
}

func (s *llamaServerRunner) noteLoadActivity(t time.Time) {
	if s == nil || t.IsZero() {
		return
	}
	if !s.loadTracking.Load() {
		return
	}

	ns := t.UnixNano()
	for {
		prev := s.loadActivity.Load()
		if ns <= prev {
			return
		}
		if s.loadActivity.CompareAndSwap(prev, ns) {
			return
		}
	}
}

func (s *llamaServerRunner) lastLoadActivity() time.Time {
	if s == nil {
		return time.Time{}
	}
	if ns := s.loadActivity.Load(); ns > 0 {
		return time.Unix(0, ns)
	}
	return time.Time{}
}

// getServerStatus checks llama-server's /health endpoint.
// llama-server returns {"status":"ok"}, {"status":"loading model"}, or {"status":"error"}.
func (s *llamaServerRunner) getServerStatus(ctx context.Context) (ServerStatus, error) {
	if s.cmd.ProcessState != nil {
		msg := s.lastErrMsg()
		if s.cmd.ProcessState.ExitCode() == -1 {
			slog.Warn("llama-server process no longer running", "sys", s.cmd.ProcessState.Sys(), "string", s.cmd.ProcessState)
		}
		return ServerStatusError, fmt.Errorf("llama-server process no longer running: %s %s", ExitStatus(s.cmd.ProcessState.ExitCode()), msg)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return ServerStatusError, fmt.Errorf("error creating health request: %v", err)
	}

	resp, err := s.httpClient().Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return ServerStatusNotResponding, errors.New("server not responding")
		}
		if strings.Contains(err.Error(), "connection refused") {
			return ServerStatusNotResponding, errors.New("connection refused")
		}
		return ServerStatusError, fmt.Errorf("health resp: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ServerStatusError, fmt.Errorf("read health response: %w", err)
	}

	// llama-server returns {"status":"ok"}, {"status":"loading model"}, {"status":"error", ...}
	var result struct {
		Status string `json:"status"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return ServerStatusError, fmt.Errorf("health unmarshal: %w", err)
	}

	switch result.Status {
	case "ok":
		return ServerStatusReady, nil
	case "loading model":
		return ServerStatusLoadingModel, nil
	case "no slot available":
		return ServerStatusNoSlotsAvailable, nil
	default:
		if result.Error != nil {
			switch strings.ToLower(strings.TrimSpace(result.Error.Message)) {
			case "loading model":
				return ServerStatusLoadingModel, nil
			case "no slot available":
				return ServerStatusNoSlotsAvailable, nil
			}
		}
		return ServerStatusError, fmt.Errorf("llama-server error: %s", string(body))
	}
}

func (s *llamaServerRunner) getServerStatusRetry(ctx context.Context) (ServerStatus, error) {
	var retries int
	for {
		status, err := s.getServerStatus(ctx)
		if err != nil {
			return status, err
		}
		if status == ServerStatusNoSlotsAvailable {
			if retries >= 10 {
				return status, fmt.Errorf("no slots available after %d retries", retries)
			}
			time.Sleep(5 * time.Millisecond)
			retries++
			continue
		}
		return status, nil
	}
}

func (s *llamaServerRunner) Ping(ctx context.Context) error {
	_, err := s.getServerStatus(ctx)
	if err != nil {
		slog.Debug("llama-server unhealthy", "error", err)
	}
	return err
}

func (s *llamaServerRunner) WaitUntilRunning(ctx context.Context) error {
	s.startLoadTracking(time.Now())
	defer s.stopLoadTracking()

	stallTimeout := envconfig.LoadTimeout()
	lastActivity := s.lastLoadActivity()
	if lastActivity.IsZero() {
		lastActivity = s.loadStart
	}
	if lastActivity.IsZero() {
		lastActivity = time.Now()
	}
	loadDeadline := lastActivity.Add(stallTimeout)

	slog.Info("waiting for llama-server to start responding")
	var lastStatus ServerStatus = -1

	for {
		select {
		case <-ctx.Done():
			slog.Warn("client connection closed before llama-server finished loading, aborting load")
			return fmt.Errorf("timed out waiting for llama-server to start: %w", ctx.Err())
		case <-s.done:
			if msg := s.lastErrMsg(); msg != "" {
				if s.doneErr == nil {
					return fmt.Errorf("llama-server process has terminated: %s", msg)
				}
				if s.cmd != nil && s.cmd.ProcessState != nil && s.cmd.ProcessState.ExitCode() >= 0 {
					return fmt.Errorf("llama-server process has terminated: %s: %s", ExitStatus(s.cmd.ProcessState.ExitCode()), msg)
				}
				if exit := ExitStatusFromError(s.doneErr); exit.Known() {
					return fmt.Errorf("llama-server process has terminated: %s: %s", exit, msg)
				}
				return fmt.Errorf("llama-server process has terminated: %w: %s", s.doneErr, msg)
			}
			if s.doneErr == nil {
				if s.cmd != nil && s.cmd.ProcessState != nil {
					return fmt.Errorf("llama-server process has terminated: %s", ExitStatus(s.cmd.ProcessState.ExitCode()))
				}
				return errors.New("llama-server process has terminated")
			}
			if exit := ExitStatusFromError(s.doneErr); exit.Known() {
				return fmt.Errorf("llama-server process has terminated: %s", exit)
			}
			return fmt.Errorf("llama-server process has terminated: %w", s.doneErr)
		default:
		}

		if activity := s.lastLoadActivity(); activity.After(lastActivity) {
			lastActivity = activity
			loadDeadline = lastActivity.Add(stallTimeout)
		}

		if time.Now().After(loadDeadline) {
			msg := s.lastErrMsg()
			return fmt.Errorf("timed out waiting for llama-server to start - %s", msg)
		}

		if s.cmd.ProcessState != nil {
			msg := s.lastErrMsg()
			return fmt.Errorf("llama-server process no longer running: %s %s", ExitStatus(s.cmd.ProcessState.ExitCode()), msg)
		}

		pollCtx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		status, statusErr := s.getServerStatus(pollCtx)
		cancel()

		statusChanged := lastStatus != status
		if statusChanged && status != ServerStatusReady {
			slog.Info("waiting for llama-server to become available", "status", status)
		}
		if statusChanged && status == ServerStatusLoadingModel {
			lastActivity = time.Now()
			loadDeadline = lastActivity.Add(stallTimeout)
		}

		switch status {
		case ServerStatusReady:
			if s.status != nil {
				s.status.SetLastError("")
			}
			slog.Info(fmt.Sprintf("llama-server started in %0.2f seconds", time.Since(s.loadStart).Seconds()))
			return nil
		case ServerStatusError:
			msg := s.lastErrMsg()
			if isRecoverableOutOfMemoryMessage(msg) || isRecoverableOutOfMemory(statusErr) {
				lastStatus = status
				time.Sleep(time.Millisecond * 250)
				continue
			}
			if IsOutOfMemoryMessage(msg) {
				return fmt.Errorf("llama-server reported out-of-memory during startup: %s", msg)
			}
			if IsOutOfMemory(statusErr) {
				return fmt.Errorf("llama-server reported out-of-memory during startup: %w", statusErr)
			}
			lastStatus = status
			time.Sleep(time.Millisecond * 250)
		default:
			lastStatus = status
			time.Sleep(time.Millisecond * 250)
		}
	}
}

func (s *llamaServerRunner) lastErrMsg() string {
	if s.status == nil {
		return ""
	}
	return s.status.LastError()
}

// llamaServerCompletionRequest is the request format for llama-server's POST /completion endpoint.
type llamaServerCompletionRequest struct {
	Prompt          any             `json:"prompt"`
	Stream          bool            `json:"stream"`
	CachePrompt     bool            `json:"cache_prompt"`
	NPredict        int             `json:"n_predict,omitempty"`
	NKeep           int             `json:"n_keep,omitempty"`
	Temperature     float32         `json:"temperature"`
	TopK            int             `json:"top_k"`
	TopP            float32         `json:"top_p"`
	MinP            float32         `json:"min_p"`
	Stop            []string        `json:"stop,omitempty"`
	RepeatPenalty   float32         `json:"repeat_penalty"`
	RepeatLastN     int             `json:"repeat_last_n"`
	FreqPenalty     float32         `json:"frequency_penalty"`
	PresPenalty     float32         `json:"presence_penalty"`
	TypicalP        float32         `json:"typical_p,omitempty"`
	Seed            int             `json:"seed"`
	Grammar         string          `json:"grammar,omitempty"`
	JsonSchema      json.RawMessage `json:"json_schema,omitempty"`
	NProbs          int             `json:"n_probs,omitempty"`
	IDSlot          int             `json:"id_slot,omitempty"`
	PreservedTokens []string        `json:"preserved_tokens,omitempty"`
}

func llamaServerPreservedTokens(parserTokens []string, toolCallTag string) []string {
	tokens := append([]string{}, parserTokens...)
	tokens = append(tokens, llamaServerPreservedTokensForToolTag(toolCallTag)...)
	return tokens
}

// llama-server only preserves strings that tokenize to one special token. Some
// Go templates use a parser tag like "[TOOL_CALLS][", where the first segment
// is the special token and the trailing "[" is regular JSON punctuation.
func llamaServerPreservedTokensForToolTag(tag string) []string {
	if tag == "" || tag == "{" || tag == "[" {
		return nil
	}

	if token := leadingSpecialTokenCandidate(tag); token != "" {
		return []string{token}
	}

	return []string{tag}
}

func leadingSpecialTokenCandidate(tag string) string {
	if len(tag) == 0 {
		return ""
	}

	var close byte
	switch tag[0] {
	case '[':
		close = ']'
	case '<':
		close = '>'
	default:
		return ""
	}

	end := strings.IndexByte(tag, close)
	if end <= 0 {
		return ""
	}

	return tag[:end+1]
}

// llamaServerMultimodalPrompt is used when images are present.
// llama-server's /completion endpoint accepts this as the "prompt" field.
type llamaServerMultimodalPrompt struct {
	PromptString   string   `json:"prompt_string"`
	MultimodalData []string `json:"multimodal_data"`
}

// llamaServerCompletionResponse is the response format from llama-server's /completion endpoint.
type llamaServerCompletionResponse struct {
	Content                 string                 `json:"content"`
	Stop                    bool                   `json:"stop"`
	StopType                string                 `json:"stop_type"`
	Timings                 llamaServerTimings     `json:"timings"`
	CompletionProbabilities []llamaServerTokenProb `json:"completion_probabilities"`
}

type llamaServerChatChoice struct {
	Delta struct {
		Content          string `json:"content"`
		ReasoningContent string `json:"reasoning_content"`
		ToolCalls        []struct {
			Index    int    `json:"index"`
			ID       string `json:"id"`
			Type     string `json:"type"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
	Logprobs     struct {
		Content []llamaServerTokenProb `json:"content"`
	} `json:"logprobs"`
}

type llamaServerChatResponse struct {
	Choices []llamaServerChatChoice `json:"choices"`
	Timings llamaServerTimings      `json:"timings"`
	Error   any                     `json:"error"`
}

type llamaServerTimings struct {
	CacheN    int     `json:"cache_n"`
	PromptN   int     `json:"prompt_n"`
	PromptMS  float64 `json:"prompt_ms"`
	PredictN  int     `json:"predicted_n"`
	PredictMS float64 `json:"predicted_ms"`
}

func (t llamaServerTimings) promptEvalCount() int {
	return t.CacheN + t.PromptN
}

type llamaServerApplyTemplateResponse struct {
	Prompt string `json:"prompt"`
	Error  any    `json:"error"`
}

type llamaServerTokenProb struct {
	ID          int                    `json:"id"`
	Token       string                 `json:"token"`
	Logprob     float64                `json:"logprob"`
	Prob        float64                `json:"prob"`
	TopLogprobs []llamaServerTokenProb `json:"top_logprobs"`
	TopProbs    []llamaServerTokenProb `json:"top_probs"`
}

func (s *llamaServerRunner) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	req.Media = completionMediaFromRequest(req)
	numComputed := llamaSessionPrefixTracker.estimate(s.modelPath, req.PromptCacheKey, req.PromptTokens, req.CacheReset)
	stripCoveredCompletionMedia(&req, numComputed, s.visionSpanHints(ctx, req))
	if numComputed > 0 && len(req.PromptTokens) > 0 {
		slog.Debug("llama-server strip covered mm payload",
			"num_computed", numComputed,
			"prompt_cache_key", req.PromptCacheKey,
			"media", len(req.Media),
		)
	}
	slog.Debug("llama-server completion request", "media", len(req.Media), "prompt_len", len(req.Prompt), "prompt_tokens", len(req.PromptTokens), "padded_layout_consume", req.PaddedLayoutConsume)

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		}
		return err
	}
	defer s.sem.Release(1)

	req.Options.NumPredict = boundedNumPredict(req.Options.NumPredict, s.options.NumCtx)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	prompt, promptTruncated, originalPromptTokens, err := s.completionPromptForRequest(ctx, req)
	if err != nil {
		return err
	}

	// Build the llama-server request
	lsReq := llamaServerCompletionRequest{
		Prompt:          prompt,
		Stream:          true,
		CachePrompt:     true,
		NPredict:        req.Options.NumPredict,
		NKeep:           req.Options.NumKeep,
		Temperature:     req.Options.Temperature,
		TopK:            req.Options.TopK,
		TopP:            req.Options.TopP,
		MinP:            req.Options.MinP,
		Stop:            req.Options.Stop,
		RepeatPenalty:   req.Options.RepeatPenalty,
		RepeatLastN:     req.Options.RepeatLastN,
		FreqPenalty:     req.Options.FrequencyPenalty,
		PresPenalty:     req.Options.PresencePenalty,
		TypicalP:        req.Options.TypicalP,
		Seed:            req.Options.Seed,
		PreservedTokens: llamaServerPreservedTokens(req.PreservedTokens, req.ToolCallTag),
	}

	if req.Logprobs {
		lsReq.NProbs = max(req.TopLogprobs, 1)
	}

	// Handle format: pass JSON schema directly to llama-server, or use grammar.
	// M15f: also accept {"type":"gbnf","grammar":"..."}.
	if len(req.Format) > 0 {
		switch string(req.Format) {
		case `null`, `""`:
			// not set
		case `"json"`:
			lsReq.Grammar = grammarJSON
		default:
			if req.Format[0] == '{' {
				var probe struct {
					Type    string `json:"type"`
					Grammar string `json:"grammar"`
				}
				if err := json.Unmarshal(req.Format, &probe); err == nil &&
					strings.EqualFold(strings.TrimSpace(probe.Type), "gbnf") {
					if strings.TrimSpace(probe.Grammar) == "" {
						return fmt.Errorf("gbnf format requires non-empty grammar")
					}
					lsReq.Grammar = probe.Grammar
				} else {
					lsReq.JsonSchema = req.Format
				}
			} else {
				return fmt.Errorf("invalid format: %q; expected \"json\", a JSON Schema object, or {\"type\":\"gbnf\",\"grammar\":\"...\"}", req.Format)
			}
		}
	} else if req.Grammar != "" {
		lsReq.Grammar = req.Grammar
	}

	// Convert media: padded inject uses pretokenized vision blocks; otherwise
	// replace Ollama's stable [img-N] markers with the per-process llama-server marker.
	switch p := lsReq.Prompt.(type) {
	case llamaServerPaddedInjectPrompt:
		mediaData, err := paddedInjectMediaPayloads(req.Media, p.MediaCount)
		if err != nil {
			return err
		}
		lsReq.Prompt = llamaServerMultimodalPrompt{
			PromptString:   p.PromptString,
			MultimodalData: mediaData,
		}
		slog.Info("padded_input_ids llama-server inject",
			"prompt_tokens", len(req.PromptTokens),
			"media", p.MediaCount,
			"prompt_string_len", len(p.PromptString),
		)
	case string:
		if len(req.Media) > 0 {
			promptStr := p
			var mediaData []string
			for _, media := range req.Media {
				marker := fmt.Sprintf("[img-%d]", media.ID)
				promptStr = strings.Replace(promptStr, marker, s.llamaServerMediaMarker(), 1)
				data, err := llamaServerMediaBytes(media.Data)
				if err != nil {
					return err
				}
				mediaData = append(mediaData, base64.StdEncoding.EncodeToString(data))
			}
			lsReq.Prompt = llamaServerMultimodalPrompt{
				PromptString:   promptStr,
				MultimodalData: mediaData,
			}
		}
	}

	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(lsReq); err != nil {
		return fmt.Errorf("failed to marshal completion request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating completion request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(serverReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		slog.Error("llama-server completion error", "error", err)
		if msg := s.lastErrMsg(); msg != "" {
			return fmt.Errorf("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details: %s", msg)
		}
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details")
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llama-server error response: %w", err)
		}

		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(bodyBytes)}
	}

	// Parse SSE stream from llama-server. Delay the final Done callback until
	// after the response body is closed because routes may tokenize from that
	// callback to build the final Generate context.
	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, llamaServerStreamInitialBufferSize)
	scanner.Buffer(buf, llamaServerStreamMaxBufferSize)

	var lastToken string
	var tokenRepeat int
	var finalResp CompletionResponse
	var hasFinalResp bool

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			if bytes.HasPrefix(line, []byte(":")) {
				continue
			}

			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				evt = line
			}

			var lsResp llamaServerCompletionResponse
			if err := json.Unmarshal(evt, &lsResp); err != nil {
				return fmt.Errorf("error unmarshalling llama-server response: %v", err)
			}

			// Token repeat detection
			switch {
			case strings.TrimSpace(lsResp.Content) == lastToken:
				tokenRepeat++
			default:
				lastToken = strings.TrimSpace(lsResp.Content)
				tokenRepeat = 0
			}
			if tokenRepeat > 30 {
				slog.Debug("prediction aborted, token repeat limit reached")
				return ctx.Err()
			}

			if lsResp.Content != "" && !lsResp.Stop {
				resp := CompletionResponse{
					Content: lsResp.Content,
				}
				resp.Logprobs = convertLogprobs(lsResp.CompletionProbabilities, req.TopLogprobs > 0)
				fn(resp)
			}

			if lsResp.Stop {
				doneReason := DoneReasonStop
				if lsResp.StopType == "limit" {
					doneReason = DoneReasonLength
				}

				finalResp = CompletionResponse{
					Content:               lsResp.Content,
					Done:                  true,
					DoneReason:            doneReason,
					PromptEvalCount:       lsResp.Timings.promptEvalCount(),
					PromptEvalCachedCount: lsResp.Timings.CacheN,
					PromptEvalDuration:    time.Duration(lsResp.Timings.PromptMS * float64(time.Millisecond)),
					EvalCount:             lsResp.Timings.PredictN,
					EvalDuration:          time.Duration(lsResp.Timings.PredictMS * float64(time.Millisecond)),
					PromptTruncated:       promptTruncated,
					OriginalPromptTokens:  originalPromptTokens,
				}
				hasFinalResp = true
			}
		}

		if hasFinalResp {
			break
		}
	}

	if hasFinalResp {
		llamaSessionPrefixTracker.record(s.modelPath, req.PromptCacheKey, req.PromptTokens, req.CacheReset)
		return deliverFinalLlamaServerStream(ctx, scanner, res.Body, func() { fn(finalResp) }, "response")
	}

	if err := scanner.Err(); err != nil {
		if err := llamaServerStreamLimitError("response", err); err != nil {
			return err
		}
		if isBenignLlamaServerStreamError(ctx, err) {
			return ctx.Err()
		}
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			msg := s.lastErrMsg()
			if msg == "" {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}
		return fmt.Errorf("error reading llama-server response: %v", err)
	}

	return nil
}

func deliverFinalLlamaServerStream(ctx context.Context, scanner *bufio.Scanner, body io.Closer, deliver func(), label string) error {
	for scanner.Scan() {
	}
	drainErr := scanner.Err()
	if err := body.Close(); err != nil && drainErr == nil && !isBenignLlamaServerStreamError(ctx, err) {
		return fmt.Errorf("error closing llama-server %s: %v", label, err)
	}
	deliver()
	if drainErr != nil {
		if err := llamaServerStreamLimitError(label, drainErr); err != nil {
			return err
		}
		if isBenignLlamaServerStreamError(ctx, drainErr) {
			return nil
		}
		return fmt.Errorf("error reading llama-server %s: %v", label, drainErr)
	}
	return nil
}

func isBenignLlamaServerStreamError(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "context canceled") || strings.Contains(msg, "operation was canceled")
}

func llamaServerStreamLimitError(label string, err error) error {
	if !strings.Contains(err.Error(), "token too long") {
		return nil
	}

	return fmt.Errorf("llama-server %s stream event exceeded %d MB limit", label, llamaServerStreamMaxBufferSize/(1000*1000))
}

func (s *llamaServerRunner) statusErrorMessage(body []byte) string {
	errMsg := strings.TrimSpace(string(body))
	statusMsg := s.lastErrMsg()
	if statusMsg == "" {
		return errMsg
	}

	if IsOutOfMemoryMessage(statusMsg) && !strings.Contains(strings.ToLower(errMsg), strings.ToLower(statusMsg)) {
		return strings.TrimSpace(errMsg + "\n" + statusMsg)
	}

	return errMsg
}

// convertLogprobs converts llama-server's completion_probabilities to Ollama's Logprob format.
// includeTop controls whether top alternatives are included in the output.
func convertLogprobs(probs []llamaServerTokenProb, includeTop bool) []Logprob {
	if len(probs) == 0 {
		return nil
	}
	result := make([]Logprob, len(probs))
	for i, p := range probs {
		// llama-server uses "logprob" for log-probs mode, "prob" for sampling-probs mode
		logprob := p.Logprob
		if logprob == 0 && p.Prob != 0 {
			logprob = p.Prob // Use whichever is set
		}
		result[i] = Logprob{
			TokenLogprob: TokenLogprob{
				Token:   p.Token,
				Logprob: logprob,
			},
		}

		if !includeTop {
			continue
		}

		// Convert top logprobs (could be top_logprobs or top_probs depending on mode)
		topProbs := p.TopLogprobs
		if len(topProbs) == 0 {
			topProbs = p.TopProbs
		}
		for _, tp := range topProbs {
			tl := tp.Logprob
			if tl == 0 && tp.Prob != 0 {
				tl = tp.Prob
			}
			result[i].TopLogprobs = append(result[i].TopLogprobs, TokenLogprob{
				Token:   tp.Token,
				Logprob: tl,
			})
		}
	}
	return result
}

func (s *llamaServerRunner) ApplyChatTemplate(ctx context.Context, req ChatRequest) (string, error) {
	data, err := s.llamaServerChatRequest(req, false)
	if err != nil {
		return "", err
	}

	body, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat template request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/apply-template", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("error creating chat template request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(serverReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return "", err
		}
		return "", fmt.Errorf("llama-server apply-template error: %w", err)
	}
	defer res.Body.Close()

	bodyBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return "", fmt.Errorf("failed reading llama-server template response: %w", err)
	}
	if res.StatusCode >= 400 {
		return "", api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(bodyBytes)}
	}

	var lsResp llamaServerApplyTemplateResponse
	if err := json.Unmarshal(bodyBytes, &lsResp); err != nil {
		return "", fmt.Errorf("error unmarshalling llama-server template response: %v", err)
	}
	if lsResp.Error != nil {
		return "", fmt.Errorf("llama-server template error: %v", lsResp.Error)
	}

	return lsResp.Prompt, nil
}

func (s *llamaServerRunner) Chat(ctx context.Context, req ChatRequest, fn func(ChatResponse)) error {
	slog.Debug("llama-server chat request", "messages", len(req.Messages), "tools", len(req.Tools))

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting chat request due to client closing the connection")
		}
		return err
	}
	defer s.sem.Release(1)

	req.Options.NumPredict = boundedNumPredict(req.Options.NumPredict, s.options.NumCtx)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	lsReq, err := s.llamaServerChatRequest(req, true)
	if err != nil {
		return err
	}

	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(lsReq); err != nil {
		return fmt.Errorf("failed to marshal chat request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating chat request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(serverReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		slog.Error("llama-server chat error", "error", err)
		if msg := s.lastErrMsg(); msg != "" {
			return fmt.Errorf("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details: %s", msg)
		}
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details")
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llama-server error response: %w", err)
		}

		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(bodyBytes)}
	}

	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, llamaServerStreamInitialBufferSize)
	scanner.Buffer(buf, llamaServerStreamMaxBufferSize)

	toolCalls := map[int]*llamaServerToolCallAccumulator{}
	var finalResp ChatResponse
	var hasFinalResp bool

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			if bytes.HasPrefix(line, []byte(":")) {
				continue
			}

			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				evt = line
			}
			if bytes.Equal(evt, []byte("[DONE]")) {
				continue
			}

			var lsResp llamaServerChatResponse
			if err := json.Unmarshal(evt, &lsResp); err != nil {
				return fmt.Errorf("error unmarshalling llama-server chat response: %v", err)
			}
			if lsResp.Error != nil {
				return fmt.Errorf("llama-server chat error: %v", lsResp.Error)
			}
			if len(lsResp.Choices) == 0 {
				continue
			}

			choice := lsResp.Choices[0]
			resp := ChatResponse{
				Message: api.Message{
					Role:     "assistant",
					Content:  choice.Delta.Content,
					Thinking: choice.Delta.ReasoningContent,
				},
				Logprobs: convertLogprobs(choice.Logprobs.Content, req.TopLogprobs > 0),
			}

			for _, tc := range choice.Delta.ToolCalls {
				acc := toolCalls[tc.Index]
				if acc == nil {
					acc = &llamaServerToolCallAccumulator{index: tc.Index}
					toolCalls[tc.Index] = acc
				}
				acc.id += tc.ID
				if tc.Function.Name != "" {
					acc.name += tc.Function.Name
				}
				acc.arguments += tc.Function.Arguments
			}

			if choice.FinishReason != nil {
				doneReason := DoneReasonStop
				if *choice.FinishReason == "length" {
					doneReason = DoneReasonLength
				}

				resp.Done = true
				resp.DoneReason = doneReason
				resp.PromptEvalCount = lsResp.Timings.promptEvalCount()
				resp.PromptEvalCachedCount = lsResp.Timings.CacheN
				resp.PromptEvalDuration = time.Duration(lsResp.Timings.PromptMS * float64(time.Millisecond))
				resp.EvalCount = lsResp.Timings.PredictN
				resp.EvalDuration = time.Duration(lsResp.Timings.PredictMS * float64(time.Millisecond))
				toolCalls, err := accumulatedToolCalls(toolCalls)
				if err != nil {
					return err
				}
				resp.Message.ToolCalls = toolCalls
				finalResp = resp
				hasFinalResp = true
				break
			}

			if resp.Message.Content != "" || resp.Message.Thinking != "" || len(resp.Logprobs) > 0 {
				fn(resp)
			}
		}

		if hasFinalResp {
			break
		}
	}

	if hasFinalResp {
		return deliverFinalLlamaServerStream(ctx, scanner, res.Body, func() { fn(finalResp) }, "chat response")
	}

	if err := scanner.Err(); err != nil {
		if err := llamaServerStreamLimitError("chat response", err); err != nil {
			return err
		}
		if isBenignLlamaServerStreamError(ctx, err) {
			return ctx.Err()
		}
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			msg := s.lastErrMsg()
			if msg == "" {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}
		return fmt.Errorf("error reading llama-server chat response: %v", err)
	}

	return nil
}

type llamaServerToolCallAccumulator struct {
	index     int
	id        string
	name      string
	arguments string
}

type llamaServerChatToolCall struct {
	ID       string `json:"id,omitempty"`
	Index    int    `json:"index"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func accumulatedToolCalls(accs map[int]*llamaServerToolCallAccumulator) ([]api.ToolCall, error) {
	if len(accs) == 0 {
		return nil, nil
	}

	maxIndex := 0
	for index := range accs {
		maxIndex = max(maxIndex, index)
	}

	toolCalls := make([]api.ToolCall, 0, len(accs))
	for index := 0; index <= maxIndex; index++ {
		acc := accs[index]
		if acc == nil {
			continue
		}

		var args api.ToolCallFunctionArguments
		if strings.TrimSpace(acc.arguments) != "" {
			if err := json.Unmarshal([]byte(acc.arguments), &args); err != nil {
				return nil, fmt.Errorf("llama-server returned invalid tool call arguments for %q: %w", acc.name, err)
			}
		}

		toolCalls = append(toolCalls, api.ToolCall{
			ID: acc.id,
			Function: api.ToolCallFunction{
				Index:     acc.index,
				Name:      acc.name,
				Arguments: args,
			},
		})
	}

	return toolCalls, nil
}

func (s *llamaServerRunner) llamaServerChatRequest(req ChatRequest, stream bool) (map[string]any, error) {
	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	messages := make([]map[string]any, 0, len(req.Messages))
	for _, msg := range req.Messages {
		converted, err := llamaServerChatMessage(MessageFromAPI(msg))
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}

	body := map[string]any{
		"messages":          messages,
		"stream":            stream,
		"cache_prompt":      true,
		"n_predict":         req.Options.NumPredict,
		"n_keep":            req.Options.NumKeep,
		"temperature":       req.Options.Temperature,
		"top_k":             req.Options.TopK,
		"top_p":             req.Options.TopP,
		"min_p":             req.Options.MinP,
		"stop":              req.Options.Stop,
		"repeat_penalty":    req.Options.RepeatPenalty,
		"repeat_last_n":     req.Options.RepeatLastN,
		"frequency_penalty": req.Options.FrequencyPenalty,
		"presence_penalty":  req.Options.PresencePenalty,
		"typical_p":         req.Options.TypicalP,
		"seed":              req.Options.Seed,
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}
	if req.Logprobs {
		body["logprobs"] = true
		body["top_logprobs"] = max(req.TopLogprobs, 1)
	}
	if kwargs := llamaServerChatTemplateKwargs(req.Think); kwargs != nil {
		body["chat_template_kwargs"] = kwargs
	}
	if format, err := llamaServerChatResponseFormat(req.Format); err != nil {
		return nil, err
	} else if format != nil {
		body["response_format"] = format
	}

	return body, nil
}

func llamaServerChatTemplateKwargs(think *api.ThinkValue) map[string]any {
	if think == nil {
		return nil
	}

	kwargs := map[string]any{
		"enable_thinking": think.Bool(),
	}
	if think.IsString() {
		if effort := think.String(); effort != "" {
			kwargs["reasoning_effort"] = effort
		}
	}
	return kwargs
}

func llamaServerChatMessage(msg Message) (map[string]any, error) {
	converted := map[string]any{
		"role": msg.Role,
	}
	if msg.ToolCallID != "" {
		converted["tool_call_id"] = msg.ToolCallID
	}
	if msg.ToolName != "" {
		converted["name"] = msg.ToolName
	}
	if len(msg.ToolCalls) > 0 {
		toolCalls, err := llamaServerChatToolCalls(msg.ToolCalls)
		if err != nil {
			return nil, err
		}
		converted["tool_calls"] = toolCalls
	}

	if len(msg.Media) == 0 {
		converted["content"] = msg.Content
		return converted, nil
	}

	parts := make([]map[string]any, 0, len(msg.Media)+1)
	if msg.Content != "" {
		parts = append(parts, map[string]any{
			"type": "text",
			"text": msg.Content,
		})
	}
	for _, media := range msg.Media {
		part, err := llamaServerChatMediaPart(media)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	converted["content"] = parts
	return converted, nil
}

func llamaServerChatMediaPart(media MediaData) (map[string]any, error) {
	if format, ok := AudioFormat(media.Data); ok {
		return map[string]any{
			"type": "input_audio",
			"input_audio": map[string]any{
				"data":   base64.StdEncoding.EncodeToString(media.Data),
				"format": format,
			},
		}, nil
	}

	data, err := llamaServerMediaBytes(media.Data)
	if err != nil {
		return nil, err
	}
	mime := http.DetectContentType(data)
	if !strings.HasPrefix(mime, "image/") {
		mime = "image/jpeg"
	}
	return map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data),
		},
	}, nil
}

func llamaServerMediaBytes(data []byte) ([]byte, error) {
	if http.DetectContentType(data) != "image/webp" {
		return data, nil
	}

	img, err := webp.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode WebP image: %w", err)
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, fmt.Errorf("encode WebP image as PNG: %w", err)
	}
	return buf.Bytes(), nil
}

func llamaServerChatToolCalls(tcs []api.ToolCall) ([]llamaServerChatToolCall, error) {
	toolCalls := make([]llamaServerChatToolCall, len(tcs))
	for i, tc := range tcs {
		toolCalls[i].ID = tc.ID
		toolCalls[i].Index = tc.Function.Index
		toolCalls[i].Type = "function"
		toolCalls[i].Function.Name = tc.Function.Name

		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool call arguments for %q: %w", tc.Function.Name, err)
		}
		toolCalls[i].Function.Arguments = string(args)
	}

	return toolCalls, nil
}

func llamaServerChatResponseFormat(format json.RawMessage) (map[string]any, error) {
	if len(format) == 0 {
		return nil, nil
	}

	switch string(format) {
	case `null`, `""`:
		return nil, nil
	case `"json"`:
		return map[string]any{"type": "json_object"}, nil
	default:
		if format[0] != '{' {
			return nil, fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", format)
		}

		var schema map[string]any
		if err := json.Unmarshal(format, &schema); err != nil {
			return nil, fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", format)
		}

		return map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "schema",
				"schema": schema,
			},
		}, nil
	}
}

func (s *llamaServerRunner) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return nil, 0, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return nil, 0, err
	} else if status != ServerStatusReady {
		return nil, 0, fmt.Errorf("unexpected server status: %s", status)
	}

	// Use "input" field (not "content") to get the OAI-compatible response format
	// which includes tokens_evaluated for prompt token counting
	req := map[string]any{"input": input}
	if s.rawEmbeddings {
		req["embd_normalize"] = -1
	}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, 0, fmt.Errorf("error marshaling embed data: %w", err)
	}

	// Use /v1/embeddings (OAI-compatible) to get tokens_evaluated in the response
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/embeddings", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, 0, fmt.Errorf("error creating embed request: %w", err)
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(r)
	if err != nil {
		return nil, 0, fmt.Errorf("do embedding request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, fmt.Errorf("error reading embed response: %w", err)
	}

	if resp.StatusCode >= 400 {
		statusCode, errMsg := normalizeEmbeddingError(resp.StatusCode, body)
		return nil, 0, api.StatusError{StatusCode: statusCode, ErrorMessage: errMsg}
	}

	// With "input" field, llama-server returns OAI-compatible format:
	//   {"data": [{"embedding": [0.1, ...], "tokens_evaluated": N}], "usage": {"prompt_tokens": N}}
	// With "content" field, it returns:
	//   [{"embedding": [[0.1, ...]], "index": 0}]
	var oaiResp struct {
		Data []struct {
			Embedding       json.RawMessage `json:"embedding"`
			TokensEvaluated int             `json:"tokens_evaluated"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &oaiResp); err == nil && len(oaiResp.Data) > 0 {
		var embedding []float32
		if err := json.Unmarshal(oaiResp.Data[0].Embedding, &embedding); err != nil {
			return nil, 0, fmt.Errorf("unmarshal embedding values: %w", err)
		}
		promptTokens := oaiResp.Usage.PromptTokens
		if promptTokens == 0 {
			promptTokens = oaiResp.Data[0].TokensEvaluated
		}
		return embedding, promptTokens, nil
	}

	// Fallback: non-OAI array format [{"embedding": [[0.1, ...]], "index": 0}]
	var results []struct {
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(body, &results); err != nil {
		return nil, 0, fmt.Errorf("unmarshal embedding response: %w", err)
	}
	if len(results) == 0 {
		return nil, 0, fmt.Errorf("empty embedding response")
	}

	var embedding []float32
	if err := json.Unmarshal(results[0].Embedding, &embedding); err != nil {
		var nested [][]float32
		if err2 := json.Unmarshal(results[0].Embedding, &nested); err2 != nil {
			return nil, 0, fmt.Errorf("unmarshal embedding values: %w (also tried nested: %w)", err, err2)
		}
		if len(nested) > 0 {
			embedding = nested[0]
		}
	}

	return embedding, 0, nil
}

func normalizeEmbeddingError(statusCode int, body []byte) (int, string) {
	raw := strings.TrimSpace(string(body))
	errMsg := extractLlamaServerErrorMessage(body)
	if errMsg == "" {
		errMsg = raw
	}

	if isEmbeddingInputLimitError(errMsg) || isEmbeddingInputLimitError(raw) {
		return http.StatusBadRequest, "the input length exceeds the context length"
	}

	return statusCode, errMsg
}

func extractLlamaServerErrorMessage(body []byte) string {
	var resp struct {
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(body, &resp); err != nil || len(resp.Error) == 0 {
		return ""
	}

	var msg string
	if err := json.Unmarshal(resp.Error, &msg); err == nil {
		return strings.TrimSpace(msg)
	}

	var nested struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(resp.Error, &nested); err == nil {
		return strings.TrimSpace(nested.Message)
	}

	return ""
}

func isEmbeddingInputLimitError(errMsg string) bool {
	msg := strings.ToLower(errMsg)
	return strings.Contains(msg, "too large") ||
		strings.Contains(msg, "context size") ||
		strings.Contains(msg, "context length") ||
		strings.Contains(msg, "physical batch size") ||
		strings.Contains(msg, "exceeds the available context")
}

func (s *llamaServerRunner) tokenize(ctx context.Context, content any, addSpecial bool, parseSpecial *bool) ([]int, error) {
	req := struct {
		Content      any   `json:"content"`
		AddSpecial   bool  `json:"add_special,omitempty"`
		ParseSpecial *bool `json:"parse_special,omitempty"`
	}{
		Content:      content,
		AddSpecial:   addSpecial,
		ParseSpecial: parseSpecial,
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/tokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("tokenize error: %s", body)
	}

	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return result.Tokens, nil
}

// Tokenize calls llama-server's /tokenize endpoint.
func (s *llamaServerRunner) Tokenize(ctx context.Context, content string) ([]int, error) {
	return s.tokenize(ctx, content, false, nil)
}

// Detokenize calls llama-server's /detokenize endpoint.
func (s *llamaServerRunner) Detokenize(ctx context.Context, tokens []int) (string, error) {
	data, err := json.Marshal(map[string][]int{"tokens": tokens})
	if err != nil {
		return "", err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/detokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(r)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("detokenize error: %s", body)
	}

	var result struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	return result.Content, nil
}

func (s *llamaServerRunner) Close() error {
	return s.stopProcess()
}

func (s *llamaServerRunner) stopProcess() error {
	if s.cmd != nil && s.cmd.Process != nil {
		if s.cmd.ProcessState != nil {
			return nil
		}
		slog.Debug("stopping llama-server", "pid", s.Pid())
		if err := s.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			return err
		}
		if s.done != nil {
			slog.Debug("waiting for llama-server to exit", "pid", s.Pid())
			<-s.done
		}
		slog.Debug("llama-server stopped", "pid", s.Pid())
	}
	return nil
}

// GetDeviceInfos returns device info for GPUs used by this runner, with FreeMemory
// updated to reflect actual usage. Uses the minimum of:
//   - Our accounting: TotalMemory minus tracked VRAM allocations
//   - System-reported: free VRAM from llama-server at load time minus our allocations
//
// The min-of-two approach handles both our own usage (accurate) and external
// consumers (system-reported, may be optimistic on some platforms).
func (s *llamaServerRunner) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if len(s.gpus) == 0 {
		return nil
	}
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()

	infos := make([]ml.DeviceInfo, len(s.gpus))
	for i, gpu := range s.gpus {
		infos[i] = gpu
		used := s.vramByDevice[gpu.Name]

		// Our accounting: total minus what we allocated
		var accountedFree uint64
		if used < gpu.TotalMemory {
			accountedFree = gpu.TotalMemory - used
		}

		// System-reported: what the GPU said was free at load time, minus what
		// we've allocated since. This captures external consumers on platforms
		// where the driver reports accurately.
		systemFree := accountedFree // default to our accounting
		if sysFree, ok := s.systemFreeAtLoad[gpu.Name]; ok {
			if used < sysFree {
				systemFree = sysFree - used
			} else {
				systemFree = 0
			}
		}

		// Take the minimum — never optimistic
		infos[i].FreeMemory = min(accountedFree, systemFree)
	}
	return infos
}

// MemorySize returns total and GPU memory usage parsed from llama-server's
// post-load log output. Full model-layer offload is reported as 100% GPU.
func (s *llamaServerRunner) MemorySize() (total, vram uint64) {
	s.memoryMu.RLock()
	memTotal := s.memTotal
	memGPU := s.memGPU
	memModelFileBacked := s.memModelFileBacked
	memCPUMappedModel := s.memCPUMappedModel
	totalLayers := s.totalLayers
	gpuLayers := s.gpuLayers
	gpuLayerOverflow := s.gpuLayerOverflow
	s.memoryMu.RUnlock()

	if memTotal > 0 {
		total, vram = memTotal, memGPU
		if memCPUMappedModel > 0 {
			if info, err := os.Stat(s.modelPath); err == nil && memModelFileBacked > uint64(info.Size()) {
				total -= min(memCPUMappedModel, memModelFileBacked-uint64(info.Size()))
			}
		}
		if totalLayers > 0 && gpuLayers >= totalLayers && gpuLayerOverflow == 0 {
			total = vram
		}
		return total, vram
	}
	// Fallback: use model file size as a rough proxy
	slog.Debug("llama-server buffer sizes not available, falling back to file size estimate", "model", s.modelPath)
	if info, err := os.Stat(s.modelPath); err == nil {
		total = uint64(info.Size())
		vram = total
	}
	return total, vram
}

// PredictServerVRAM estimates VRAM usage for a model without spawning llama-server.
// Uses model file size as a proxy for weights plus a rough KV cache estimate.
// This is intentionally conservative — it overestimates to avoid VRAM contention.
func PredictServerVRAM(modelPath string, f *ggml.GGML, numCtx int) uint64 {
	var weights uint64
	if info, err := os.Stat(modelPath); err == nil {
		weights = uint64(info.Size())
	}

	// KV cache: 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes (f16)
	layers := f.KV().BlockCount()
	kvHeads := f.KV().HeadCountKVMin()
	if kvHeads == 0 {
		kvHeads = 1
	}
	headDim := uint64(0)
	if f.KV().HeadCountMax() > 0 {
		headDim = f.KV().EmbeddingLength() / f.KV().HeadCountMax()
	}
	kvCache := 2 * layers * kvHeads * headDim * uint64(numCtx) * 2

	return weights + kvCache
}

// memoryParsingWriter wraps an io.Writer and parses llama-server log output
// for buffer size lines. It updates the runner's per-device VRAM tracking.
//
// Parsed line formats (all backends):
//
//	CUDA0 model buffer size =   852.89 MiB
//	CUDA0 KV buffer size =  1920.00 MiB
//	CUDA0 compute buffer size =   378.04 MiB
//	CPU_Mapped model buffer size =   308.23 MiB
//	CUDA_Host compute buffer size =   268.05 MiB
//	MTL0_Mapped model buffer size =  1918.35 MiB
//	ROCm0 model buffer size =  1918.35 MiB
type memoryParsingWriter struct {
	inner   io.Writer
	runner  *llamaServerRunner
	buffers map[memoryBufferKey]memoryBuffer
}

type memoryBufferKey struct {
	component string
	backend   string
	kind      string
}

type memoryBuffer struct {
	bytes uint64
}

// deviceFreeRegex matches per-device free VRAM reported at model load time:
//
//	using device CUDA0 (NVIDIA GeForce RTX 4060 Ti) (0000:01:00.0) - 15221 MiB free
//	using device MTL0 (Apple M5 Max) (unknown id) - 110100 MiB free
//	using device ROCm0 (AMD Radeon RX 6800) (0000:06:00.0) - 16196 MiB free
var deviceFreeRegex = regexp.MustCompile(`using device (\S+)\s+\(.*\)\s+-\s+(\d+)\s+MiB free`)

// bufferSizeRegex matches llama-server buffer size lines and captures the
// component so repeated fit/probe values can be replaced by the final load.
var bufferSizeRegex = regexp.MustCompile(`(?m)(?:^|\n)[^\n:]*?([A-Za-z_][A-Za-z0-9_]*):\s+(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB`)

var (
	offloadedLayersRegex      = regexp.MustCompile(`offloaded\s+(\d+)/(\d+)\s+layers to GPU`)
	fitOverflowingLayersRegex = regexp.MustCompile(`common_params_fit_impl:\s+-\s+.+:\s+\d+\s+layers\s+\(\s*(\d+)\s+overflowing\)`)
)

// isGPUBuffer returns true if the backend buffer name represents GPU memory.
// CPU, BLAS, and host-pinned buffers (*_Host) are not GPU memory.
// Device-mapped buffers (e.g., MTL0_Mapped) ARE GPU memory — they're model
// weights in device-accessible memory. Only CPU_Mapped is CPU memory.
func isGPUBuffer(name string) bool {
	if name == "CPU" || name == "BLAS" || strings.HasPrefix(name, "CPU_") {
		return false
	}
	if strings.HasSuffix(name, "_Host") {
		return false
	}
	return true
}

// deviceName returns the base device name for per-device VRAM tracking.
// Strips suffixes like _Mapped, _REPACK so that e.g. "MTL0_Mapped" is
// tracked under "MTL0" alongside "MTL0 KV buffer" and "MTL0 compute buffer".
func deviceName(backendName string) string {
	for _, suffix := range []string{"_Mapped", "_REPACK", "_Private"} {
		if strings.HasSuffix(backendName, suffix) {
			return strings.TrimSuffix(backendName, suffix)
		}
	}
	return backendName
}

func (w *memoryParsingWriter) Write(b []byte) (int, error) {
	if w.runner != nil {
		if len(b) > 0 && w.runner.loadTracking.Load() {
			w.runner.noteLoadActivity(time.Now())
		}

		func() {
			w.runner.memoryMu.Lock()
			defer w.runner.memoryMu.Unlock()

			if match := deviceFreeRegex.FindSubmatch(b); match != nil {
				devName := string(match[1])
				if mib, err := strconv.ParseUint(string(match[2]), 10, 64); err == nil {
					w.runner.systemFreeAtLoad[devName] = mib * 1024 * 1024
				}
			}
			for _, match := range offloadedLayersRegex.FindAllSubmatch(b, -1) {
				loaded, loadedErr := strconv.ParseUint(string(match[1]), 10, 64)
				total, totalErr := strconv.ParseUint(string(match[2]), 10, 64)
				if loadedErr == nil && totalErr == nil {
					w.runner.gpuLayers = loaded
					w.runner.totalLayers = total
				}
			}
			for _, match := range fitOverflowingLayersRegex.FindAllSubmatch(b, -1) {
				overflowing, err := strconv.ParseUint(string(match[1]), 10, 64)
				if err == nil && overflowing > 0 {
					w.runner.gpuLayerOverflow += int(overflowing)
				}
			}
			for _, match := range bufferSizeRegex.FindAllSubmatch(b, -1) {
				backendName := string(match[2])
				if mib, err := strconv.ParseFloat(string(match[4]), 64); err == nil {
					if w.buffers == nil {
						w.buffers = make(map[memoryBufferKey]memoryBuffer)
					}
					w.buffers[memoryBufferKey{
						component: string(match[1]),
						backend:   backendName,
						kind:      string(match[3]),
					}] = memoryBuffer{bytes: uint64(mib * 1024 * 1024)}
					w.updateRunnerMemoryLocked()
				}
			}
		}()
	}
	return w.inner.Write(b)
}

func (w *memoryParsingWriter) updateRunnerMemoryLocked() {
	var total, gpu, modelFileBacked, cpuMappedModel uint64
	byDevice := make(map[string]uint64)

	for key, buffer := range w.buffers {
		total += buffer.bytes
		if key.kind == "model" {
			onGPU := isGPUBuffer(key.backend)
			mmapBacked := strings.HasSuffix(key.backend, "_Mapped")
			if onGPU || mmapBacked {
				modelFileBacked += buffer.bytes
			}
			if !onGPU && mmapBacked {
				cpuMappedModel += buffer.bytes
			}
		}
		if isGPUBuffer(key.backend) {
			gpu += buffer.bytes
			byDevice[deviceName(key.backend)] += buffer.bytes
		}
	}

	w.runner.memTotal = total
	w.runner.memGPU = gpu
	w.runner.memModelFileBacked = modelFileBacked
	w.runner.memCPUMappedModel = cpuMappedModel
	w.runner.vramByDevice = byDevice
}

// VRAMByGPU returns the VRAM used by this runner on the specified device.
// The values are parsed from llama-server's buffer size log output during model load
// (model tensors + KV cache + compute buffers).
func (s *llamaServerRunner) VRAMByGPU(id ml.DeviceID) uint64 {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()

	// Map DeviceID to the log device name used by llama-server.
	// Discovery stores the device name (e.g., "CUDA0", "ROCm0", "MTL0") from
	// --list-devices stdout, which matches the buffer log prefix.
	for _, gpu := range s.gpus {
		if gpu.DeviceID == id {
			return s.vramByDevice[gpu.Name]
		}
	}
	return 0
}
