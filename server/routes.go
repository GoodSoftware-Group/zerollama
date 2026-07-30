package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/signal"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"golang.org/x/image/webp"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/agentstats"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/registry"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/server/modality/comfyui"
	"github.com/ollama/ollama/server/openapi"
	"github.com/ollama/ollama/server/vram"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tools"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
	imagegenmanifest "github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/size"
	"github.com/ollama/ollama/x/runtimeworker"
	xserver "github.com/ollama/ollama/x/server"
	"github.com/ollama/ollama/x/trainingworker"
)

const signinURLStr = "https://ollama.com/connect?name=%s&key=%s"

const (
	cloudErrRemoteInferenceUnavailable    = "remote model is unavailable"
	cloudErrRemoteModelDetailsUnavailable = "remote model details are unavailable"
	cloudErrWebSearchUnavailable          = "web search is unavailable"
	cloudErrWebFetchUnavailable           = "web fetch is unavailable"
	copilotChatUserAgentPrefix            = "GitHubCopilotChat/"
	errCloudModelsNotSupported            = "cloud models are not supported"
	errCloudUseOpenAICompat               = "use POST /v1/chat/completions or /v1/messages with a model id ending in :cloud (Eliza Cloud)"
)

func writeModelRefParseError(c *gin.Context, err error, fallbackStatus int, fallbackMessage string) {
	switch {
	case errors.Is(err, errConflictingModelSource):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, model.ErrUnqualifiedName):
		c.JSON(http.StatusBadRequest, gin.H{"error": errtypes.InvalidModelNameErrMsg})
	default:
		c.JSON(fallbackStatus, gin.H{"error": fallbackMessage})
	}
}

func shouldUseHarmony(model *Model) bool {
	if model == nil {
		return false
	}
	if !isGptOSSFamily(model.Config.ModelFamily) && !isGptOSSFamily(model.PrimaryFamily()) {
		return false
	}
	// MLX imports often ship with {{ .Prompt }} only; still route through harmony.
	if model.IsMLX() {
		return true
	}
	// Heuristic for GGUF: template must include harmony structural tags.
	if model.Template != nil && model.Template.Contains("<|start|>") && model.Template.Contains("<|end|>") {
		return true
	}
	return false
}

func experimentEnabled(name string) bool {
	return slices.Contains(strings.Split(os.Getenv("OLLAMA_EXPERIMENT"), ","), name)
}

var useClient2 = experimentEnabled("client2")

var mode string = gin.DebugMode

type Server struct {
	addr          net.Addr
	sched         *Scheduler
	defaultNumCtx int
	requestLogger *inferenceRequestLogger
	training      *trainingworker.Client
	trainingDefer *trainingDeferQueue
	runtimeEmbed  *runtimeworker.Client

	trainingVRAMMu      sync.Mutex
	trainingVRAMBlocked bool

	runtimeFifoMu     sync.RWMutex
	runtimeFifoOldest uint64

	// ggmlFreeVRAM* — TTL cache for /api/show suggest (M12). Why: show is hot-path;
	// load path calls effectiveGgmlFreeVRAMForSuggest with refresh=true instead.
	ggmlFreeVRAMMu     sync.Mutex
	ggmlFreeVRAMCached uint64
	ggmlFreeVRAMAt     time.Time

	// assignHolds — F5 short-TTL soft holds from fleet assign tokens.
	assignHolds *AssignHoldRegistry
}

func init() {
	switch mode {
	case gin.DebugMode:
	case gin.ReleaseMode:
	case gin.TestMode:
	default:
		mode = gin.DebugMode
	}

	gin.SetMode(mode)

	// Tell renderers to use [img] tags
	renderers.RenderImgTags = true
}

var (
	errRequired    = errors.New("is required")
	errBadTemplate = errors.New("template error")
)

// modelOptions merges server default → manifest parameters → request options.
// Runner fields (num_ctx, num_gpu, …) from the result are passed to llama.Load and
// pre-size KV at load time — very large manifest num_ctx can hang before first token.
// scheduleRunner caps num_ctx to the model maximum (n_ctx_train / manifest context_length),
// warns when load exceeds memory, and applies per-request VRAM policy from options
// (ggml_clamp_num_ctx, ggml_auto_kv_quant, kv_cache_type).
// show surfaces VRAM suggest via enrichShowGgmlNumCtx without clamping.
func (s *Server) modelOptions(model *Model, requestOpts map[string]any) (api.Options, error) {
	opts := api.DefaultOptions()
	if opts.NumCtx == 0 {
		opts.NumCtx = s.defaultNumCtx
	}

	// api.Options stores defaulted values, so lower layers cannot distinguish
	// an unset draft_num_predict from the default. Track that while we still
	// have the raw model/request option maps.
	draftNumPredictSet := hasOption(requestOpts, "draft_num_predict")
	if model != nil {
		draftNumPredictSet = draftNumPredictSet || hasOption(model.Options, "draft_num_predict")
		if err := opts.FromMap(model.Options); err != nil {
			return api.Options{}, err
		}
	}

	if err := opts.FromMap(requestOpts); err != nil {
		return api.Options{}, err
	}

	if model != nil && model.DraftPath == "" && !model.EmbeddedMTP && !draftNumPredictSet {
		opts.DraftNumPredict = 0
	}

	return opts, nil
}

func hasOption(opts map[string]any, name string) bool {
	_, ok := opts[name]
	return ok
}

func llamaServerConfigForModel(m *Model, contextShift bool, opts api.Options) llm.LlamaServerConfig {
	if m == nil {
		return llm.LlamaServerConfig{}
	}
	manifestDraft := m.DraftPath
	sidecarDraft := sidecarDraftModelPath(m)
	cfg := llm.LlamaServerConfig{
		DisableJinja:   usesOllamaRenderedChat(m),
		DraftModelPath: cmp.Or(manifestDraft, sidecarDraft),
		ContextShift:   contextShift,
		SpecType:       opts.SpecType,
		NgramSizeN:     opts.SpecNgramSizeN,
		NgramSizeM:     opts.SpecNgramSizeM,
		NgramMinHits:   opts.SpecNgramMinHits,
	}
	if cfg.SpecType == "" {
		switch {
		case manifestDraft != "":
			cfg.SpecType = "draft-mtp"
		case sidecarDraft != "":
			cfg.SpecType = "draft-eagle3"
		case modelUsesElizaNgramDefault(m):
			cfg.SpecType = "ngram-simple"
		}
	}
	return cfg
}

func sidecarDraftModelPath(m *Model) string {
	if m == nil || m.Options == nil {
		return ""
	}
	v, ok := m.Options["draft_model_path"]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(s)
}

func modelUsesElizaNgramDefault(m *Model) bool {
	if m == nil || !envconfig.ElizaNgramDefault() {
		return false
	}
	return strings.HasPrefix(model.ParseName(m.Name).Model, "eliza-1")
}

func usesOllamaRenderedChat(m *Model) bool {
	return m != nil && (m.Config.Renderer != "" || m.Config.Parser != "" || shouldUseHarmony(m))
}

type chatExecutionMode int

// chatExecutionMode controls whether /api/generate and /api/chat send raw messages
// to llama-server (native template) or pre-rendered prompts from Go parsers.
//
// WHY native mode on llama-server (upstream v0.30.11): upstream dropped the
// Python middleman and applies chat templates inside llama-server for plain GGUF.
// Go still renders for parser/renderer/harmony/MLX models; native mode avoids
// double templating and keeps merge parity with ollama/ollama.
const (
	chatExecutionModeNative chatExecutionMode = iota
	chatExecutionModeRendered
)

func chatModeForModel(m *Model) chatExecutionMode {
	if m.IsMLX() || usesOllamaRenderedChat(m) {
		return chatExecutionModeRendered
	}
	return chatExecutionModeNative
}

func optionsForPrompt(opts *api.Options, runner llm.LlamaServer) *api.Options {
	if opts == nil || runner == nil {
		return opts
	}
	if ctxLen := runner.ContextLength(); ctxLen > 0 && opts.NumCtx > ctxLen {
		copied := *opts
		copied.NumCtx = ctxLen
		return &copied
	}
	return opts
}

func prepareNativeChatRequest(ctx context.Context, m *Model, r llm.LlamaServer, opts *api.Options, nativeReq llm.ChatRequest, truncate bool) (llm.ChatRequest, error) {
	var err error
	nativeReq.Messages, err = truncateNativeChatMessages(ctx, m, r, optionsForPrompt(opts, r), nativeReq, truncate)
	return nativeReq, err
}

func truncateNativeChatMessages(ctx context.Context, m *Model, r llm.LlamaServer, opts *api.Options, req llm.ChatRequest, truncate bool) ([]api.Message, error) {
	if !truncate || opts == nil || opts.NumCtx <= 0 || len(req.Messages) <= 1 {
		return req.Messages, nil
	}

	lastMsgIdx := len(req.Messages) - 1
	currMsgIdx := 0
	var system []api.Message

	for i := 0; i <= lastMsgIdx; i++ {
		system = system[:0]
		for j := range i {
			if req.Messages[j].Role == "system" {
				system = append(system, req.Messages[j])
			}
		}

		renderReq := req
		renderReq.Messages = append(slices.Clone(system), req.Messages[i:]...)
		prompt, err := r.ApplyChatTemplate(ctx, renderReq)
		if err != nil {
			return nil, err
		}

		tokens, err := r.Tokenize(ctx, prompt)
		if err != nil {
			return nil, err
		}

		ctxLen := len(tokens)
		if m != nil && m.ProjectorPaths != nil {
			for _, msg := range renderReq.Messages {
				ctxLen += 768 * len(msg.Images)
			}
		}

		if ctxLen <= opts.NumCtx {
			currMsgIdx = i
			break
		}
		if i == lastMsgIdx {
			currMsgIdx = lastMsgIdx
			break
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating native chat messages which exceed context length", "truncated", currMsgIdx)
	}

	system = system[:0]
	for j := range currMsgIdx {
		if req.Messages[j].Role == "system" {
			system = append(system, req.Messages[j])
		}
	}
	return append(slices.Clone(system), req.Messages[currMsgIdx:]...), nil
}

// scheduleRunner schedules a runner after validating inputs such as capabilities and model options.
// It returns the allocated runner, model instance, consolidated options, and optional ggml num_ctx
// clamp metadata if successful and error otherwise.
func (s *Server) scheduleRunner(ctx context.Context, name string, caps []model.Capability, requestOpts map[string]any, keepAlive *api.Duration, shift *bool, statusCh chan<- any, writeStatus func(ch chan<- any, model, status, detail string, position, queueDepth int)) (llm.LlamaServer, *Model, *api.Options, *api.GgmlNumCtx, func(), error) {
	if name == "" {
		return nil, nil, nil, nil, func() {}, fmt.Errorf("model %w", errRequired)
	}

	if repaired, err := RepairLMStudioModelIfNeeded(ctx, name); err != nil {
		slog.Warn("lm studio repair before load failed", "model", name, "error", err)
	} else if repaired {
		slog.Info("lm studio repaired model before load", "model", name)
	}

	model, err := GetModel(name)
	if err != nil {
		return nil, nil, nil, nil, func() {}, err
	}

	if slices.Contains(model.Config.ModelFamilies, "mllama") && len(model.ProjectorPaths) > 0 {
		return nil, nil, nil, nil, func() {}, fmt.Errorf("'llama3.2-vision' is no longer compatible with your version of Ollama and has been replaced by a newer version. To re-download, run 'ollama pull llama3.2-vision'")
	}

	if err := model.CheckCapabilities(caps...); err != nil {
		return nil, nil, nil, nil, func() {}, fmt.Errorf("%s %w", name, err)
	}

	// Deprecated runner override option; ignore if present.
	delete(requestOpts, "use_imagegen_runner")

	opts, err := s.modelOptions(model, requestOpts)
	if err != nil {
		return nil, nil, nil, nil, func() {}, err
	}

	capMLXScheduleOptions(model, &opts)
	keepAlive = mlxKeepAliveFloor(model, keepAlive)
	keepAlive = fulfillmentKeepAliveFloor(mlxQoSFromOptions(requestOpts), keepAlive)
	ggmlCtx := capNumCtxToModelMax(model, &opts)
	if vram := s.applyGgmlNumCtxClamp(ctx, model, &opts); vram != nil {
		ggmlCtx = mergeGgmlNumCtxInfo(ggmlCtx, vram)
	}

	releaseQoS, err := s.reserveScheduleQoS(ctx, model, requestOpts)
	if err != nil {
		return nil, nil, nil, nil, func() {}, err
	}

	schedStart := time.Now()
	slog.Debug("scheduleRunner: waiting for runner",
		"model", name,
		"num_ctx", opts.NumCtx,
		"keep_alive", schedKeepAliveDesc(keepAlive),
	)
	runnerCh, errCh, ticket := s.sched.GetRunner(ctx, model, opts, keepAlive, shift)
	var runner *runnerRef
	if statusCh != nil {
		writeFn := writeStatus
		if writeFn == nil {
			writeFn = writeGenerateStatus
		}
		var err error
		runner, err = s.waitRunnerWithStatus(ctx, model.ShortName, ticket, runnerCh, errCh, statusCh, writeFn)
		if err != nil {
			releaseQoS()
			slog.Warn("scheduleRunner: failed",
				"model", name,
				"elapsed", time.Since(schedStart),
				"error", err,
			)
			return nil, nil, nil, nil, func() {}, enrichModelLoadError(name, err)
		}
	} else {
		select {
		case runner = <-runnerCh:
			slog.Info("scheduleRunner: runner acquired",
				"model", name,
				"elapsed", time.Since(schedStart),
				"pid", runner.pid,
				"loading", runner.loading,
			)
		case err = <-errCh:
			releaseQoS()
			slog.Warn("scheduleRunner: failed",
				"model", name,
				"elapsed", time.Since(schedStart),
				"error", err,
			)
			return nil, nil, nil, nil, func() {}, enrichModelLoadError(name, err)
		case <-ctx.Done():
			releaseQoS()
			slog.Warn("scheduleRunner: client canceled while waiting",
				"model", name,
				"elapsed", time.Since(schedStart),
				"err", ctx.Err(),
			)
			return nil, nil, nil, nil, func() {}, ctx.Err()
		}
	}

	// Reflect effective load-time runner options (num_ctx may be clamped to n_ctx_train).
	if runner.Options != nil {
		opts.Runner = runner.Options.Runner
	}

	return runner.llama, model, &opts, ggmlCtx, releaseQoS, nil
}

func signinURL() (string, error) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		return "", err
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))
	h, _ := os.Hostname()
	return fmt.Sprintf(signinURLStr, url.PathEscape(h), encKey), nil
}

func (s *Server) GenerateHandler(c *gin.Context) {
	checkpointStart := time.Now()

	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	// Trap 77: reject invented top-level keys on native /api/generate (parity with /api/chat).
	if err := api.CheckUnknownGenerateFields(body); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.Request.Body = io.NopCloser(bytes.NewReader(body))

	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := api.ApplyGenerateThinkingAliases(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	EnsureGeneratePromptCacheKey(&req)

	reqCtx, cancelTimeout := applyRequestTimeout(c.Request.Context(), req.Timeout)
	if cancelTimeout != nil {
		defer cancelTimeout()
	}
	c.Request = c.Request.WithContext(reqCtx)
	if req.Timeout != nil {
		c.Set("request_timeout", req.Timeout)
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}

	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": errCloudUseOpenAICompat})
		return
	}

	name := modelRef.Name

	// We cannot currently consolidate this into GetModel because all we'll
	// induce infinite recursion given the current code structure.
	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	if modelRef.Source == modelSourceLocal && m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusForbidden, gin.H{"error": errCloudModelsNotSupported})
		return
	}

	// Unload without scheduling a runner (zerollama stop, keep_alive:0 preload eviction).
	// Why not scheduleRunner: that would increment refCount and reload; we only expire.
	if req.Prompt == "" && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "",
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	// Handle image generation models
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		s.handleImageGenerate(c, req, name.String(), checkpointStart)
		return
	}

	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "raw mode does not support template, system, or context"})
		return
	}

	var builtinParser parsers.Parser
	if shouldUseHarmony(m) && m.Config.Parser == "" {
		m.Config.Parser = "harmony"
	}

	parserName := resolveParserName(m)
	if !req.Raw && parserName != "" {
		builtinParser = parsers.ParserForName(parserName)
		if builtinParser != nil {
			// no tools or last message for generate endpoint
			builtinParser.Init(nil, nil, req.Think)
		}
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if req.Suffix != "" {
		caps = append(caps, model.CapabilityInsert)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: false}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
			return
		}
	}

	if err := applyThinkingGate(&req.Think); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	streaming := req.Stream == nil || *req.Stream
	var streamCh chan any
	if streaming {
		streamCh = make(chan any, 32)
	}
	statusWriter := func(ch chan<- any, _ string, status, detail string, pos, depth int) {
		writeGenerateStatus(ch, req.Model, status, detail, pos, depth)
	}

	schedCtx := ctxWithMLXScheduleHints(c.Request.Context(), mlxScheduleHints{
		Route:  "generate",
		Stream: streaming,
	})

	r, m, opts, ggmlCtx, releaseQoS, err := s.scheduleRunner(schedCtx, name.String(), caps, req.Options, req.KeepAlive, req.Shift, streamCh, statusWriter)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support generate", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	checkpointLoaded := time.Now()
	logInferencePhase(c, "runner_ready", req.Model, checkpointStart)

	// load the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(req.Images) > 1 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "this model only supports one image while more than one image requested"})
		return
	}

	var streamKeepalive *chatStreamSession
	if streaming && !req.DebugRenderOnly {
		streamKeepalive = beginChatStream(c, streamCh, req.Model)
		defer streamKeepalive.Wait()
	}

	images := make([]llm.ImageData, len(req.Images))
	for i := range req.Images {
		images[i] = llm.ImageData{ID: i, Data: req.Images[i]}
	}

	prompt := req.Prompt
	var messagesDropped int
	var promptTokens []int
	var originalPromptTokens int
	var leadingBOS string
	var generateChainMessages []api.Message
	if !req.Raw {
		tmpl := m.Template
		if req.Template != "" {
			tmpl, err = template.Parse(req.Template)
			if err != nil {
				abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
				return
			}
		}

		var values template.Values
		if req.Suffix != "" {
			values.Prompt = prompt
			values.Suffix = req.Suffix
		} else {
			var msgs []api.Message
			if req.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: req.System})
			} else if m.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: m.System})
			}

			if req.Context == nil {
				msgs = append(msgs, m.Messages...)
			}

			userMsg := api.Message{Role: "user", Content: req.Prompt}
			for _, i := range images {
				userMsg.Images = append(userMsg.Images, i.Data)
			}
			values.Messages = append(msgs, userMsg)
		}

		values.Think = req.Think != nil && req.Think.Bool()
		values.ThinkLevel = ""
		if req.Think != nil {
			values.ThinkLevel = req.Think.String()
		}
		values.IsThinkSet = req.Think != nil

		var b bytes.Buffer
		if req.Context != nil {
			slog.Warn("the context field is deprecated and will be removed in a future version of Ollama")
			s, err := r.Detokenize(c.Request.Context(), req.Context)
			if err != nil {
				abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
				return
			}
			b.WriteString(s)
		}

		// check that we're in the `api/chat`-like flow, and if so, generate the
		// prompt the same way
		// TEMP(drifkin): we should really just detect the chat-like flow and call
		// the real chat handler, but doing this as a stopgap to get renderer
		// support for generate
		if values.Messages != nil && values.Suffix == "" && req.Template == "" {
			genTruncate := req.Truncate == nil || *req.Truncate
			if m.HasChatTemplate && chatModeForModel(m) == chatExecutionModeNative {
				nativeReq, err := prepareNativeChatRequest(c.Request.Context(), m, r, opts, llm.ChatRequest{
					Messages:    values.Messages,
					Format:      req.Format,
					Options:     opts,
					Think:       req.Think,
					Shift:       req.Shift == nil || *req.Shift,
					Logprobs:    req.Logprobs,
					TopLogprobs: req.TopLogprobs,
				}, genTruncate)
				if err != nil {
					slog.Error("chat template prompt error", "error", err)
					var serr api.StatusError
					if errors.As(err, &serr) {
						abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, serr.StatusCode, serr.ErrorMessage)
					} else {
						abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
					}
					return
				}
				nativeReq.Messages, images, err = imageTaggedMessages(m, nativeReq.Messages, 0, true)
				if err != nil {
					abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
					return
				}
				prompt, err = r.ApplyChatTemplate(c.Request.Context(), nativeReq)
				if err != nil {
					var serr api.StatusError
					if errors.As(err, &serr) {
						abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, serr.StatusCode, serr.ErrorMessage)
					} else {
						abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
					}
					return
				}
				if req.Context != nil {
					b.WriteString(prompt)
					prompt = b.String()
				}
			} else {
				tokenBudget, detok := chatPromptLimits(m, opts, genTruncate, r.ContextLength(), r.Detokenize)
				genCtx := withPromptCacheKey(c.Request.Context(), modality.ExtractPromptCacheKey(req.Options))
				prompt, images, messagesDropped, promptTokens, originalPromptTokens, err = chatPrompt(genCtx, m, r.Tokenize, opts, values.Messages, []api.Tool{}, req.Think, genTruncate, tokenBudget, detok)
				generateChainMessages = values.Messages
				if err != nil {
					abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
					return
				}
				if req.Context != nil {
					b.WriteString(prompt)
					prompt = b.String()
				}
				leadingBOS = leadingBOSForModel(m)
			}
		} else {
			// legacy flow
			if err := tmpl.Execute(&b, values); err != nil {
				abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
				return
			}

			prompt = b.String()
		}
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(images),
			},
		})
		return
	}

	checkpointPromptReady := time.Now()
	logInferencePhase(c, "prompt_ready", req.Model, checkpointLoaded)
	logLargeMLXPromptIfNeeded(m, promptTokens, opts)
	recordInferencePromptSize(c, len(promptTokens), originalPromptTokens, messagesDropped)

	var thinkingState *thinking.Parser
	if builtinParser == nil {
		openingTag, closingTag := thinking.TagsForModel(m.PrimaryFamily(), m.Template.Template)
		if req.Think != nil && req.Think.Bool() && openingTag != "" && closingTag != "" {
			thinkingState = &thinking.Parser{
				OpeningTag: openingTag,
				ClosingTag: closingTag,
			}
			if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
				thinkingState.AddContent(openingTag)
			}
		}
	}

	ch := streamCh
	if ch == nil {
		ch = make(chan any)
	}
	go func() {
		var sentDone bool
		// TODO (jmorganca): avoid building the response twice both here and below
		var sb strings.Builder
		defer func() {
			if streamKeepalive != nil {
				streamKeepalive.StopKeepalive()
			}
			if req.Stream != nil && *req.Stream && !sentDone {
				emitSyntheticGenerateFinish(ch, req.Model)
			}
			close(ch)
		}()
		firstToken := true
		var firstTokenAt time.Time
		inferCtx, cancelPreempt := s.bindInferPreemptCancel(c.Request.Context(), m, req.Options)
		if cancelPreempt != nil {
			defer cancelPreempt()
		}
		if err := r.Completion(inferCtx, llm.CompletionRequest{
			Prompt:            prompt,
			PromptTokens:      mlxCompletionPromptTokens(m, promptTokens),
			Images:            images,
			Format:            req.Format,
			Options:           opts,
			Shift:             req.Shift == nil || *req.Shift,
			Truncate:          req.Truncate == nil || *req.Truncate,
			PreservedTokens:   preservedTokensForCompletion(builtinParser),
			LeadingBOS:        leadingBOS,
			Logprobs:          req.Logprobs,
			TopLogprobs:       req.TopLogprobs,
			PromptCacheKey:    modality.ExtractPromptCacheKey(req.Options),
			CacheReset:        mlxQoSFromOptions(req.Options).CacheReset,
			SessionViTOverlay: modality.SessionViTOverlayEnabled(req.Options),
		}, func(cr llm.CompletionResponse) {
			if emitMLXPrefillGenerateStatus(ch, req.Model, cr.PrefillProcessed, cr.PrefillTotal, cr.Content, cr.Done) {
				return
			}
			if firstToken {
				firstToken = false
				firstTokenAt = time.Now()
				if streamKeepalive != nil {
					streamKeepalive.StopKeepalive()
				}
				logInferencePhase(c, "first_token", req.Model, checkpointPromptReady)
			}
			res := api.GenerateResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Response:  cr.Content,
				Done:      cr.Done,
				Metrics: api.Metrics{
					PromptEvalCount:            cr.PromptEvalCount,
					PromptEvalDuration:         cr.PromptEvalDuration,
					EvalCount:                  cr.EvalCount,
					EvalDuration:               cr.EvalDuration,
					CachedPromptTokens:         cr.PromptEvalCachedCount,
					CachedTokensHost:           cr.PromptEvalCachedHost,
					CachedTokensStorage:        cr.PromptEvalCachedStorage,
					CachedTokensStorageBackend: cr.PromptEvalCachedStorageBackend,
					CacheCreationTokens:        cr.PromptEvalCacheCreationCount,
				},
				Logprobs: toAPILogprobs(cr.Logprobs),
			}

			if builtinParser != nil {
				content, thinking, toolCalls, err := builtinParser.Add(cr.Content, cr.Done)
				if err != nil {
					enqueueGenerateStreamErrorExtra(ch, req.Model, &sentDone, err.Error(), 0,
						errorExtraFromCheckpoints(checkpointStart, checkpointLoaded, firstTokenAt, !firstTokenAt.IsZero()))
					return
				}
				res.Response = sanitizeAssistantContent(content)
				res.Thinking = thinking
				if cr.Done && len(toolCalls) > 0 {
					res.ToolCalls = toolCalls
				}
			} else if thinkingState != nil {
				thinking, content := thinkingState.AddContent(cr.Content)
				res.Thinking = thinking
				res.Response = sanitizeAssistantContent(content)
			}

			if _, err := sb.WriteString(cr.Content); err != nil {
				enqueueGenerateStreamErrorExtra(ch, req.Model, &sentDone, err.Error(), 0,
					errorExtraFromCheckpoints(checkpointStart, checkpointLoaded, firstTokenAt, !firstTokenAt.IsZero()))
			}

			if cr.Done {
				res.DoneReason = cr.DoneReason.String()
				if cr.PreemptedReason != "" {
					res.PreemptedReason = cr.PreemptedReason
				}
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
				applyGenerateTruncation(&res, cr, messagesDropped, originalPromptTokens)
				applyGgmlNumCtxResponse(&res, ggmlCtx)
				rememberMLXPromptChain(m, req.Options, prompt, generateChainMessages, r.Tokenize)
				recordInferenceCompletionDetails(c, res.DoneReason, cr.PromptEvalCount, cr.EvalCount, cr.PromptEvalCachedCount, cr.PromptEvalCachedHost, cr.PromptEvalCachedStorage, cr.PromptEvalCachedStorageBackend)
				if cr.OriginalPromptTokens > 0 {
					recordInferencePromptSize(c, cr.PromptEvalCount, cr.OriginalPromptTokens, messagesDropped)
				}
				applyEmptyGenClassifyGenerate(&res, opts.NumPredict, !checkpointLoaded.IsZero())

				if !req.Raw {
					if len(cr.Tokens) > 0 {
						// F0686: sampled ids from mlxrunner (weight-parity oracle).
						res.Context = append([]int(nil), cr.Tokens...)
					} else {
						tokens, err := r.Tokenize(c.Request.Context(), prompt+sb.String())
						if err != nil {
							enqueueGenerateStreamErrorExtra(ch, req.Model, &sentDone, err.Error(), 0,
								errorExtraFromCheckpoints(checkpointStart, checkpointLoaded, firstTokenAt, !firstTokenAt.IsZero()))
							return
						}
						res.Context = tokens
					}
				}
			}

			if builtinParser != nil {
				// only send messages with meaningful content (empty messages confuse clients)
				if res.Response != "" || res.Thinking != "" || res.Done || len(res.ToolCalls) > 0 {
					ch <- res
					if res.Done {
						sentDone = true
					}
				}

				return
			}

			ch <- res
			if res.Done {
				sentDone = true
			}
		}); err != nil {
			if isContextCanceled(err) && s.maybeEnqueueGeneratePreempted(
				ch, m, req.Options, req.Model, sb.String(), &sentDone,
				checkpointStart, checkpointLoaded, ggmlCtx,
			) {
				return
			}
			var serr api.StatusError
			extra := errorExtraFromCheckpoints(checkpointStart, checkpointLoaded, firstTokenAt, !firstTokenAt.IsZero())
			if errors.As(err, &serr) {
				enqueueGenerateStreamErrorExtra(ch, req.Model, &sentDone, serr.ErrorMessage, serr.StatusCode, extra)
			} else {
				enqueueGenerateStreamErrorExtra(ch, req.Model, &sentDone, err.Error(), 0, extra)
			}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		var r api.GenerateResponse
		var allLogprobs []api.Logprob
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.GenerateResponse:
				sbThinking.WriteString(t.Thinking)
				sbContent.WriteString(t.Response)
				r = t
				// Accumulate logprobs from all chunks for non-streaming response
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		r.Thinking = sbThinking.String()
		r.Response = sanitizeAssistantContent(sbContent.String())
		r.Logprobs = allLogprobs

		c.JSON(http.StatusOK, r)
		return
	}

	if streamKeepalive == nil {
		streamResponse(c, ch)
	}
}

func (s *Server) EmbedHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.EmbedRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}

	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
		return
	}

	var input []string

	switch i := req.Input.(type) {
	case string:
		if len(i) > 0 {
			input = append(input, i)
		}
	case []any:
		for _, v := range i {
			if _, ok := v.(string); !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
				return
			}
			input = append(input, v.(string))
		}
	default:
		if req.Input != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
			return
		}
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, m, opts, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{}})
		return
	}

	kvData, _, err := getModelData(m.ModelPath, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()

	embedWithRetry := func(text string) ([]float32, int, error) {
		emb, tokCount, err := r.Embedding(ctx, text)
		if err == nil {
			return emb, tokCount, nil
		}

		var serr api.StatusError
		if !errors.As(err, &serr) || serr.StatusCode != http.StatusBadRequest {
			return nil, 0, err
		}
		if req.Truncate != nil && !*req.Truncate {
			return nil, 0, err
		}

		tokens, err := r.Tokenize(ctx, text)
		if err != nil {
			return nil, 0, err
		}

		// TODO @nicolepardal: avoid reaching into kvData here; pass required tokenizer metadata via model/options instead
		ctxLen := min(opts.NumCtx, int(kvData.ContextLength()))
		if bos := kvData.Uint("tokenizer.ggml.bos_token_id"); len(tokens) > 0 && tokens[0] != int(bos) && kvData.Bool("add_bos_token", true) {
			ctxLen--
		}
		if eos := kvData.Uint("tokenizer.ggml.eos_token_id"); len(tokens) > 0 && tokens[len(tokens)-1] != int(eos) && kvData.Bool("add_eos_token", true) {
			ctxLen--
		}

		if len(tokens) <= ctxLen {
			return nil, 0, fmt.Errorf("input exceeds maximum context length and cannot be truncated further")
		}
		if ctxLen <= 0 {
			return nil, 0, fmt.Errorf("input after truncation exceeds maximum context length")
		}

		truncatedTokens := tokens[:ctxLen]
		truncated, err := r.Detokenize(ctx, truncatedTokens)
		if err != nil {
			return nil, 0, err
		}
		return r.Embedding(ctx, truncated)
	}

	var g errgroup.Group
	embeddings := make([][]float32, len(input))
	var totalTokens uint64
	for i, text := range input {
		g.Go(func() error {
			embedding, tokenCount, err := embedWithRetry(text)
			if err != nil {
				return err
			}
			// TODO: this first normalization should be done by the model
			embedding, err = normalize(embedding)
			if err != nil {
				return err
			}
			if req.Dimensions > 0 && req.Dimensions < len(embedding) {
				embedding, err = normalize(embedding[:req.Dimensions])
				if err != nil {
					return err
				}
			}
			embeddings[i] = embedding
			atomic.AddUint64(&totalTokens, uint64(tokenCount))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.AbortWithStatusJSON(serr.StatusCode, gin.H{
				"error": strings.TrimSpace(serr.ErrorMessage),
			})
			return
		}

		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"error": strings.TrimSpace(err.Error()),
		})
		return
	}

	resp := api.EmbedResponse{
		Model:           req.Model,
		Embeddings:      embeddings,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: int(totalTokens),
	}
	c.JSON(http.StatusOK, resp)
}

func normalize(vec []float32) ([]float32, error) {
	var sum float32
	for _, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("embedding contains NaN or Inf values")
		}
		sum += v * v
	}

	norm := float32(1.0 / max(math.Sqrt(float64(sum)), 1e-12))
	for i := range vec {
		vec[i] *= norm
	}
	return vec, nil
}

func (s *Server) EmbeddingsHandler(c *gin.Context) {
	var req api.EmbeddingRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}

	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
		return
	}

	name := modelRef.Name

	r, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	// an empty request loads the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.EmbeddingResponse{Embedding: []float64{}})
		return
	}

	embedding, _, err := r.Embedding(c.Request.Context(), req.Prompt)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}

	var e []float64
	for _, v := range embedding {
		e = append(e, float64(v))
	}

	resp := api.EmbeddingResponse{
		Embedding: e,
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) PullHandler(c *gin.Context) {
	var req api.PullRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	rawModel := strings.TrimSpace(cmp.Or(req.Model, req.Name))
	if rawModel == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	if IsHFPull(rawModel) {
		ch := make(chan any)
		go func() {
			defer close(ch)
			fn := func(r api.ProgressResponse) { ch <- r }
			ctx, cancel := context.WithCancel(c.Request.Context())
			defer cancel()
			if err := PullModel(ctx, rawModel, &registryOptions{Insecure: req.Insecure}, fn); err != nil {
				ch <- gin.H{"error": err.Error()}
			}
		}()
		if req.Stream != nil && !*req.Stream {
			waitForStream(c, ch)
			return
		}
		streamResponse(c, ch)
		return
	}

	if strings.TrimSpace(req.Source) != "" {
		localName := model.ParseName(rawModel)
		if !localName.IsValid() {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid model name %q", rawModel)})
			return
		}
		localName, err = getExistingName(localName)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		ch := make(chan any)
		go func() {
			defer close(ch)
			fn := func(r api.ProgressResponse) { ch <- r }
			ctx, cancel := context.WithCancel(c.Request.Context())
			defer cancel()
			regOpts := &registryOptions{Insecure: req.Insecure, HFSource: req.Source}
			if err := PullModel(ctx, localName.DisplayShortest(), regOpts, fn); err != nil {
				ch <- gin.H{"error": err.Error()}
			}
		}()
		if req.Stream != nil && !*req.Stream {
			waitForStream(c, ch)
			return
		}
		streamResponse(c, ch)
		return
	}

	// TEMP(drifkin): we're temporarily allowing to continue pulling cloud model
	// stub-files until we integrate cloud models into `/api/tags` (in which case
	// this roundabout way of "adding" cloud models won't be needed anymore). So
	// right here normalize any `:cloud` models into the legacy-style suffixes
	// `:<tag>-cloud` and `:cloud`
	modelRef, err := parseNormalizePullModelRef(rawModel)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, errtypes.InvalidModelNameErrMsg)
		return
	}

	name := modelRef.Name

	name, err = getExistingName(name)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		if err := PullModel(ctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

func (s *Server) PushHandler(c *gin.Context) {
	var req api.PushRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var mname string
	if req.Model != "" {
		mname = req.Model
	} else if req.Name != "" {
		mname = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		name, err := getExistingName(model.ParseName(mname))
		if err != nil {
			ch <- gin.H{"error": err.Error()}
			return
		}

		if err := PushModel(ctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

// getExistingName returns the on-disk manifest name for n (case-insensitive).
// It does not borrow tag/model parts from unrelated manifests.
func getExistingName(n model.Name) (model.Name, error) {
	var zero model.Name
	existing, err := manifest.Manifests(true)
	if err != nil {
		return zero, err
	}
	for e := range existing {
		if e.EqualFold(n) {
			return e, nil
		}
	}
	return n, nil
}

func (s *Server) DeleteHandler(c *gin.Context) {
	var r api.DeleteRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseNormalizePullModelRef(cmp.Or(r.Model, r.Name))
	if err != nil {
		switch {
		case errors.Is(err, errConflictingModelSource):
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		case errors.Is(err, model.ErrUnqualifiedName):
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("name %q is invalid", cmp.Or(r.Model, r.Name))})
		default:
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		}
		return
	}

	n, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		return
	}

	m, err := manifest.ParseNamedManifest(n)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if err := m.Remove(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if err := m.RemoveLayers(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
}

func (s *Server) ShowHandler(c *gin.Context) {
	var req api.ShowRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		logShowHandlerOutcome("", c.Request.UserAgent(), http.StatusBadRequest, err)
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		logShowHandlerOutcome("", c.Request.UserAgent(), http.StatusBadRequest, err)
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model != "" {
		// noop
	} else if req.Name != "" {
		req.Model = req.Name
	} else {
		logShowHandlerOutcome("", c.Request.UserAgent(), http.StatusBadRequest, errors.New("model is required"))
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		logShowHandlerOutcome(req.Model, c.Request.UserAgent(), http.StatusBadRequest, err)
		writeModelRefParseError(c, err, http.StatusBadRequest, err.Error())
		return
	}

	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		// Eliza GET /api/v1/models/... JSON is returned as-is (not api.ShowResponse).
		proxyCloudElizaGET(c, ElizaModelDetailPath(modelRef.Base), cloudErrRemoteModelDetailsUnavailable)
		return
	}

	req.Model = modelRef.Base
	userAgent := c.Request.UserAgent()

	resp, err := GetModelInfo(req)
	if err != nil {
		var statusErr api.StatusError
		var status int
		switch {
		case os.IsNotExist(err):
			status = http.StatusNotFound
			c.JSON(status, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case errors.As(err, &statusErr):
			status = statusErr.StatusCode
			c.JSON(status, gin.H{"error": statusErr.ErrorMessage})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			status = http.StatusBadRequest
			c.JSON(status, gin.H{"error": err.Error()})
		default:
			status = http.StatusInternalServerError
			c.JSON(status, gin.H{"error": err.Error()})
		}
		logShowHandlerOutcome(req.Model, userAgent, status, err)
		return
	}

	if modelRef.Source == modelSourceLocal && resp.RemoteHost != "" {
		logShowHandlerOutcome(modelRef.Original, userAgent, http.StatusNotFound, nil)
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", modelRef.Original)})
		return
	}

	if modelRef.Source == modelSourceLocal {
		if m, err := GetModel(modelRef.Base); err == nil {
			s.enrichShowGgmlNumCtx(c.Request.Context(), resp, m)
		} else {
			slog.Debug("api/show ggml_num_ctx enrich skipped", "model", modelRef.Base, "error", err)
		}
	}

	if strings.HasPrefix(userAgent, copilotChatUserAgentPrefix) {
		if resp.ModelInfo == nil {
			resp.ModelInfo = map[string]any{}
		}
		// Copilot Chat prefers `general.basename`, but this is usually not what
		// users are familiar with, so let's just echo back what we had returned in
		// `/api/tags`
		resp.ModelInfo["general.basename"] = req.Model
	}

	c.JSON(http.StatusOK, resp)
}

func GetModelInfo(req api.ShowRequest) (*api.ShowResponse, error) {
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return nil, showErr("parse_name", model.Unqualified(name))
	}
	name, err := getExistingName(name)
	if err != nil {
		return nil, showErr("list_manifests", err)
	}

	m, err := GetModel(name.String())
	if err != nil {
		return nil, showErr("load_model", err)
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		return nil, api.StatusError{
			StatusCode:   http.StatusNotFound,
			ErrorMessage: fmt.Sprintf("model '%s' not found", req.Model),
		}
	}

	modelDetails := api.ModelDetails{
		ParentModel:       m.ParentModel,
		Format:            m.Config.ModelFormat,
		Family:            m.PrimaryFamily(),
		Families:          m.Config.ModelFamilies,
		ParameterSize:     m.Config.ModelType,
		QuantizationLevel: m.Config.FileType,
	}

	// For image generation models, populate details from imagegen package
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		if info, err := imagegenmanifest.GetModelInfo(name.String()); err == nil {
			modelDetails.Family = info.Architecture
			modelDetails.ParameterSize = format.HumanNumber(uint64(info.ParameterCount))
			modelDetails.QuantizationLevel = info.Quantization
		}
	}

	// For safetensors LLM models (experimental), populate details from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		if info, err := xserver.GetSafetensorsLLMInfo(name); err == nil {
			if arch, ok := info["general.architecture"].(string); ok && arch != "" {
				modelDetails.Family = arch
			}
			if paramCount, ok := info["general.parameter_count"].(int64); ok && paramCount > 0 {
				modelDetails.ParameterSize = format.HumanNumber(uint64(paramCount))
			}
		}
		enrichModelDetailsFromSafetensors(&modelDetails, name)
		// Older manifests may not have file_type populated for safetensors models.
		if modelDetails.QuantizationLevel == "" {
			if dtype, err := xserver.GetSafetensorsDtype(name); err == nil && dtype != "" {
				modelDetails.QuantizationLevel = dtype
			}
		}
	}

	if req.System != "" {
		m.System = req.System
	}

	msgs := make([]api.Message, len(m.Messages))
	for i, msg := range m.Messages {
		msgs[i] = api.Message{Role: msg.Role, Content: msg.Content}
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return nil, showErr("parse_manifest", err)
	}

	resp := &api.ShowResponse{
		License:      strings.Join(m.License, "\n"),
		System:       m.System,
		Template:     m.Template.String(),
		Details:      modelDetails,
		Messages:     msgs,
		Capabilities: m.Capabilities(),
		ModifiedAt:   mf.FileInfo().ModTime(),
		Requires:     m.Config.Requires,
		// Several integrations crash on a nil/omitempty+empty ModelInfo, so by
		// default we return an empty map.
		ModelInfo: make(map[string]any),
	}

	if m.Config.RemoteHost != "" {
		resp.RemoteHost = m.Config.RemoteHost
		resp.RemoteModel = m.Config.RemoteModel

		if m.Config.ModelFamily != "" {
			resp.ModelInfo = make(map[string]any)
			resp.ModelInfo["general.architecture"] = m.Config.ModelFamily

			if m.Config.BaseName != "" {
				resp.ModelInfo["general.basename"] = m.Config.BaseName
			}

			if m.Config.ContextLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.context_length", m.Config.ModelFamily)] = m.Config.ContextLen
			}

			if m.Config.EmbedLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.embedding_length", m.Config.ModelFamily)] = m.Config.EmbedLen
			}
		}
	}

	var params []string
	cs := 30
	for k, v := range m.Options {
		switch val := v.(type) {
		case []any:
			for _, nv := range val {
				params = append(params, fmt.Sprintf("%-*s %#v", cs, k, nv))
			}
		default:
			params = append(params, fmt.Sprintf("%-*s %#v", cs, k, v))
		}
	}
	resp.Parameters = strings.Join(params, "\n")

	if len(req.Options) > 0 {
		if m.Options == nil {
			m.Options = make(map[string]any)
		}
		for k, v := range req.Options {
			m.Options[k] = v
		}
	}

	var sb strings.Builder
	fmt.Fprintln(&sb, "# Modelfile generated by \"ollama show\"")
	modelfile := m.String()
	if m.IsMLX() {
		fmt.Fprintf(&sb, "FROM %s\n", m.ShortName)
		if _, rest, ok := strings.Cut(modelfile, "\n"); ok {
			fmt.Fprint(&sb, rest)
		}
	} else {
		fmt.Fprintln(&sb, "# To build a new Modelfile based on this, replace FROM with:")
		fmt.Fprintf(&sb, "# FROM %s\n\n", m.ShortName)
		fmt.Fprint(&sb, modelfile)
	}
	resp.Modelfile = sb.String()

	// skip loading tensor information if this is a remote model
	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		return resp, nil
	}

	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		// Populate tensor info if verbose
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name); err == nil {
				resp.Tensors = tensors
			}
		}
		return resp, nil
	}

	// Config-only video_gen manifests (Wan T2V) have no GGUF path; skip getModelData.
	if slices.Contains(m.Capabilities(), model.CapabilityVideoGen) {
		return resp, nil
	}

	// Config-only speech (Piper TTS) / STT (Whisper) manifests reference external
	// ONNX/ggml weights via backend_paths — there is no GGUF model layer.
	if slices.Contains(m.Capabilities(), model.CapabilitySpeech) {
		return resp, nil
	}
	if m.ModelPath == "" {
		return resp, nil
	}

	// For safetensors LLM models (experimental), populate ModelInfo from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		info, infoErr := xserver.GetSafetensorsLLMInfo(name)
		if infoErr != nil {
			slog.Warn("api/show safetensors model_info skipped",
				"model", name.String(),
				"show_stage", "safetensors_info",
				"error", infoErr,
			)
		} else {
			resp.ModelInfo = info
		}
		// Populate tensor info if verbose
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name); err == nil {
				resp.Tensors = tensors
			} else {
				slog.Debug("api/show safetensors tensors skipped",
					"model", name.String(),
					"error", err,
				)
			}
		}
		return resp, nil
	}

	kvData, tensors, err := getModelData(m.ModelPath, req.Verbose)
	if err != nil {
		return nil, showErr("gguf_metadata", err)
	}
	enrichModelDetailsFromGGML(&resp.Details, kvData, tensors)

	delete(kvData, "general.name")
	delete(kvData, "tokenizer.chat_template")
	resp.ModelInfo = kvData

	tensorData := make([]api.Tensor, len(tensors.Items()))
	for cnt, t := range tensors.Items() {
		tensorData[cnt] = api.Tensor{Name: t.Name, Type: t.Type(), Shape: t.Shape}
	}
	resp.Tensors = tensorData

	if len(m.ProjectorPaths) > 0 {
		projectorData, _, err := getModelData(m.ProjectorPaths[0], req.Verbose)
		if err != nil {
			return nil, showErr("projector_metadata", err)
		}
		resp.ProjectorInfo = projectorData
	}

	return resp, nil
}

func getModelData(digest string, verbose bool) (ggml.KV, ggml.Tensors, error) {
	maxArraySize := 0
	if verbose {
		maxArraySize = -1
	}
	data, err := llm.LoadModel(digest, maxArraySize)
	if err != nil {
		return nil, ggml.Tensors{}, err
	}

	kv := data.KV()

	if !verbose {
		for k := range kv {
			if t, ok := kv[k].([]any); len(t) > 5 && ok {
				kv[k] = []any{}
			}
		}
	}

	return kv, data.Tensors(), nil
}

func (s *Server) ListHandler(c *gin.Context) {
	if err := SyncLMStudioModels(c.Request.Context()); err != nil {
		slog.Warn("lm studio sync before list failed", "error", err)
	}

	ms, err := manifest.Manifests(true)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	models := []api.ListModelResponse{}
	for n, m := range ms {
		var cf model.ConfigV2

		if m.Config.Digest != "" {
			f, err := m.Config.Open()
			if err != nil {
				slog.Warn("bad manifest filepath", "name", n, "error", err)
				continue
			}
			defer f.Close()

			if err := json.NewDecoder(f).Decode(&cf); err != nil {
				slog.Warn("bad manifest config", "name", n, "error", err)
				continue
			}
		}

		if cf.RemoteModel != "" {
			continue
		}

		details := api.ModelDetails{
			Format:            cf.ModelFormat,
			Family:            cf.ModelFamily,
			Families:          cf.ModelFamilies,
			ParameterSize:     cf.ModelType,
			QuantizationLevel: cf.FileType,
		}
		var capabilities []model.Capability
		if mdl, err := GetModel(n.String()); err == nil {
			enrichModelDetailsFromPath(&details, mdl.ModelPath)
			capabilities = mdl.Capabilities()
		}
		if cf.ModelFormat == "safetensors" && slices.Contains(cf.Capabilities, "completion") {
			enrichModelDetailsFromSafetensors(&details, n)
		}

		// Capabilities + enriched Details feed cmd/launch modelInventory (one /api/tags
		// load per zerollama launch run). WHY list not show: launch configures N models
		// without loading each into a runner first.
		models = append(models, api.ListModelResponse{
			Model:        n.DisplayShortest(),
			Name:         n.DisplayShortest(),
			RemoteModel:  cf.RemoteModel,
			RemoteHost:   cf.RemoteHost,
			Size:         m.Size(),
			Digest:       m.Digest(),
			ModifiedAt:   m.FileInfo().ModTime(),
			Details:      details,
			Capabilities: capabilities,
		})
	}

	slices.SortStableFunc(models, func(i, j api.ListModelResponse) int {
		// most recently modified first
		return cmp.Compare(j.ModifiedAt.Unix(), i.ModifiedAt.Unix())
	})

	models = mergeElizaCloudModels(c.Request.Context(), models)
	models = mergeLMStudioModels(models)

	// Stock ollama clients (through at least 0.31.x) do digest[:12] unconditionally
	// in `ollama ls` and panic on empty digests. Ensure every row is safe.
	for i := range models {
		if models[i].Digest == "" {
			seed := models[i].Model
			if seed == "" {
				seed = models[i].Name
			}
			models[i].Digest = listCatalogDigest("tags:" + seed)
		}
	}

	c.JSON(http.StatusOK, api.ListResponse{Models: models})
}

func (s *Server) CopyHandler(c *gin.Context) {
	var r api.CopyRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	src := model.ParseName(r.Source)
	if !src.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("source %q is invalid", r.Source)})
		return
	}
	src, err := getExistingName(src)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	dst := model.ParseName(r.Destination)
	if !dst.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("destination %q is invalid", r.Destination)})
		return
	}
	dst, err = getExistingName(dst)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := CopyModel(src, dst); errors.Is(err, os.ErrNotExist) {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found", r.Source)})
	} else if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func (s *Server) HeadBlobHandler(c *gin.Context) {
	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if _, err := os.Stat(path); err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("blob %q not found", c.Param("digest"))})
		return
	}

	c.Status(http.StatusOK)
}

func (s *Server) CreateBlobHandler(c *gin.Context) {
	if ib, ok := intermediateBlobs[c.Param("digest")]; ok {
		p, err := manifest.BlobsPath(ib)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
			slog.Info("evicting intermediate blob which no longer exists", "digest", ib)
			delete(intermediateBlobs, c.Param("digest"))
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		} else {
			c.Status(http.StatusOK)
			return
		}
	}

	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	_, err = os.Stat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop
	case err != nil:
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	default:
		c.Status(http.StatusOK)
		return
	}

	layer, err := manifest.NewLayer(c.Request.Body, "")
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if layer.Digest != c.Param("digest") {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("digest mismatch, expected %q, got %q", c.Param("digest"), layer.Digest)})
		return
	}

	c.Status(http.StatusCreated)
}

func isLocalIP(ip netip.Addr) bool {
	if interfaces, err := net.Interfaces(); err == nil {
		for _, iface := range interfaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}

			for _, a := range addrs {
				if parsed, _, err := net.ParseCIDR(a.String()); err == nil {
					if parsed.String() == ip.String() {
						return true
					}
				}
			}
		}
	}

	return false
}

func allowedHost(host string) bool {
	host = strings.ToLower(host)

	if host == "" || host == "localhost" {
		return true
	}

	if hostname, err := os.Hostname(); err == nil && host == strings.ToLower(hostname) {
		return true
	}

	tlds := []string{
		"localhost",
		"local",
		"internal",
	}

	// check if the host is a local TLD
	for _, tld := range tlds {
		if strings.HasSuffix(host, "."+tld) {
			return true
		}
	}

	return false
}

func allowedHostsMiddleware(addr net.Addr) gin.HandlerFunc {
	return func(c *gin.Context) {
		if addr == nil {
			c.Next()
			return
		}

		if addr, err := netip.ParseAddrPort(addr.String()); err == nil && !addr.Addr().IsLoopback() {
			c.Next()
			return
		}

		host, _, err := net.SplitHostPort(c.Request.Host)
		if err != nil {
			host = c.Request.Host
		}

		if addr, err := netip.ParseAddr(host); err == nil {
			if addr.IsLoopback() || addr.IsPrivate() || addr.IsUnspecified() || isLocalIP(addr) {
				c.Next()
				return
			}
		}

		if allowedHost(host) {
			if c.Request.Method == http.MethodOptions {
				c.AbortWithStatus(http.StatusNoContent)
				return
			}

			c.Next()
			return
		}

		c.AbortWithStatus(http.StatusForbidden)
	}
}

// maybeProxyElizaV1ModelGet proxies GET /v1/models/:model to Eliza (raw upstream JSON, not OpenAI ToModel) for :cloud names.
func (s *Server) maybeProxyElizaV1ModelGet() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodGet {
			c.Next()
			return
		}
		modelName := strings.TrimSpace(c.Param("model"))
		if modelName == "" {
			c.Next()
			return
		}
		modelRef, err := parseAndValidateModelRef(modelName)
		if err != nil || modelRef.Source != modelSourceCloud {
			c.Next()
			return
		}
		proxyCloudElizaGET(c, ElizaModelDetailPath(modelRef.Base), cloudErrRemoteModelDetailsUnavailable)
		c.Abort()
	}
}

func (s *Server) GenerateRoutes(rc *ollama.Registry) (http.Handler, error) {
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowWildcard = true
	corsConfig.AllowBrowserExtensions = true
	corsConfig.AllowHeaders = []string{
		"Authorization",
		"Content-Type",
		"User-Agent",
		"Accept",
		"X-Requested-With",

		// OpenAI compatibility headers
		"OpenAI-Beta",
		"x-stainless-arch",
		"x-stainless-async",
		"x-stainless-custom-poll-interval",
		"x-stainless-helper-method",
		"x-stainless-lang",
		"x-stainless-os",
		"x-stainless-package-version",
		"x-stainless-poll-helper",
		"x-stainless-retry-count",
		"x-stainless-runtime",
		"x-stainless-runtime-version",
		"x-stainless-timeout",
	}
	corsConfig.AllowOrigins = envconfig.AllowedOrigins()

	r := gin.Default()
	r.HandleMethodNotAllowed = true
	r.Use(
		cors.New(corsConfig),
		allowedHostsMiddleware(s.addr),
	)

	// General
	r.HEAD("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
	r.GET("/", func(c *gin.Context) {
		accept := c.GetHeader("Accept")
		if strings.Contains(accept, "text/html") && !strings.Contains(accept, "text/plain") {
			c.Redirect(http.StatusFound, "/docs")
			return
		}
		c.String(http.StatusOK, "Ollama is running\n%s\n", openapi.SpecSummary())
	})
	openapi.Register(r)
	r.HEAD("/api/version", VersionHandler)
	r.GET("/api/version", VersionHandler)
	r.GET("/api/status", s.StatusHandler)
	r.POST("/api/can-load", s.CanLoadHandler)
	r.POST("/api/propose-load", s.ProposeLoadHandler)
	r.POST("/api/pin", s.PinHandler)
	r.DELETE("/api/pin/:id", s.UnpinHandler)
	r.POST("/api/cache/pin", s.CachePinHandler)
	r.DELETE("/api/cache/pin/:id", s.CacheUnpinHandler)
	r.GET("/api/metrics", s.MetricsHandler)
	r.GET("/api/kv/blob/:digest", s.KvBlobHandler)
	r.POST("/api/fleet/assign-hold", s.AssignHoldHandler)
	internal := r.Group("/internal", internalLoopbackOnly())
	internal.POST("/cross-queue-seq", s.CrossQueueSeqHandler)
	internal.POST("/render-chat", s.RenderChatHandler)
	internal.POST("/parse-tool-output", s.ParseToolOutputHandler)
	internal.POST("/parse-tool-output/session", s.OpenToolParseSessionHandler)
	internal.POST("/parse-tool-output/chunk", s.ToolParseSessionChunkHandler)
	internal.POST("/parse-tool-output/close", s.CloseToolParseSessionHandler)
	internal.GET("/kv-snapshot", s.RuntimeKVSnapshotHandler)

	// Local model cache management (new implementation is at end of function)
	r.POST("/api/pull", s.PullHandler)
	r.POST("/api/push", s.PushHandler)
	r.HEAD("/api/tags", s.ListHandler)
	r.GET("/api/tags", s.ListHandler)
	r.POST("/api/show", s.ShowHandler)
	r.DELETE("/api/delete", s.DeleteHandler)

	r.POST("/api/me", s.WhoamiHandler)

	r.POST("/api/signout", s.SignoutHandler)
	// deprecated
	r.DELETE("/api/user/keys/:encodedKey", s.SignoutHandler)

	// Create
	r.POST("/api/create", s.CreateHandler)
	r.POST("/api/blobs/:digest", s.CreateBlobHandler)
	r.HEAD("/api/blobs/:digest", s.HeadBlobHandler)
	r.POST("/api/copy", s.CopyHandler)
	r.POST("/api/repair", s.RepairHandler)
	r.POST("/api/experimental/web_search", s.WebSearchExperimentalHandler)
	r.POST("/api/experimental/web_fetch", s.WebFetchExperimentalHandler)

	// Inference
	r.GET("/api/ps", s.PsHandler)
	r.GET("/api/image/workflows", s.ImageWorkflowsHandler)
	r.POST("/api/generate", s.withInferenceRequestLogging("/api/generate", s.assignmentTokenMiddleware(), s.runtimeGenerateProxy(), s.GenerateHandler)...)
	r.POST("/api/chat", s.withInferenceRequestLogging("/api/chat", s.assignmentTokenMiddleware(), s.runtimeChatProxy(), s.ChatHandler)...)
	r.POST("/api/embed", s.EmbedHandler)
	r.POST("/api/embeddings", s.EmbeddingsHandler)
	r.POST("/api/score", s.ScoreHandler)

	// Inference (OpenAI compatibility)
	// TODO(cloud-stage-a): apply Modelfile overlay deltas for local models with cloud
	// parents on v1 request families while preserving this explicit :cloud passthrough.
	r.POST("/v1/chat/completions", s.withInferenceRequestLogging("/v1/chat/completions", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable), s.runtimeV1ChatCompletionsProxy(), s.sglangChatCompletionsProxy(), middleware.ChatMiddleware(), s.ChatHandler)...)
	r.POST("/v1/chat/completions/batch", s.withInferenceRequestLogging("/v1/chat/completions/batch", s.runtimeV1ChatCompletionsBatchProxy())...)
	r.POST("/v1/completions", s.withInferenceRequestLogging("/v1/completions", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable), middleware.CompletionsMiddleware(), s.GenerateHandler)...)
	r.POST("/v1/embeddings", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable), middleware.EmbeddingsMiddleware(), s.EmbedHandler)
	r.GET("/v1/models", middleware.ListMiddleware(), s.ListHandler)
	r.GET("/v1/models/:model", s.maybeProxyElizaV1ModelGet(), middleware.RetrieveMiddleware(), s.ShowHandler)
	r.POST("/v1/responses", s.withInferenceRequestLogging("/v1/responses", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable), middleware.ResponsesMiddleware(), s.ChatHandler)...)
	// OpenAI-compatible image generation endpoints
	r.POST("/v1/images/generations", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable), middleware.ImageGenerationsMiddleware(), s.GenerateHandler)
	r.POST("/v1/images/edits", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable), middleware.ImageEditsMiddleware(), s.GenerateHandler)
	// OpenAI-compatible audio endpoints
	r.POST("/v1/audio/transcriptions", middleware.TranscriptionMiddleware(), s.TranscriptionHandler)
	r.POST("/v1/audio/speech", middleware.SpeechMiddleware(), s.SpeechHandler)
	r.GET("/v1/audio/voices", s.VoicesHandler)
	// OpenAI-compatible async text-to-video (local Wan via training run_script queue)
	r.POST("/v1/videos", middleware.VideoCreateMiddleware(), s.VideoCreateHandler)
	r.GET("/v1/videos/:id", s.VideoGetHandler)
	r.GET("/v1/videos/:id/content", s.VideoContentHandler)

	// Inference (Anthropic compatibility)
	r.POST("/v1/messages", s.withInferenceRequestLogging("/v1/messages", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), cloudV1InferencePassthrough(cloudErrRemoteInferenceUnavailable), middleware.AnthropicMessagesMiddleware(), s.ChatHandler)...)

	s.registerTrainingRoutes(r)

	if rc != nil {
		// wrap old with new
		rs := &registry.Local{
			Client:   rc,
			Logger:   slog.Default(), // TODO(bmizerany): Take a logger, do not use slog.Default()
			Fallback: r,

			Prune: PruneLayers,
		}
		return rs, nil
	}

	return r, nil
}

func Serve(ln net.Listener) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	discover.LogStartupBanner()
	slog.Info("server config", "env", envconfig.Values())
	cloudDisabled, _ := internalcloud.Status()
	slog.Info(fmt.Sprintf("Ollama cloud disabled: %t", cloudDisabled))

	blobsDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}
	if err := fixBlobs(blobsDir); err != nil {
		return err
	}

	if envconfig.LMStudioImport(true) {
		go func() {
			if err := SyncLMStudioModels(context.Background()); err != nil {
				slog.Warn("lm studio startup sync failed", "error", err)
			}
		}()
	}

	if !envconfig.NoPrune() {
		if _, err := manifest.Manifests(false); err != nil {
			slog.Warn("corrupt manifests detected, skipping prune operation.  Re-pull or delete to clear", "error", err)
		} else {
			// clean up unused layers and manifests
			if err := PruneLayers(); err != nil {
				return err
			}

			manifestsPath, err := manifest.Path()
			if err != nil {
				return err
			}

			if err := manifest.PruneDirectory(manifestsPath); err != nil {
				return err
			}
		}
	}

	s := &Server{addr: ln.Addr()}
	ensureLoopbackGoURLEnv()

	var darwinSidecar *DarwinSidecar
	defer func() {
		if darwinSidecar != nil {
			darwinSidecar.Stop()
		}
	}()

	if err := s.initRequestLogging(); err != nil {
		return err
	}
	if err := agentstats.Init(envconfig.GemmaAgentLogPath()); err != nil {
		slog.Warn("gemma agent stats log disabled", "error", err)
	} else if p := agentstats.Path(); p != "" {
		slog.Info("gemma agent stats log enabled", "path", p, "version", version.Version)
	}

	var rc *ollama.Registry
	if useClient2 {
		var err error
		rc, err = ollama.DefaultRegistry()
		if err != nil {
			return err
		}
	}

	ctx, done := context.WithCancel(context.Background())
	schedCtx, schedDone := context.WithCancel(ctx)
	defer func() {
		// Cancel both contexts on early return (e.g. GenerateRoutes error).
		// On the normal serve path the signal handler calls done/schedDone explicitly.
		schedDone()
		done()
	}()

	if sc, err := BootstrapDarwinSidecar(ctx); err != nil {
		slog.Warn("darwin runtime sidecar bootstrap failed", "error", err)
	} else if sc != nil {
		darwinSidecar = sc
	}

	sched := InitScheduler(schedCtx)
	s.sched = sched
	sched.fifoYield = s.schedYieldToRuntimeFifo

	// Optional GPU training: Go owns public TCP :9500 and /api/train; embedded CPython runs training.py.
	// Default on (OLLAMA_TRAINING) so integrators see the feature; set false if libpython / torch deps are absent.
	// Close order on signals: training worker (stops Python job thread) before tearing down inference runners.
	if envconfig.TrainingEnabled(true) {
		if runtime.GOOS == "darwin" {
			if repo, rerr := trainingworker.RepoRoot(); rerr == nil && repo != "" {
				if err := EnsureDarwinTrainingEnv(ctx, repo); err != nil {
					slog.Warn("darwin training env not ready", "error", err)
				}
			}
		}
		if tw, terr := trainingworker.Start(ctx, sched); terr != nil {
			slog.Warn("training worker not started", "error", terr)
		} else {
			s.training = tw
			s.trainingDefer = newTrainingDeferQueue(s)
			tw.SetInferenceSubmitGuard(s.checkTrainingSubmitAllowed)
			tw.SetSubmitHandler(s.handleTrainingSubmitRequest)
			tw.SetDeferredJobStatusFn(s.deferredTrainingJobStatusJSON)
			tw.SetDeferredJobCancelFn(s.cancelDeferredTrainingJob)
			tw.SetDeferredListMergeFn(s.mergeDeferredJobsListJSON)
			go s.trainingDefer.start(ctx)
			if envconfig.BlockInferenceDuringTraining() {
				go s.runTrainingGPUPolicyMonitor(ctx)
			}
			tcp := strings.TrimSpace(os.Getenv("OLLAMA_TRAINING_TCP"))
			switch tcp {
			case "", "1":
				tcp = ":9500"
			}
			if tcp != "0" && tcp != "-" {
				go func() {
					if err := tw.ServePublicTCP(ctx, tcp); err != nil && !errors.Is(err, context.Canceled) {
						slog.Error("training public TCP stopped", "error", err)
					}
				}()
			}
		}
	} else {
		slog.Info("training disabled", "env", "OLLAMA_TRAINING=false")
	}

	if strings.TrimSpace(effectiveRuntimeURL()) != "" {
		startCoordPusher := s.training == nil || !envconfig.BlockInferenceDuringTraining()
		if startCoordPusher {
			go s.runRuntimeCoordinationPusher(ctx)
		}
	}

	if envconfig.RuntimeEmbedEnabled() {
		rw, rerr := runtimeworker.Start(ctx, "")
		if rerr != nil {
			slog.Warn("embedded runtime not started", "error", rerr)
		} else {
			s.runtimeEmbed = rw
			sharedPy := s.training != nil
			if sharedPy {
				// Runtime VRAM probes use nvidia-smi (GIL-releasing) instead of pynvml.
				_ = os.Setenv("ZEROLLAMA_RUNTIME_SHARED_PYTHON", "1")
			}
			attrs := []any{
				"url", runtimeworker.BaseURL(),
				"shared_interpreter", sharedPy,
			}
			if m := strings.TrimSpace(os.Getenv("LLAMA_MODEL")); m != "" {
				attrs = append(attrs, "gguf", m)
			}
			slog.Info("embedded python runtime (in-process)", attrs...)
		}
	}

	if runtimeProxyConfigured() {
		mode := "external sidecar"
		if runtimeworker.BaseURL() != "" {
			mode = "embedded"
		}
		if envconfig.RuntimeDefaultOn() {
			slog.Info(
				"python runtime proxy enabled",
				"url", effectiveRuntimeURL(),
				"mode", mode,
			)
		}
	}

	h, err := s.GenerateRoutes(rc)
	if err != nil {
		return err
	}

	http.Handle("/", h)

	bindHost, bindPort, _ := net.SplitHostPort(envconfig.Host().Host)
	if bindHost == "" {
		bindHost = "0.0.0.0"
	}
	// Do not claim "listening" yet — GPU discovery still runs before Serve accepts.
	slog.Info("listener prepared",
		"bind", net.JoinHostPort(bindHost, bindPort),
		"listener", ln.Addr().String(),
		"version", version.Version,
	)
	srvr := &http.Server{
		// Use http.DefaultServeMux so we get net/http/pprof for
		// free.
		//
		// TODO(bmizerany): Decide if we want to make this
		// configurable so it is not exposed by default, or allow
		// users to bind it to a different port. This was a quick
		// and easy way to get pprof, but it may not be the best
		// way.
		Handler: nil,
	}

	// listen for a ctrl+c and stop any loaded llm
	signals := make(chan os.Signal, 8)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	var shutdownViaSignal atomic.Bool
	go func() {
		<-signals
		slog.Info("shutting down server (Ctrl+C again to force quit)")
		// Second signal must kill the whole process immediately.
		go func() {
			<-signals
			slog.Warn("forced exit on second shutdown signal")
			// Restore default disposition so a third signal cannot be swallowed
			// if exitProcess somehow fails to terminate.
			signal.Reset(syscall.SIGINT, syscall.SIGTERM)
			exitProcess(1)
		}()

		sched.ResumeLoads()
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 3*time.Second)
		if err := srvr.Shutdown(shutCtx); err != nil {
			slog.Warn("HTTP shutdown timed out, forcing close", "error", err)
			_ = srvr.Close()
		}
		shutCancel()

		if s.training != nil {
			done := make(chan struct{})
			go func() {
				s.training.Close()
				close(done)
			}()
			select {
			case <-done:
			case <-time.After(2 * time.Second):
				slog.Warn("training close timed out during shutdown")
			}
		}
		schedDone()

		unloadDone := make(chan struct{})
		go func() {
			sched.unloadAllRunners()
			close(unloadDone)
		}()
		select {
		case <-unloadDone:
		case <-time.After(3 * time.Second):
			slog.Warn("runner unload timed out during shutdown")
		}
		if s.runtimeEmbed != nil {
			s.runtimeEmbed.Close()
		}
		if darwinSidecar != nil {
			darwinSidecar.Stop()
		}
		shutdownViaSignal.Store(true)
		done()
	}()

	s.sched.Run(schedCtx)

	// register the experimental webp decoder
	// so webp images can be used in multimodal inputs
	image.RegisterFormat("webp", "RIFF????WEBP", webp.Decode, webp.DecodeConfig)

	// At startup we retrieve GPU information so we can get log messages before loading a model
	// This will log warnings to the log in case we have problems with detected GPUs.
	// Why this matters: empty discovery → totalVRAM=0 → defaultNumCtx=4096 and CPU-only
	// layer layout even on Macs with Metal (see DiscoverBackendDevices / apple-silicon-metal.md).
	gpus := discover.GPUDevices(ctx, nil)
	discover.LogDetails(gpus)
	discover.LogStartupHardware(gpus)

	var totalVRAM uint64
	for _, gpu := range gpus {
		totalVRAM += gpu.TotalMemory - envconfig.GpuOverhead()
	}

	// Set default context based on VRAM tier
	// Use slightly lower thresholds (47/23 GiB vs. 48/24 GiB) to account for small differences in the exact value
	switch {
	case totalVRAM >= 47*format.GibiByte:
		s.defaultNumCtx = 262144
	case totalVRAM >= 23*format.GibiByte:
		s.defaultNumCtx = 32768
	default:
		s.defaultNumCtx = 4096
	}
	slog.Info("vram-based default context", "total_vram", format.HumanBytes2(totalVRAM), "default_num_ctx", s.defaultNumCtx)

	slog.Info("server listening",
		"bind", net.JoinHostPort(bindHost, bindPort),
		"listener", ln.Addr().String(),
		"url", envconfig.ConnectableHost().String(),
		"version", version.Version,
		"started_at", time.Now().Format(time.RFC3339),
	)

	// Register mDNS after startup work so the HTTP listener is about to accept connections.
	startNodeMDNS(ctx, ln)

	err = srvr.Serve(ln)
	// If server is closed from the signal handler, wait for the ctx to be done
	// otherwise error out quickly
	if !errors.Is(err, http.ErrServerClosed) {
		if s.training != nil {
			s.training.Close()
		}
		return err
	}
	<-ctx.Done()
	if shutdownViaSignal.Load() {
		// Exit from the main goroutine via _exit(2): os.Exit from the signal handler
		// can SIGSEGV in runtime.exit when embedded CPython/torch is loaded.
		exitProcess(0)
	}
	return nil
}

func waitForStream(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/json")
	var latest api.ProgressResponse
	for resp := range ch {
		switch r := resp.(type) {
		case api.ProgressResponse:
			latest = r
		case gin.H:
			status, ok := r["status"].(int)
			if !ok {
				status = http.StatusInternalServerError
			}
			errorMsg, ok := r["error"].(string)
			if !ok {
				errorMsg = "unknown error"
			}
			c.JSON(status, gin.H{"error": errorMsg})
			return
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown message type"})
			return
		}
	}

	c.JSON(http.StatusOK, latest)
}

func streamResponse(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/x-ndjson")
	c.Stream(func(w io.Writer) bool {
		val, ok := <-ch
		if !ok {
			return false
		}

		// errors are provided as a gin.H with an "error" field and
		// an optional "status" field.  For errors that are streamed
		// before any content, we need to set the status code and
		// content type for the error.
		if h, ok := val.(gin.H); ok {
			if e, ok := h["error"].(string); ok {
				status, ok := h["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				if !c.Writer.Written() {
					c.Header("Content-Type", "application/json")
					c.JSON(status, gin.H{"error": e})
				} else {
					if err := json.NewEncoder(c.Writer).Encode(gin.H{"error": e}); err != nil {
						slog.Error("streamResponse failed to encode json error", "error", err)
					}
				}

				return false
			}
		}

		bts, err := json.Marshal(val)
		if err != nil {
			slog.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
			return false
		}

		// Delineate chunks with new-line delimiter
		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			slog.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
			return false
		}

		return true
	})
}

func (s *Server) StatusHandler(c *gin.Context) {
	c.JSON(http.StatusOK, s.statusResponse(c.Request.Context()))
}

func (s *Server) WebSearchExperimentalHandler(c *gin.Context) {
	s.webExperimentalProxyHandler(c, "/api/web_search", cloudErrWebSearchUnavailable)
}

func (s *Server) WebFetchExperimentalHandler(c *gin.Context) {
	s.webExperimentalProxyHandler(c, "/api/web_fetch", cloudErrWebFetchUnavailable)
}

func (s *Server) webExperimentalProxyHandler(c *gin.Context, proxyPath, disabledOperation string) {
	body, err := readRequestBody(c.Request)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if len(bytes.TrimSpace(body)) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	}

	proxyCloudRequestWithPath(c, body, proxyPath, disabledOperation)
}

func (s *Server) WhoamiHandler(c *gin.Context) {
	base, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	if !strings.EqualFold(base.Hostname(), "ollama.com") {
		c.JSON(http.StatusOK, gin.H{
			"name":          "eliza-cloud",
			"cloud_host":    strings.TrimSuffix(cloudProxyBaseURL, "/"),
			"documentation": "Set ELIZACLOUD_API_KEY and use model names like {provider/model}:cloud; manage account at Eliza Cloud.",
		})
		return
	}

	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	user, err := client.Whoami(c)
	if err != nil {
		slog.Error(err.Error())
	}

	// user isn't signed in
	if user != nil && user.Name == "" {
		sURL, sErr := signinURL()
		if sErr != nil {
			slog.Error(sErr.Error())
			c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
			return
		}

		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": sURL})
		return
	}

	c.JSON(http.StatusOK, user)
}

func (s *Server) SignoutHandler(c *gin.Context) {
	base, err := url.Parse(cloudProxyBaseURL)
	if err == nil && !strings.EqualFold(base.Hostname(), "ollama.com") {
		c.JSON(http.StatusOK, gin.H{"status": "Eliza Cloud sessions are managed in the Eliza Cloud dashboard; no local sign-out required."})
		return
	}

	pubKey, err := auth.GetPublicKey()
	if err != nil {
		slog.Error("couldn't get public key", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))

	// todo allow other hosts
	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	err = client.Disconnect(c, encKey)
	if err != nil {
		var authError api.AuthorizationError
		if errors.As(err, &authError) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "you are not currently signed in"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	c.JSON(http.StatusOK, nil)
}

func (s *Server) PsHandler(c *gin.Context) {
	c.JSON(http.StatusOK, s.sched.ProcessSnapshot())
}

func toolCallId() string {
	const letterBytes = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return "call_" + strings.ToLower(string(b))
}

func preservedTokensForCompletion(builtinParser parsers.Parser) []string {
	if builtinParser != nil {
		return builtinParser.PreservedTokens()
	}
	return nil
}

func toolCallTagForCompletion(toolParser *tools.Parser) string {
	if toolParser == nil {
		return ""
	}
	return toolParser.Tag()
}

func (s *Server) ChatHandler(c *gin.Context) {
	checkpointStart := time.Now()

	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	// Trap 77: reject invented top-level keys on native /api/chat (parity with /v1).
	if err := api.CheckUnknownChatFields(body); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.Request.Body = io.NopCloser(bytes.NewReader(body))

	var req api.ChatRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := api.ApplyChatThinkingAliases(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if len(req.Tools) > 0 && formatHasGrammarConstraint(req.Format) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errGrammarWithTools})
		return
	}

	EnsureAgentPromptCacheKey(&req)
	recordInferenceAccessCacheKey(c, modality.ExtractPromptCacheKey(req.Options))
	modality.WarnPrefixMMCacheWithoutSessionKey(&req)

	reqCtx, cancelTimeout := applyRequestTimeout(c.Request.Context(), req.Timeout)
	if cancelTimeout != nil {
		defer cancelTimeout()
	}
	c.Request = c.Request.WithContext(reqCtx)
	if req.Timeout != nil {
		c.Set("request_timeout", req.Timeout)
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}

	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": errCloudUseOpenAICompat})
		return
	}

	name := modelRef.Name

	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	if modelRef.Source == modelSourceLocal && m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	// expire the runner
	if len(req.Messages) == 0 && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusForbidden, gin.H{"error": errCloudModelsNotSupported})
		return
	}

	// Native VLM: expand Videos before chatPrompt so the rendered prompt and llm.ImageData list match.
	// ResolveVideoPolicy here (not inside modality) so one policy value crosses preflight + expand.
	// ChatRequestHasVideoPayload includes pre-expanded video_spans — not only raw videos[] — so
	// SGLang-style clients cannot skip capability checks or video preflight before load/ffmpeg.
	hasVideo := modality.ChatRequestHasVideoPayload(&req)
	hasAudio := false
	for _, msg := range req.Messages {
		if len(msg.AudioClips) > 0 {
			hasAudio = true
			break
		}
	}
	if hasVideo {
		if err := m.CheckCapabilities(model.CapabilityVision, model.CapabilityVideo); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}
	if hasAudio {
		if err := m.CheckCapabilities(model.CapabilityAudio); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}

	policy := modality.ResolveVideoPolicy(m.Config)
	if req.Options == nil {
		req.Options = map[string]any{}
	}
	for k, v := range m.Options {
		if _, ok := req.Options[k]; !ok {
			req.Options[k] = v
		}
	}
	if hasVideo || hasAudio {
		opts, err := s.modelOptions(m, req.Options)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if err := modality.PreflightVideoVisionBudget(policy, opts.NumCtx, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if err := modality.PreflightMllamaSingleImage(policy, m.Config.ModelFamilies, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}
	imgLim, vidLim, audLim := envconfig.LimitMMDataPerRequest()
	if err := modality.PreflightLimitMMDataPerRequest(modality.LimitMMDataPerRequest{
		Image: imgLim, Video: vidLim, Audio: audLim,
	}, &req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	// WHY before ffmpeg: cheap hint for misconfigured SGLang clients; preprocessed preflight
	// catches invalid tensor shapes before ViT/subprocess work.
	modality.WarnPrefixMMCacheWithoutSessionKey(&req)
	if err := modality.PreflightPreprocessedInputs(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// SGLang #31832: demux WebM/MP4 input_audio containers to WAV before prompt flatten.
	if err := modality.ExpandAudioClipsInChatRequest(c.Request.Context(), &req); err != nil {
		c.JSON(modality.MediaHTTPStatus(err), gin.H{"error": err.Error()})
		return
	}
	// SGLang #31417: client media → 400; missing ffmpeg / host IO → 500.
	if err := modality.ExpandVideosInChatRequest(c.Request.Context(), policy, &req); err != nil {
		c.JSON(modality.MediaHTTPStatus(err), gin.H{"error": err.Error()})
		return
	}
	if modality.ChatRequestHasMultimodalPayload(&req) {
		modality.LogViTEmbedCacheSizing(req.Messages)
	}
	var mmTokenEstimate modality.MultimodalTokenEstimate
	if modality.ChatRequestHasMultimodalPayload(&req) {
		mmTokenEstimate = modality.EstimateMultimodalTokens(policy, &req)
		if mmTokenEstimate.HasValues() {
			recordInferenceMultimodalEstimate(c, mmTokenEstimate.ImageTokens, mmTokenEstimate.VideoTokens, mmTokenEstimate.AudioTokens)
		}
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if len(req.Tools) > 0 {
		caps = append(caps, model.CapabilityTools)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: false}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			// Set think to nil when being used with Anthropic API to connect to tools like claude code
			if _, ok := c.Get("relax_thinking"); ok {
				slog.Warn("model does not support thinking, relaxing thinking to nil", "model", req.Model)
				req.Think = nil
			} else if req.ThinkFromAlias {
				// Harness aliases on a non-thinking model are no-ops (minefield ceiling/stream probes).
				req.Think = nil
			} else if _, ok := c.Get("think_from_alias"); ok {
				req.Think = nil
			} else {
				c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
				return
			}
		}
	}

	if err := applyThinkingGate(&req.Think); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	streaming := req.Stream == nil || *req.Stream
	var streamCh chan any
	if streaming {
		streamCh = make(chan any, 32)
	}
	statusWriter := func(ch chan<- any, _ string, status, detail string, pos, depth int) {
		writeChatStatus(ch, req.Model, status, detail, pos, depth)
	}

	chatRoute := "chat"
	if c.Request.URL != nil && strings.HasPrefix(c.Request.URL.Path, "/v1/") {
		chatRoute = "openai"
	}
	chatModality := mlxModalityFromChat(&req)
	if req.Options == nil {
		req.Options = map[string]any{}
	}
	chatHints := mlxScheduleHints{
		Route:    chatRoute,
		Modality: chatModality,
		Stream:   streaming,
	}
	ensureQoSDefaults(req.Options, chatHints)
	schedCtx := ctxWithMLXScheduleHints(c.Request.Context(), chatHints)

	r, m, opts, ggmlCtx, releaseQoS, err := s.scheduleRunner(schedCtx, name.String(), caps, req.Options, req.KeepAlive, req.Shift, streamCh, statusWriter)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support chat", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	checkpointLoaded := time.Now()
	logInferencePhase(c, "runner_ready", req.Model, checkpointStart)

	if len(req.Messages) == 0 {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	var streamKeepalive *chatStreamSession
	if streaming && !req.DebugRenderOnly {
		streamKeepalive = beginChatStream(c, streamCh, req.Model)
		defer streamKeepalive.Wait()
	}

	msgs := append(m.Messages, req.Messages...)
	if req.Messages[0].Role != "system" && m.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, m)
	msgs = preservePriorThinkingForRender(msgs, req.Think)

	paddedLayoutPlan := modality.PaddedLayoutConsumePlanForChat(
		resolveRendererName(m), m.Config.ModelFamilies, msgs, renderers.RenderImgTags,
	)

	if shouldUseHarmony(m) && m.Config.Parser == "" {
		m.Config.Parser = "harmony"
	}

	var builtinParser parsers.Parser
	processedTools := req.Tools

	parserName := resolveParserName(m)
	if parserName != "" {
		builtinParser = parsers.ParserForName(parserName)
		if builtinParser != nil {
			// Determine last message for chat prefill
			var lastMessage *api.Message
			if len(msgs) > 0 {
				lastMessage = &msgs[len(msgs)-1]
			}
			// Initialize parser and get processed tools
			processedTools = builtinParser.Init(req.Tools, lastMessage, req.Think)
		}
	}

	truncate := req.Truncate == nil || *req.Truncate
	tokenBudget, detok := chatPromptLimits(m, opts, truncate, r.ContextLength(), r.Detokenize)
	chatCtx := withPromptCacheKey(c.Request.Context(), modality.ExtractPromptCacheKey(req.Options))
	prompt, images, messagesDropped, promptTokens, originalPromptTokens, err := chatPrompt(chatCtx, m, r.Tokenize, opts, msgs, processedTools, req.Think, truncate, tokenBudget, detok)
	if err != nil {
		slog.Error("chat prompt error", "error", err)
		abortStreamingJSON(c, streamKeepalive, streamCh, req.Model, http.StatusInternalServerError, err.Error())
		return
	}

	completionPromptTokens := mlxCompletionPromptTokens(m, promptTokens)
	paddedLayoutConsume := ""
	// WHY propagate mode without paddedIDs: splice failure downgrades to
	// deferred_multimodal_history for logs/access metrics even when inject ids are absent.
	paddedIDs, mode := ggmlPaddedCompletionPromptTokens(c.Request.Context(), m, r.Tokenize, prompt, msgs, paddedLayoutPlan)
	if mode != paddedLayoutPlan.Mode {
		paddedLayoutPlan.Mode = mode
	}
	if len(paddedIDs) > 0 {
		completionPromptTokens = paddedIDs
		paddedLayoutConsume = string(mode)
	}
	gemma4PaddedMedia := llmGemma4PaddedMediaSchedule(paddedLayoutConsume, msgs)
	if paddedLayoutPlan.Active {
		modality.LogPaddedLayoutRunnerStub(req.Model, paddedLayoutPlan.Stub, paddedLayoutPlan.Mode)
		recordInferencePaddedLayout(c, paddedLayoutPlan.Stub.Len, string(paddedLayoutPlan.Mode))
	}

	checkpointPromptReady := time.Now()
	logInferencePhase(c, "prompt_ready", req.Model, checkpointLoaded)
	logLargeMLXPromptIfNeeded(m, promptTokens, opts)
	recordInferencePromptSize(c, len(promptTokens), originalPromptTokens, messagesDropped)

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		dbg := &api.DebugInfo{
			RenderedTemplate: prompt,
			ImageCount:       len(images),
		}
		if paddedLayoutPlan.Active {
			dbg.PaddedInputIDsLen = paddedLayoutPlan.Stub.Len
			dbg.PaddedLayoutConsume = string(paddedLayoutPlan.Mode)
		}
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: dbg,
		})
		return
	}

	var thinkingState *thinking.Parser
	openingTag, closingTag := thinking.TagsForModel(m.Config.ModelFamily, m.Template.Template)
	if req.Think != nil && req.Think.Bool() && openingTag != "" && closingTag != "" {
		thinkingState = &thinking.Parser{
			OpeningTag: openingTag,
			ClosingTag: closingTag,
		}

		if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
			thinkingState.AddContent(openingTag)
		}
	}

	var toolParser *tools.Parser
	if len(req.Tools) > 0 && (builtinParser == nil || !builtinParser.HasToolSupport()) {
		toolParser = tools.NewParser(m.Template.Template, req.Tools)
	}

	type structuredOutputsState int
	const (
		structuredOutputsState_None structuredOutputsState = iota
		structuredOutputsState_ReadyToApply
		structuredOutputsState_Applying
	)

	ch := streamCh
	if ch == nil {
		ch = make(chan any)
	}
	runnerTokenize := r.Tokenize
	go func() {
		var sentDone bool
		defer func() {
			if streamKeepalive != nil {
				streamKeepalive.StopKeepalive()
			}
			// OpenAI clients (Mercury) require a final chunk with finish_reason or they
			// treat the stream as malformed when prefill ends without decode tokens.
			if req.Stream != nil && *req.Stream && !sentDone {
				slog.Warn("chat stream closed without done chunk; emitting synthetic finish",
					"model", req.Model,
					"client_canceled", c.Request.Context().Err() != nil,
				)
				emitSyntheticChatFinish(ch, req.Model)
			}
			close(ch)
		}()

		structuredOutputsState := structuredOutputsState_None
		firstToken := true
		var firstTokenAt time.Time

		for {
			var tb strings.Builder

			currentFormat := req.Format
			// structured outputs via double request is enabled when:
			// 1. the model supports the thinking capability and
			// 2. it uses a built-in parser or our generic thinking parser

			// Note that the current approach does not work for (potential future)
			// non-thinking models that emit anything before actual content. This
			// current approach uses the transition from parsed thinking content to
			// parsed non-thinking content as the signal to turn constraining on

			if req.Format != nil && structuredOutputsState == structuredOutputsState_None && ((builtinParser != nil || thinkingState != nil) && slices.Contains(m.Capabilities(), model.CapabilityThinking)) {
				currentFormat = nil
			}

			// sets up new context given parent context per request
			ctx, cancel := context.WithCancel(c.Request.Context())
			// Soft mid-stream preempt (M15f): interactive may cancel this ctx.
			if s.sched != nil {
				s.sched.mlxGate.bindPreemptCancel(schedulerModelKey(m), modality.ExtractPromptCacheKey(req.Options), cancel)
			}
			err := r.Completion(ctx, llm.CompletionRequest{
				Prompt:              prompt,
				PromptTokens:        completionPromptTokens,
				PaddedLayoutConsume: paddedLayoutConsume,
				Images:              images,
				Format:              currentFormat,
				Options:             opts,
				Shift:               req.Shift == nil || *req.Shift,
				Truncate:            truncate,
				PreservedTokens:     preservedTokensForCompletion(builtinParser),
				ToolCallTag:         toolCallTagForCompletion(toolParser),
				LeadingBOS:          leadingBOSForModel(m),
				Logprobs:            req.Logprobs,
				TopLogprobs:         req.TopLogprobs,
				PromptCacheKey:      modality.ExtractPromptCacheKey(req.Options),
				CacheReset:          mlxQoSFromOptions(req.Options).CacheReset,
				SessionViTOverlay:   modality.SessionViTOverlayEnabled(req.Options),
				Gemma4PaddedMedia:   gemma4PaddedMedia,
			}, func(r llm.CompletionResponse) {
				if emitMLXPrefillStatus(ch, req.Model, r.PrefillProcessed, r.PrefillTotal, r.Content, r.Done) {
					return
				}
				if firstToken {
					firstToken = false
					firstTokenAt = time.Now()
					if streamKeepalive != nil {
						streamKeepalive.StopKeepalive()
					}
					logInferencePhase(c, "first_token", req.Model, checkpointPromptReady)
				}
				metrics := api.Metrics{
					PromptEvalCount:            r.PromptEvalCount,
					PromptEvalDuration:         r.PromptEvalDuration,
					EvalCount:                  r.EvalCount,
					EvalDuration:               r.EvalDuration,
					CachedPromptTokens:         r.PromptEvalCachedCount,
					CachedTokensHost:           r.PromptEvalCachedHost,
					CachedTokensStorage:        r.PromptEvalCachedStorage,
					CachedTokensStorageBackend: r.PromptEvalCachedStorageBackend,
					CacheCreationTokens:        r.PromptEvalCacheCreationCount,
				}
				if mmTokenEstimate.HasValues() {
					metrics.ImageTokens = mmTokenEstimate.ImageTokens
					metrics.VideoTokens = mmTokenEstimate.VideoTokens
					metrics.AudioTokens = mmTokenEstimate.AudioTokens
				}
				res := api.ChatResponse{
					Model:     req.Model,
					CreatedAt: time.Now().UTC(),
					Message:   api.Message{Role: "assistant", Content: r.Content},
					Done:      r.Done,
					Metrics:   metrics,
					Logprobs:  toAPILogprobs(r.Logprobs),
				}

				if r.Done {
					res.DoneReason = r.DoneReason.String()
					if r.PreemptedReason != "" {
						res.PreemptedReason = r.PreemptedReason
					}
					res.TotalDuration = time.Since(checkpointStart)
					res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
					applyPromptTruncation(&res, r, messagesDropped, originalPromptTokens)
					applyGgmlNumCtxChatResponse(&res, ggmlCtx)
					rememberMLXPromptChain(m, req.Options, prompt, msgs, runnerTokenize)
					recordInferenceCompletionDetails(c, res.DoneReason, r.PromptEvalCount, r.EvalCount, r.PromptEvalCachedCount, r.PromptEvalCachedHost, r.PromptEvalCachedStorage, r.PromptEvalCachedStorageBackend)
					if r.OriginalPromptTokens > 0 {
						recordInferencePromptSize(c, r.PromptEvalCount, r.OriginalPromptTokens, messagesDropped)
					}
					sentDone = true
				}

				if builtinParser != nil {
					slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser input", "parser", m.Config.Parser, "content", r.Content)

					content, thinking, toolCalls, err := builtinParser.Add(r.Content, r.Done)
					if err != nil {
						enqueueChatStreamErrorExtra(ch, req.Model, &sentDone, err.Error(), 0,
							errorExtraFromCheckpoints(checkpointStart, checkpointLoaded, firstTokenAt, !firstTokenAt.IsZero()))
						return
					}

					res.Message.Content = content
					res.Message.Thinking = thinking
					for i := range toolCalls {
						toolCalls[i].ID = toolCallId()
					}
					res.Message.ToolCalls = toolCalls

					tb.WriteString(thinking)
					// we are now receiving content from the model - we should start applying structured outputs
					if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && res.Message.Content != "" {
						structuredOutputsState = structuredOutputsState_ReadyToApply
						cancel()
						return
					}

					if res.Message.Content != "" || res.Message.Thinking != "" || len(res.Message.ToolCalls) > 0 || r.Done || len(res.Logprobs) > 0 {
						slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser output", "parser", m.Config.Parser, "content", content, "thinking", thinking, "toolCalls", toolCalls, "done", r.Done)
						if r.Done {
							applyEmptyGenClassifyChat(&res, opts.NumPredict, !checkpointLoaded.IsZero())
						}
						ch <- res
					} else {
						slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser empty output", "parser", m.Config.Parser)
					}
					return
				}

				if thinkingState != nil {
					thinkingContent, remainingContent := thinkingState.AddContent(res.Message.Content)
					if thinkingContent == "" && remainingContent == "" && !r.Done {
						// need to accumulate more to decide what to send
						return
					}
					res.Message.Thinking = thinkingContent
					tb.WriteString(thinkingContent)
					// emit the collected thinking text before restarting with structured outputs and clear unstructured content
					// to avoid leaking mixed tokens like "</think>Hello"
					if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && remainingContent != "" {
						structuredOutputsState = structuredOutputsState_ReadyToApply
						res.Message.Content = ""
						ch <- res
						cancel()
						return
					}
					res.Message.Content = remainingContent
				}

				if len(req.Tools) > 0 {
					toolCalls, content := toolParser.Add(res.Message.Content)
					if len(content) > 0 {
						res.Message.Content = content
					} else if len(toolCalls) > 0 {
						for i := range toolCalls {
							toolCalls[i].ID = toolCallId()
						}
						res.Message.ToolCalls = toolCalls
						res.Message.Content = ""
					} else if res.Message.Thinking != "" {
						// don't return, fall through to send
					} else {
						//  Send logprobs while content is being buffered by the parser for tool calls
						if len(res.Logprobs) > 0 && !r.Done {
							logprobRes := res
							logprobRes.Message.Content = ""
							logprobRes.Message.ToolCalls = nil
							ch <- logprobRes
						}

						if r.Done {
							res.Message.Content = toolParser.Content()
							ch <- res
						}
						return
					}
				}

				if r.Done && usesQwenStyleChat(m) {
					res.Message.Content = sanitizeAssistantContent(res.Message.Content)
				}

				if r.Done {
					applyEmptyGenClassifyChat(&res, opts.NumPredict, !checkpointLoaded.IsZero())
				}
				ch <- res
			})
			if err != nil {
				if structuredOutputsState == structuredOutputsState_ReadyToApply && strings.Contains(err.Error(), "context canceled") && c.Request.Context().Err() == nil {
					// only ignores error if it's a context cancellation due to setting structured outputs
				} else if isContextCanceled(err) && s.maybeEnqueueChatPreempted(
					ch, m, req.Options, req.Model, "", tb.String(), &sentDone,
					checkpointStart, checkpointLoaded, ggmlCtx,
				) {
					return
				} else {
					slog.Error("chat completion failed",
						"model", req.Model,
						"error", err,
						"client_canceled", c.Request.Context().Err() != nil,
					)
					extra := errorExtraFromCheckpoints(checkpointStart, checkpointLoaded, firstTokenAt, !firstTokenAt.IsZero())
					var serr api.StatusError
					if errors.As(err, &serr) {
						enqueueChatStreamErrorExtra(ch, req.Model, &sentDone, serr.ErrorMessage, serr.StatusCode, extra)
					} else {
						enqueueChatStreamErrorExtra(ch, req.Model, &sentDone, err.Error(), 0, extra)
					}
					return
				}
			}

			// ignored structured outputs cancellation falls through to here, start a new request with the structured outputs and updated prompt. use the
			if structuredOutputsState == structuredOutputsState_ReadyToApply {
				structuredOutputsState = structuredOutputsState_Applying
				msg := api.Message{
					Role:     "assistant",
					Thinking: tb.String(),
				}

				msgs = append(msgs, msg)
				prompt, _, _, promptTokens, _, err = chatPrompt(chatCtx, m, r.Tokenize, opts, msgs, processedTools, req.Think, truncate, tokenBudget, detok)
				if err != nil {
					slog.Error("chat prompt error applying structured outputs", "error", err)
					enqueueChatStreamError(ch, req.Model, &sentDone, err.Error(), 0)
					return
				}
				completionPromptTokens = mlxCompletionPromptTokens(m, promptTokens)
				// force constraining by terminating thinking header, the parser is already at this state
				// when the last message is thinking, the rendered for gpt-oss cannot disambiguate between having the
				// model continue thinking or ending thinking and outputting the final message.
				// TODO(parthsareen): consider adding prefill disambiguation logic to the renderer for structured outputs.
				if shouldUseHarmony(m) || (builtinParser != nil && m.Config.Parser == "harmony") {
					prompt += "<|end|><|start|>assistant<|channel|>final<|message|>"
					if ids, err := runnerTokenize(chatCtx, prompt); err == nil {
						completionPromptTokens = mlxCompletionPromptTokens(m, ids)
					}
				}
				continue
			}

			break
		}
	}()

	if req.Stream != nil && !*req.Stream {
		var resp api.ChatResponse
		var toolCalls []api.ToolCall
		var allLogprobs []api.Logprob
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.ChatResponse:
				sbThinking.WriteString(t.Message.Thinking)
				sbContent.WriteString(t.Message.Content)
				resp = t
				if len(req.Tools) > 0 {
					toolCalls = append(toolCalls, t.Message.ToolCalls...)
				}
				// Accumulate logprobs from all chunks for non-streaming response
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		resp.Message.Content = sanitizeAssistantContent(sbContent.String())
		resp.Message.Thinking = sbThinking.String()
		resp.Logprobs = allLogprobs

		if len(toolCalls) > 0 {
			resp.Message.ToolCalls = toolCalls
		}

		c.JSON(http.StatusOK, resp)
		return
	}

	if streamKeepalive == nil {
		streamResponse(c, ch)
	}
}

func handleScheduleError(c *gin.Context, name string, err error) {
	switch {
	case errors.Is(err, errCapabilities), errors.Is(err, errRequired):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, context.DeadlineExceeded):
		// Per-call timeout (req.Timeout) — distinct from client disconnect (499).
		var to *api.Duration
		if v, ok := c.Get("request_timeout"); ok {
			to, _ = v.(*api.Duration)
		}
		body := gin.H{"error": "request timeout"}
		if to != nil && to.Duration > 0 {
			body["timeout_seconds"] = to.Duration.Seconds()
		}
		if reason := preemptedReasonFromErr(err); reason != "" {
			body["preempted_reason"] = reason
		}
		c.JSON(http.StatusGatewayTimeout, body)
	case errors.Is(err, context.Canceled):
		body := gin.H{"error": "request canceled"}
		if reason := preemptedReasonFromErr(err); reason != "" {
			// WHY: client gave up while deferred behind higher QoS — Hermes retries faster
			// when it knows this was a priority wait, not a generic cancel.
			body["preempted_reason"] = reason
		}
		c.JSON(499, body)
	case errors.Is(err, ErrMaxQueue):
		metricsIncQueueReject()
		writeBusyUnavailable(c, err.Error(), preemptedReasonFromErr(err))
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found, try pulling it first", name)})
	// Why 400: model is configured for the Python runtime sidecar; the legacy ggml
	// runner will never load it — caller should use /api/generate or /api/chat via runtime.
	case errors.Is(err, ErrRuntimeInferenceModel):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	// Why 503: runtime sidecar holds Metal; a second ggml runner would contend on the
	// same device. Caller should unload the runtime model or use runtime routing.
	case errors.Is(err, ErrDarwinMetalContention):
		writeBusyUnavailable(c, err.Error(), preemptedReasonFromErr(err))
	case errors.Is(err, ErrEdgeGgmlRunnerDisabled), errors.Is(err, llm.ErrGgmlRunnerUnlinked):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func writeBusyUnavailable(c *gin.Context, errMsg string, preemptedReason ...string) {
	c.Header("Retry-After", strconv.Itoa(defaultBusyRetryAfterSec))
	body := gin.H{
		"error":       errMsg,
		"retry_after": defaultBusyRetryAfterSec,
	}
	if len(preemptedReason) > 0 && preemptedReason[0] != "" {
		body["preempted_reason"] = preemptedReason[0]
	}
	c.JSON(http.StatusServiceUnavailable, body)
}

// handleExternalImageGenerate runs modality_backends.image=external-image via OLLAMA_EXTERNAL_IMAGE_BIN.
func (s *Server) handleExternalImageGenerate(c *gin.Context, req api.GenerateRequest, cfg model.ConfigV2, checkpointStart time.Time) {
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}
	if req.Options == nil {
		req.Options = map[string]any{}
	}
	imgHints := mlxScheduleHints{
		Route:    "image_generation",
		Modality: mlxModalityImageGeneration,
		Stream:   false,
	}
	if err := s.waitRequestQoS(c.Request.Context(), nil, req.Options, imgHints); err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	w, h := req.Width, req.Height
	if ig := cfg.ImageGeneration; ig != nil {
		if w <= 0 && ig.Width > 0 {
			w = int32(ig.Width)
		}
		if h <= 0 && ig.Height > 0 {
			h = int32(ig.Height)
		}
	}
	if w <= 0 {
		w = 512
	}
	if h <= 0 {
		h = 512
	}
	var seed int64
	if s, ok := req.Options["seed"]; ok {
		switch v := s.(type) {
		case int:
			seed = int64(v)
		case int64:
			seed = v
		case float64:
			seed = int64(v)
		}
	}
	loadStart := time.Now()
	pngData, err := modality.GenerateExternalImage(c.Request.Context(), cfg, req.Prompt, w, h, seed)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	b64 := base64.StdEncoding.EncodeToString(pngData)
	isStreaming := req.Stream == nil || *req.Stream
	contentType := "application/x-ndjson"
	if !isStreaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)
	res := api.GenerateResponse{
		Model:      req.Model,
		CreatedAt:  time.Now().UTC(),
		Done:       true,
		DoneReason: "stop",
		Image:      b64,
	}
	res.Metrics.TotalDuration = time.Since(checkpointStart)
	res.Metrics.LoadDuration = loadStart.Sub(checkpointStart)
	if isStreaming {
		data, _ := json.Marshal(res)
		c.Writer.Write(append(data, '\n'))
		c.Writer.Flush()
		return
	}
	c.JSON(http.StatusOK, res)
}

// handleComfyImageGenerate runs modality_backends.image=comfyui via a running ComfyUI server.
//
// WHY this handler (vs scheduleRunner + MLX): Comfy owns DiT weights and node graphs;
// Zerollama only injects agent fields (prompt/seed/size/lora/control) into a named
// workflow template and returns the PNG. That keeps /api/generate and /v1/images/*
// identical for agents while avoiding months of per-family MLX ports.
func (s *Server) handleComfyImageGenerate(c *gin.Context, req api.GenerateRequest, m *Model, checkpointStart time.Time) {
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	workflowDir := modality.PathFor(m.Config, "comfy_workflow_dir")
	defaultWorkflow := modality.PathFor(m.Config, "comfy_default_workflow")

	var seed int64
	var seedSet bool
	if sv, ok := req.Options["seed"]; ok {
		seedSet = true
		switch v := sv.(type) {
		case int:
			seed = int64(v)
		case int64:
			seed = v
		case float64:
			seed = int64(v)
		}
	}

	comfyReq := comfyui.Request{
		WorkflowDir:     workflowDir,
		Workflow:        stringOption(req.Options, "workflow"),
		DefaultWorkflow: defaultWorkflow,
		Prompt:          req.Prompt,
		NegativePrompt:  stringOption(req.Options, "negative_prompt"),
		Width:           req.Width,
		Height:          req.Height,
		Steps:           req.Steps,
		Seed:            seed,
		SeedSet:         seedSet,
		LoRAName:        stringOption(req.Options, "lora"),
	}
	if strength, ok := floatOption(req.Options, "lora_strength"); ok {
		comfyReq.LoRAStrength = strength
		comfyReq.LoRAStrengthSet = true
	}
	if strength, ok := floatOption(req.Options, "control_strength"); ok {
		comfyReq.ControlStrength = strength
		comfyReq.ControlStrengthSet = true
	}
	if len(req.Images) > 0 {
		comfyReq.Image = req.Images[0]
	}
	if controlB64 := stringOption(req.Options, "control_image"); controlB64 != "" {
		if data, err := base64.StdEncoding.DecodeString(controlB64); err == nil {
			comfyReq.ControlImage = data
		} else {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid options.control_image: %v", err)})
			return
		}
	}

	loadStart := time.Now()
	result, err := comfyui.Generate(c.Request.Context(), comfyReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	b64 := base64.StdEncoding.EncodeToString(result.PNG)
	isStreaming := req.Stream == nil || *req.Stream
	contentType := "application/x-ndjson"
	if !isStreaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)
	res := api.GenerateResponse{
		Model:      req.Model,
		CreatedAt:  time.Now().UTC(),
		Done:       true,
		DoneReason: "stop",
		Image:      b64,
	}
	res.Metrics.TotalDuration = time.Since(checkpointStart)
	res.Metrics.LoadDuration = loadStart.Sub(checkpointStart)
	if isStreaming {
		data, _ := json.Marshal(res)
		c.Writer.Write(append(data, '\n'))
		c.Writer.Flush()
		return
	}
	c.JSON(http.StatusOK, res)
}

// stringOption reads a string field from a generate request's options map.
func stringOption(opts map[string]any, key string) string {
	if opts == nil {
		return ""
	}
	if v, ok := opts[key].(string); ok {
		return v
	}
	return ""
}

// floatOption reads a numeric field from a generate request's options map.
func floatOption(opts map[string]any, key string) (float64, bool) {
	if opts == nil {
		return 0, false
	}
	switch v := opts[key].(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	}
	return 0, false
}

// ImageWorkflowsHandler serves GET /api/image/workflows?model=NAME.
//
// WHY a dedicated discovery route: tool-using agents need required fields (image,
// control_image, …) before queuing a multi-minute Comfy job; reading Comfy API-format
// JSON is operator territory, not agent territory.
func (s *Server) ImageWorkflowsHandler(c *gin.Context) {
	modelName := c.Query("model")
	if modelName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model query parameter is required"})
		return
	}
	m, err := GetModel(modelName)
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", modelName)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}
	if modality.BackendFor(m.Config, model.ModalityImage) != model.BackendComfyUI {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model %q is not configured with modality_backends.image=comfyui", modelName)})
		return
	}
	workflowDir := modality.PathFor(m.Config, "comfy_workflow_dir")
	if workflowDir == "" {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "model manifest is missing backend_paths.comfy_workflow_dir"})
		return
	}
	workflows, err := comfyui.ListWorkflows(workflowDir)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"model":     modelName,
		"default":   modality.PathFor(m.Config, "comfy_default_workflow"),
		"workflows": workflows,
	})
}

// handleImageGenerate handles image generation requests within GenerateHandler.
// This is called when the model has the Image capability.
//
// WHY PrepareForImageGen is conditional: re-evicting while the same model is already
// loaded and generating would kill the MLX subprocess mid-stream. WHY aspect-only
// validation here: final width/height need mlx.GPUIsAvailable() in the runner child.
func (s *Server) handleImageGenerate(c *gin.Context, req api.GenerateRequest, modelName string, checkpointStart time.Time) {
	// Validate image dimensions
	const maxDimension int32 = 4096
	if req.Width > maxDimension || req.Height > maxDimension {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("width and height must be <= %d", maxDimension)})
		return
	}

	m, err := GetModel(modelName)
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if modality.IsExternalImageBackend(modality.BackendFor(m.Config, model.ModalityImage)) {
		s.handleExternalImageGenerate(c, req, m.Config, checkpointStart)
		return
	}

	if modality.BackendFor(m.Config, model.ModalityImage) == model.BackendComfyUI {
		// WHY PrepareForImageGen here (and not for external-image): Comfy diffusion can fill
		// a 16GB card; leaving ggml/runtime resident OOMs. WHY empty keepKey: there is no
		// Go-side MLX runner to preserve — all GPU work is in the ComfyUI process.
		vram.PrepareForImageGen(c.Request.Context(), s.sched, "")
		s.handleComfyImageGenerate(c, req, m, checkpointStart)
		return
	}

	isStreaming := req.Stream == nil || *req.Stream
	if req.Options == nil {
		req.Options = map[string]any{}
	}
	imgHints := mlxScheduleHints{
		Route:    "image_generation",
		Modality: mlxModalityImageGeneration,
		Stream:   isStreaming,
	}
	ensureQoSDefaults(req.Options, imgHints)
	schedCtx := ctxWithMLXScheduleHints(c.Request.Context(), imgHints)

	// Imagegen needs exclusive GPU; evict other ggml runners and the runtime sidecar first.
	// Skip when this model is already loaded — avoids tearing down an in-flight generation.
	keepKey := s.sched.keepModelKeyForUnload(m)
	if s.sched.findLoadedRunner(m) == nil {
		vram.PrepareForImageGen(c.Request.Context(), s.sched, keepKey)
	}

	// Schedule the runner for image generation
	runner, m, _, _, releaseQoS, err := s.scheduleRunner(schedCtx, modelName, []model.Capability{model.CapabilityImage}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	checkpointLoaded := time.Now()

	// Handle load-only request (empty prompt)
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	// Check streaming preference (computed above for QoS hints)
	contentType := "application/x-ndjson"
	if !isStreaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	// Get seed from options if provided
	var seed int64
	if s, ok := req.Options["seed"]; ok {
		switch v := s.(type) {
		case int:
			seed = int64(v)
		case int64:
			seed = v
		case float64:
			seed = int64(v)
		}
	}

	var images []llm.ImageData
	for i, imgData := range req.Images {
		images = append(images, llm.ImageData{ID: i, Data: imgData})
	}

	// Validate aspect ratio early (before paying subprocess startup cost), but do not
	// pre-resolve dimensions here — the runner subprocess knows GPU availability and
	// applies the correct maxSide for the target hardware.
	if req.AspectRatio != "" {
		if _, _, ok := size.ParseAspect(req.AspectRatio); !ok {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf(
				"unsupported aspect_ratio %q (supported: %s)",
				req.AspectRatio, strings.Join(size.SupportedAspects(), ", "),
			)})
			return
		}
	}

	var streamStarted bool
	var finalResponse api.GenerateResponse

	if err := runner.Completion(c.Request.Context(), llm.CompletionRequest{
		Prompt:      req.Prompt,
		Width:       req.Width,
		Height:      req.Height,
		AspectRatio: req.AspectRatio,
		Steps:       req.Steps,
		Seed:        seed,
		Images:      images,
	}, func(cr llm.CompletionResponse) {
		if isStreaming {
			streamStarted = true
		}
		res := api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			Done:      cr.Done,
		}

		if cr.TotalSteps > 0 {
			res.Completed = int64(cr.Step)
			res.Total = int64(cr.TotalSteps)
		}

		if cr.Image != "" {
			res.Image = cr.Image
		}
		if cr.Content != "" {
			res.Response = cr.Content
		}

		if cr.Done {
			res.DoneReason = cr.DoneReason.String()
			res.Metrics.TotalDuration = time.Since(checkpointStart)
			res.Metrics.LoadDuration = checkpointLoaded.Sub(checkpointStart)
			recordInferenceCompletionDetails(c, res.DoneReason, cr.PromptEvalCount, cr.EvalCount, cr.PromptEvalCachedCount, cr.PromptEvalCachedHost, cr.PromptEvalCachedStorage, cr.PromptEvalCachedStorageBackend)
		}

		if !isStreaming {
			finalResponse = res
			return
		}

		data, _ := json.Marshal(res)
		c.Writer.Write(append(data, '\n'))
		c.Writer.Flush()
	}); err != nil {
		// Only send JSON error if streaming hasn't started yet
		// (once streaming starts, headers are committed and we can't change status code).
		// WHY NDJSON error line: imagegen clients already consumed progress chunks; a bare
		// connection close produced "completed without image data" with no root cause.
		if !isStreaming || !streamStarted {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		} else {
			errResp := api.GenerateResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Done:      true,
				Response:  "error: " + err.Error(),
			}
			data, _ := json.Marshal(errResp)
			c.Writer.Write(append(data, '\n'))
			c.Writer.Flush()
		}
		return
	}

	if !isStreaming {
		if finalResponse.Done && finalResponse.Image == "" && strings.HasPrefix(finalResponse.Response, "error:") {
			c.JSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(strings.TrimPrefix(finalResponse.Response, "error:"))})
			return
		}
		c.JSON(http.StatusOK, finalResponse)
	}
}
