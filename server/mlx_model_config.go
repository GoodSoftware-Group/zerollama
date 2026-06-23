package server

// MLX schedule helpers: context cap, prompt budget, PromptTokens passthrough, keep-alive floor.
// Why separate file: agent clients send 100k+ token megaprompts to safetensors models;
// HF config.json and VRAM tier defaults can inflate num_ctx beyond the real window.
// See docs/mlx-agent-prompts.md.

import (
	"fmt"
	"log/slog"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
	xserver "github.com/ollama/ollama/x/server"
)

// enrichMLXModelConfig fills ContextLen and ModelFamily from safetensors config.json
// when the modelfile config blob omits them (common for registry MLX models).
// Why downward-only update: multimodal exports may store vocab_size in
// text_config.max_position_embeddings; x/server/show.go corrects that before we apply.
func enrichMLXModelConfig(m *Model) {
	if m == nil || !m.IsMLX() {
		return
	}
	n := model.ParseName(m.Name)
	info, err := xserver.GetSafetensorsLLMInfo(n)
	if err != nil {
		slog.Debug("mlx model config enrich skipped", "model", m.ShortName, "error", err)
		return
	}

	if m.Config.ModelFamily == "" {
		if arch, ok := info["general.architecture"].(string); ok && arch != "" {
			m.Config.ModelFamily = arch
		}
	}
	if m.Config.Parser == "" && isGptOSSFamily(m.Config.ModelFamily) {
		m.Config.Parser = "harmony"
	}
	if m.Config.Renderer == "" && isGptOSSFamily(m.Config.ModelFamily) {
		m.Config.Renderer = "harmony"
	}

	// Context length keys use normalized arch from GetSafetensorsLLMInfo (e.g. gemma4),
	// not necessarily manifest ModelFamily (may be HF class name).
	arch, _ := info["general.architecture"].(string)
	if arch == "" {
		arch = m.Config.ModelFamily
	}
	if arch != "" {
		if ctxLen, ok := info[arch+".context_length"].(int); ok && ctxLen > 0 {
			if m.Config.ContextLen == 0 || ctxLen < m.Config.ContextLen {
				if m.Config.ContextLen > ctxLen {
					slog.Info("mlx context_length updated from safetensors config",
						"model", m.ShortName,
						"from", m.Config.ContextLen,
						"to", ctxLen,
					)
				}
				m.Config.ContextLen = ctxLen
			}
		}
	}
	if m.Config.ContextLen == 0 && arch == "gemma4" {
		m.Config.ContextLen = 131072
	}
}

func effectiveChatPromptBudget(opts *api.Options, model *Model, runnerCtxLen int) int {
	budget := chatPromptTokenBudget(opts)
	modelCap := modelMaxNumCtx(model)
	if runnerCtxLen > 0 && (modelCap <= 0 || runnerCtxLen < modelCap) {
		modelCap = runnerCtxLen
	}
	if modelCap <= 0 {
		return budget
	}
	ctxBudget := renderPromptTokenBudget(modelCap, optsNumPredict(opts))
	if budget <= 0 || ctxBudget < budget {
		return ctxBudget
	}
	return budget
}

func optsNumPredict(opts *api.Options) int {
	if opts == nil {
		return 0
	}
	return opts.NumPredict
}

// capMLXScheduleOptions aligns merged scheduler options with MLX model limits.
// Why before GetRunner: scheduler sizes KV from opts.NumCtx at load; inflated client
// or tier defaults (262144 on 128GB Mac) skip tail truncate and force reload loops.
func capMLXScheduleOptions(model *Model, opts *api.Options) {
	if opts == nil || model == nil || !model.IsMLX() {
		return
	}
	enrichMLXModelConfig(model)
	maxCtx := modelMaxNumCtx(model)
	if maxCtx <= 0 {
		// Gemma4 HF exports omit a usable context_length in config.json; mlx runner uses 131072.
		if model.Config.ModelFamily == "gemma4" {
			maxCtx = 131072
		}
	}
	if maxCtx <= 0 {
		return
	}
	if opts.NumCtx <= 0 || opts.NumCtx > maxCtx {
		from := opts.NumCtx
		opts.NumCtx = maxCtx
		if from > maxCtx {
			slog.Info("num_ctx capped to mlx model maximum",
				"model", model.ShortName,
				"from", from,
				"to", maxCtx,
			)
		}
	}
	if opts.NumPredict > maxCtx {
		opts.NumPredict = maxCtx / 4
		if opts.NumPredict < 256 {
			opts.NumPredict = 256
		}
		slog.Info("num_predict capped for mlx context window",
			"model", model.ShortName,
			"max_ctx", maxCtx,
			"num_predict", opts.NumPredict,
		)
	}
}

// mlxCompletionPromptTokens returns pre-tokenized prompt ids for MLX when the
// server already tokenized during chat truncation (or mlxEnsurePromptTokens).
// Why nil for GGUF: llama-server path re-tokenizes from string; MLX Prepare must
// ingest exact IDs after token-ID front-drop to avoid drift and skip re-encode.
func mlxCompletionPromptTokens(m *Model, tokens []int) []int {
	if m == nil || !m.IsMLX() || len(tokens) == 0 {
		return nil
	}
	return tokens
}

// mlxKeepAliveFloor returns a keep-alive duration for MLX models that is at least
// mlxMinKeepAlive. MLX model load takes 3-10s; the default 5m keep-alive means any
// inter-request gap longer than 5 minutes forces a cold reload. For large models on
// Apple Silicon this is avoidable overhead — use a longer floor so the runner survives
// typical agent think-time pauses.
// The floor is only applied when the caller has not set an explicit keep-alive.
const mlxMinKeepAlive = 30 * time.Minute

func mlxKeepAliveFloor(m *Model, ka *api.Duration) *api.Duration {
	if m == nil || !m.IsMLX() {
		return ka
	}
	if ka != nil {
		// Caller set an explicit value — honour it (including keep_alive:0 to unload).
		return ka
	}
	floor := envconfig.KeepAlive()
	if floor < mlxMinKeepAlive {
		floor = mlxMinKeepAlive
	}
	d := api.Duration{Duration: floor}
	return &d
}

const mlxLongPromptWarnTokens = 32768

func logLargeMLXPromptIfNeeded(m *Model, promptTokens []int, opts *api.Options) {
	if m == nil || !m.IsMLX() || len(promptTokens) <= mlxLongPromptWarnTokens {
		return
	}
	numCtx := 0
	if opts != nil {
		numCtx = opts.NumCtx
	}
	slog.Warn("large mlx prompt; prefill may take several minutes",
		"prompt_tokens", len(promptTokens),
		"num_ctx", numCtx,
		"model_max_ctx", modelMaxNumCtx(m),
	)
}

// emitMLXPrefillStatus forwards mlxrunner prefill heartbeats to streaming clients.
// Returns true when the chunk was handled (no token content to emit).
func emitMLXPrefillStatus(ch chan<- any, model string, prefillProcessed, prefillTotal int, content string, done bool) bool {
	if ch == nil || prefillProcessed <= 0 || content != "" || done {
		return false
	}
	detail := "processing prompt"
	if prefillTotal > 0 {
		detail = fmt.Sprintf("%d/%d tokens", prefillProcessed, prefillTotal)
	}
	writeChatStatus(ch, model, "prefill", detail, prefillProcessed, 0)
	return true
}

func emitMLXPrefillGenerateStatus(ch chan<- any, model string, prefillProcessed, prefillTotal int, content string, done bool) bool {
	if ch == nil || prefillProcessed <= 0 || content != "" || done {
		return false
	}
	detail := "processing prompt"
	if prefillTotal > 0 {
		detail = fmt.Sprintf("%d/%d tokens", prefillProcessed, prefillTotal)
	}
	writeGenerateStatus(ch, model, "prefill", detail, prefillProcessed, 0)
	return true
}
