// GGUF manifest guessing fills empty Modelfile/config fields from model headers
// without a full weight walk. Why: operators and upstream manifests often omit
// parser/arch or set train-context num_ctx — that pre-allocates KV at load and
// hangs before first token. Guess runs at create and in-memory on GetModel; it
// does not rewrite on-disk manifests during GetModel (pull enrich, create, and repair do).
package server

import (
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

// defaultManifestNumCtxCap is the largest num_ctx written into manifest parameters by
// GGUF guessing. Why cap: train context (e.g. 262144) pre-allocates KV at load and can
// hang before first token — see docs/ggml_num_ctx.go and Phase 13 clamp policy.
const defaultManifestNumCtxCap = 8192

func ggufGuessingDisabled() bool {
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_DISABLE_GGUF_GUESS"))
	if s == "1" || strings.EqualFold(s, "true") || strings.EqualFold(s, "on") {
		return true
	}
	// LocalAI-compatible kill-switch name.
	s = strings.TrimSpace(os.Getenv("LOCALAI_DISABLE_GUESSING"))
	return strings.EqualFold(s, "true")
}

// GuessConfigFromGGUF fills empty config fields from GGUF metadata (architecture, sizes).
func GuessConfigFromGGUF(config *model.ConfigV2, kv ggml.KV) {
	if ggufGuessingDisabled() || config == nil || kv == nil {
		return
	}

	arch := kv.Architecture()
	if arch != "" && arch != "unknown" {
		if config.ModelFamily == "" {
			config.ModelFamily = arch
		}
		if len(config.ModelFamilies) == 0 {
			config.ModelFamilies = []string{arch}
		} else if !slices.Contains(config.ModelFamilies, arch) {
			config.ModelFamilies = append(config.ModelFamilies, arch)
		}
	}

	if config.ContextLen == 0 {
		if n := int(kv.ContextLength()); n > 0 {
			config.ContextLen = n
		}
	}
	if config.EmbedLen == 0 {
		if n := int(kv.EmbeddingLength()); n > 0 {
			config.EmbedLen = n
		}
	}
	if config.FileType == "" {
		if ft := kv.FileType().String(); ft != "" && ft != "unknown" {
			config.FileType = ft
		}
	}
	if config.Parser == "" {
		if p := guessParserFromKV(kv); p != "" {
			config.Parser = p
		}
	}
}

// GuessParametersFromGGUF fills missing manifest parameters from GGUF metadata.
func GuessParametersFromGGUF(params map[string]any, f *ggml.GGML) {
	if ggufGuessingDisabled() || params == nil || f == nil {
		return
	}
	kv := f.KV()

	if _, ok := params["num_ctx"]; !ok {
		if n := suggestedManifestNumCtx(kv); n > 0 {
			params["num_ctx"] = n
		}
	}

	if _, ok := params["spec_type"]; !ok {
		if llm.HasMTPDraft(f) {
			params["spec_type"] = "draft-mtp"
		}
	}

	// Architecture-specific stop tokens (matches createModel heuristics).
	arch := kv.Architecture()
	if _, ok := params["stop"]; !ok {
		switch arch {
		case "gemma4":
			params["stop"] = []string{"<turn|>"}
		case "qwen35", "qwen35moe":
			params["stop"] = []string{"<|im_end|>"}
		}
	}
}

func suggestedManifestNumCtx(kv ggml.KV) int {
	train := int(kv.ContextLength())
	if train <= 0 {
		return 0
	}
	if train <= defaultManifestNumCtxCap {
		return train
	}
	return defaultManifestNumCtxCap
}

// applyGGUFGuessToModel updates in-memory model state after manifest parse.
func applyGGUFGuessToModel(m *Model, f *ggml.GGML) {
	if m == nil || f == nil {
		return
	}
	GuessConfigFromGGUF(&m.Config, f.KV())
	m.EmbeddedMTP = llm.HasMTPDraft(f)
	// Prefer metadata already loaded for list/GetModel — avoids a second gguf.Open
	// just to probe tokenizer.chat_template (dominant /api/tags cost with many GGUFs).
	m.HasChatTemplate = f.KV().ChatTemplate() != ""
	if m.Options == nil {
		m.Options = make(map[string]any)
	}
	GuessParametersFromGGUF(m.Options, f)
}

func guessPrimaryGGUFLayer(layers []*layerGGML) *layerGGML {
	var best *layerGGML
	var bestParams uint64
	for _, layer := range layers {
		if layer == nil || layer.GGML == nil {
			continue
		}
		arch := layer.GGML.KV().Architecture()
		if isProjectorArchitecture(arch) {
			continue
		}
		params := layer.GGML.KV().ParameterCount()
		if best == nil || params > bestParams {
			best = layer
			bestParams = params
		}
	}
	return best
}

func guessFromBaseLayers(config *model.ConfigV2, params map[string]any, layers []*layerGGML) {
	if layer := guessPrimaryGGUFLayer(layers); layer != nil {
		GuessConfigFromGGUF(config, layer.GGML.KV())
		if params == nil {
			return
		}
		GuessParametersFromGGUF(params, layer.GGML)
	}
}

type ggufMetadataCacheEntry struct {
	ggml *ggml.GGML
	size int64
	mod  time.Time
}

var ggufMetadataCache sync.Map // path → ggufMetadataCacheEntry

func loadGGUFMetadataAt(path string) (*ggml.GGML, error) {
	defer func() {
		if r := recover(); r != nil {
			slog.Warn("gguf metadata load panicked", "path", path, "panic", r)
		}
	}()
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if v, ok := ggufMetadataCache.Load(path); ok {
		ent := v.(ggufMetadataCacheEntry)
		if ent.ggml != nil && ent.size == fi.Size() && ent.mod.Equal(fi.ModTime()) {
			return ent.ggml, nil
		}
	}
	data, err := llm.LoadModelMetadata(path)
	if err != nil {
		return nil, err
	}
	ggufMetadataCache.Store(path, ggufMetadataCacheEntry{
		ggml: data,
		size: fi.Size(),
		mod:  fi.ModTime(),
	})
	return data, nil
}
