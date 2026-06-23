// Package server — runner metadata probing after load.
//
// probeRunnerMetadata records ground-truth runner shape (effective num_ctx, parser,
// backend) for /api/ps and fleet status. Why: manifest parameters drift from what
// actually loaded; fleet routing and operators need probed fields without re-pulling
// weights. Disk reads run outside refMu when serving snapshots.
package server

import (
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/parsers"
)

func probeRunnerMetadata(runner *runnerRef) api.LoadedModelMetadata {
	if runner == nil {
		return api.LoadedModelMetadata{
			ProbedAt: time.Now().UTC(),
		}
	}

	meta := api.LoadedModelMetadata{
		ProbedAt: time.Now().UTC(),
		Backend:  runnerBackendLabel(runner),
	}
	if runner.Options != nil {
		meta.NumCtx = runner.Options.NumCtx
		meta.NumGPU = runner.Options.NumGPU
	}
	meta.NumParallel = runner.numParallel

	if runner.model != nil {
		meta.ManifestNumCtx = manifestNumCtxFromModel(runner.model)
		meta.Parser = strings.TrimSpace(resolveParserName(runner.model))
		if p := parsers.ParserForName(meta.Parser); p != nil {
			meta.SupportsThinking = p.HasThinkingSupport()
			meta.SupportsTools = p.HasToolSupport()
		}
		if runner.model.ModelPath != "" {
			if f, err := llm.LoadModelMetadata(runner.model.ModelPath); err == nil {
				if train := int(f.KV().ContextLength()); train > 0 {
					meta.TrainContextLength = train
				}
				if tmpl := f.KV().ChatTemplate(); tmpl != "" {
					meta.HasChatTemplate = true
				}
				if meta.Parser == "" {
					meta.Parser = guessParserFromKV(f.KV())
				}
			}
		}
	}

	if runner.llama != nil {
		if effective := runner.llama.ContextLength(); effective > 0 {
			meta.NumCtx = effective
		}
	}

	return meta
}

func manifestNumCtxFromModel(model *Model) int {
	if model == nil || model.Options == nil {
		return 0
	}
	return manifestIntOption(model.Options["num_ctx"])
}

func manifestIntOption(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int32:
		return int(n)
	case int64:
		return int(n)
	case float64:
		return int(n)
	case float32:
		return int(n)
	default:
		return 0
	}
}

func runnerBackendLabel(runner *runnerRef) string {
	if runner == nil {
		return ""
	}
	if runner.isImagegen {
		return "imagegen"
	}
	if runner.model != nil && runner.model.IsMLX() {
		return "mlx"
	}
	return "ggml"
}

func buildProcessModelResponse(runner *runnerRef) api.ProcessModelResponse {
	model := runner.model
	modelDetails := api.ModelDetails{
		Format:            model.Config.ModelFormat,
		Family:            model.Config.ModelFamily,
		Families:          model.Config.ModelFamilies,
		ParameterSize:     model.Config.ModelType,
		QuantizationLevel: model.Config.FileType,
	}

	mr := api.ProcessModelResponse{
		Model:     model.ShortName,
		Name:      model.ShortName,
		Size:      int64(runner.totalSize),
		SizeVRAM:  int64(runner.vramSize),
		Digest:    model.Digest,
		Details:   modelDetails,
		ExpiresAt: runner.expiresAt,
	}
	if runner.llama != nil {
		mr.ContextLength = runner.llama.ContextLength()
		total, vram := runner.llama.MemorySize()
		mr.Size = int64(total)
		mr.SizeVRAM = int64(vram)
	}

	var epoch time.Time
	if runner.expiresAt == epoch {
		mr.ExpiresAt = time.Now().Add(runner.sessionDuration)
	}

	return mr
}

func loadedMetadataForRunner(runner *runnerRef) api.LoadedModelMetadata {
	if runner == nil {
		return api.LoadedModelMetadata{}
	}
	if !runner.loadedMeta.ProbedAt.IsZero() {
		return runner.loadedMeta
	}
	return probeRunnerMetadata(runner)
}
