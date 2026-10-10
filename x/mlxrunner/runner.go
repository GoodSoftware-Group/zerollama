package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
	"github.com/ollama/ollama/x/uma"
)

// startLoadHeartbeat emits stderr lines while Load blocks inside mlx.Eval so the
// parent StatusWriter resets OLLAMA_LOAD_TIMEOUT. Gemma4-class first chunks can
// run >5m before the post-Eval progress log would fire.
func startLoadHeartbeat(modelName string) (stop func()) {
	done := make(chan struct{})
	go func() {
		t := time.NewTicker(15 * time.Second)
		defer t.Stop()
		n := 0
		for {
			select {
			case <-done:
				return
			case <-t.C:
				n++
				slog.Info("mlx load in progress", "model", modelName, "heartbeat", n)
			}
		}
	}()
	return func() { close(done) }
}

// Request is a short-lived struct that carries a completion request through
// a channel from the HTTP handler to the runner goroutine. The ctx field
// must travel with the request so that cancellation propagates across the
// channel boundary.
type Request struct {
	CompletionRequest
	Responses chan CompletionResponse
	Pipeline  func(context.Context, Request) error

	Ctx         context.Context //nolint:containedctx // Queued requests carry caller cancellation to the runner.
	Tokens      []int32
	SamplerOpts sample.Options
	// compiledGrammar is set by Prepare when Format asks for structured output.
	compiledGrammar *grammarCompilation
}

type Runner struct {
	Model         base.Model
	Tokenizer     *tokenizer.Tokenizer
	Requests      chan Request
	EmbedRequests chan EmbeddingRequest
	Sampler       *sample.Sampler
	cache         kvCache
	scoreCache    *kvCache
	scoreHidden   *cache.HiddenCache
	contextLength int
	mlxThread     *mlxthread.Thread
	// spec is the speculative-decoding subsystem (MTP and/or PLD).
	spec      *speculation
	modelName string
	// grammarEngine is the structured-output subsystem; nil when the grammar
	// library is unavailable (plain inference still works).
	grammarEngine *grammarEngine
}

func (r *Runner) Load(modelName string) error {
	r.modelName = modelName
	stopHB := startLoadHeartbeat(modelName)
	defer stopHB()

	root, err := model.Open(modelName)
	if err != nil {
		return err
	}
	defer root.Close()

	m, err := base.New(root)
	if err != nil {
		return err
	}

	// Load all tensor blobs from manifest
	tensors, err := loadTensorsFromManifest(root)
	if err != nil {
		return err
	}

	// One GPU lease for weight materialization + pin/sweep/eval + compile enable.
	var draftModel base.DraftModel
	if err := func() error {
		if err := uma.LeaseBegin("load"); err != nil {
			return err
		}
		defer uma.LeaseEnd()
		defer mlx.Synchronize() // drain Metal before RELEASE (wishlist)

		// On Metal, materialize loaded tensors with CPU reads before any
		// weight graph exists so later Eval never commits a command buffer
		// that waits on file data (#17902). Chunk like the post-load eval:
		// one giant Eval of a 60GiB MoE is a Metal timeout / jetsam.
		if mlx.MetalIsAvailable() {
			vals := make([]*mlx.Array, 0, len(tensors))
			for t := range maps.Values(tensors) {
				if t != nil && t.Valid() {
					vals = append(vals, t)
				}
			}
			// Log *before* Eval: first gemma4 chunk can exceed OLLAMA_LOAD_TIMEOUT
			// (default 5m); post-Eval-only logs never reset the parent stall timer.
			const evalChunk = 32
			slog.Info("mlx materialize eval starting", "total", len(vals))
			for i := 0; i < len(vals); i += evalChunk {
				end := min(i+evalChunk, len(vals))
				if i == 0 || i%256 == 0 {
					slog.Info("mlx materialize eval", "done", i, "next", end, "total", len(vals), "peak", mlx.PrettyBytes(mlx.PeakMemory()))
				}
				mlx.Eval(vals[i:end]...)
			}
			slog.Info("mlx materialize eval done", "total", len(vals), "peak", mlx.PrettyBytes(mlx.PeakMemory()))
		}

		// Assign weights to model (model-specific logic). Target and draft weights
		// must be loaded before sweeping so tensors from a combined manifest are
		// not discarded before the draft model can retain them.
		slog.Info("mlx LoadWeights starting", "peak", mlx.PrettyBytes(mlx.PeakMemory()))
		if err := m.LoadWeights(tensors); err != nil {
			return err
		}
		slog.Info("mlx LoadWeights done", "peak", mlx.PrettyBytes(mlx.PeakMemory()))

		draft, err := loadDraftCompanion(root, m, tensors)
		if err != nil {
			return err
		}
		draftModel = draft

		slog.Info("mlx collect/pin starting")
		collected := mlx.Collect(m)
		if draft != nil {
			draftArrays := mlx.Collect(draft)
			collected = append(collected, draftArrays...)
			if root.Draft != nil {
				slog.Info("Loaded draft model", "tensor_prefix", root.Draft.TensorPrefix, "config", root.Draft.Config, "arrays", len(draftArrays))
			} else {
				slog.Info("Loaded draft model", "arrays", len(draftArrays))
			}
		}
		for _, arr := range collected {
			mlx.Pin(arr)
		}
		slog.Info("mlx sweep starting", "pinned", len(collected), "peak", mlx.PrettyBytes(mlx.PeakMemory()))
		mlx.Sweep()
		slog.Info("mlx sweep done", "peak", mlx.PrettyBytes(mlx.PeakMemory()))
		// One giant Eval of a 60GiB MoE is a Metal command-buffer / jetsam
		// on 128GiB UMA; materialize in slices.
		const evalChunk = 32
		slog.Info("mlx load eval starting", "total", len(collected))
		for i := 0; i < len(collected); i += evalChunk {
			end := i + evalChunk
			if end > len(collected) {
				end = len(collected)
			}
			if i == 0 || i%256 == 0 {
				slog.Info("mlx load eval", "done", i, "next", end, "total", len(collected), "peak", mlx.PrettyBytes(mlx.PeakMemory()))
			}
			mlx.Eval(collected[i:end]...)
		}
		slog.Info("mlx load eval done", "total", len(collected), "peak", mlx.PrettyBytes(mlx.PeakMemory()))
		// LoadWeights / MoE fuse / linear-attn pack leave transform buffers
		// in MLX's allocator pool until something clears it. Drop them now so
		// idle post-load memory matches the pinned weights (upstream b68b112b).
		mlx.ClearCache()
		configureWiredMemory()

		r.Model = m
		r.Tokenizer = m.Tokenizer()
		r.contextLength = m.MaxContextLength()
		r.Sampler = sample.New(r.contextLength)
		if suppressReservedEnabled() {
			if ids := reservedSampleBanIDs(r.Tokenizer); len(ids) > 0 {
				r.Sampler.SetBannedIDs(ids)
				slog.Info("mlx suppress reserved sample ids", "n", len(ids))
			}
		}
		r.spec = newSpeculation(r, draftModel)
		if r.spec != nil {
			if data, err := root.Manifest.ReadConfig("config.json"); err == nil {
				r.spec.sparseMoE = configSparseMoE(data)
			}
			if r.spec.sparseMoE && r.spec.draft != nil {
				slog.Info("mlx MTP default off (MoE); set enable_mtp=true to use the draft head")
			}
		}
		if MTPRequire() && (r.spec == nil || r.spec.draft == nil) {
			return fmt.Errorf("ZEROLLAMA_MLX_MTP=require: checkpoint has no MTP/draft head (refuse silent AR demotion)")
		}
		r.installOptiqOwnedHook()
		if uma.OptiqTokenTailEnabled() {
			if err := uma.EnsureOptiqTokenTailSession(); err != nil {
				slog.Warn("uma: optiq token-tail session not ready", "error", err)
				if uma.OptiqTokenTailRequire() {
					return fmt.Errorf("uma optiq token-tail: %w", err)
				}
			} else {
				slog.Info("uma: optiq GRAPH token-tail session ready",
					"mode", os.Getenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_TAIL"))
			}
		}

		r.grammarEngine = newGrammarEngine(logitsWidth(m), r.Tokenizer)

		mlx.EnableCompile()
		return nil
	}(); err != nil {
		return err
	}

	return nil
}

func configureWiredMemory() {
	if !mlx.GPUIsAvailable() {
		return
	}

	active := mlx.ActiveMemory()
	maxRecommended, err := mlx.MaxRecommendedWorkingSetSize()
	if err != nil {
		slog.Warn("Unable to query MLX recommended working set; using pageable memory", "error", err)
		return
	}

	limit := min(active, maxRecommended)
	previous, err := mlx.SetWiredLimit(limit)
	if err != nil {
		slog.Warn("Unable to configure MLX wired memory; using pageable memory",
			"active", mlx.PrettyBytes(active),
			"limit", mlx.PrettyBytes(limit),
			"error", err)
		return
	}

	if active > maxRecommended {
		slog.Warn("MLX model exceeds the recommended working set; performance may be degraded",
			"active", mlx.PrettyBytes(active),
			"recommended", mlx.PrettyBytes(maxRecommended))
	}
	// Limiting residency to the loaded model's active allocations avoids
	// reserving the remaining capacity for growing KV caches.
	slog.Debug("Configured MLX wired memory",
		"active", mlx.PrettyBytes(active),
		"limit", mlx.PrettyBytes(limit),
		"previous", mlx.PrettyBytes(previous))
}

// loadTensorsFromManifest loads all tensor blobs from the manifest into a
// flat map, deduplicating by digest and remapping safetensors key suffixes.
//
// Uses a two-phase approach: first loads all raw tensors, then remaps
// .bias → _qbias with complete knowledge of which base names have .scale
// entries. This avoids a race condition where Go map iteration order could
// cause .bias to be processed before .scale within the same blob.
func loadTensorsFromManifest(root *model.Root) (map[string]*mlx.Array, error) {
	// Phase 1: Load all tensors raw from all blobs
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	layers := root.Manifest.GetTensorLayers("")
	var digests int
	for _, layer := range layers {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		digests++
		blobPath := root.Manifest.BlobPath(layer.Digest)
		if digests == 1 || digests%8 == 0 {
			slog.Info("mlx loading safetensors blob", "n", digests, "path", blobPath)
		}
		for name, arr := range mlx.Load(blobPath) {
			rawTensors[name] = arr
		}
	}

	allTensors := remapLoadedTensors(rawTensors)
	slog.Info("Loaded tensors from manifest", "count", len(allTensors), "blobs", digests, "source_dir", root.Manifest.SourceDir)
	return allTensors, nil
}

// remapLoadedTensors maps mlx-lm affine suffixes onto the names MakeLinearLayer
// expects (.weight_scale / .weight_qbias). Companion MTP packs use .scales/.biases
// next to .weight rather than .weight.scale.
func remapLoadedTensors(rawTensors map[string]*mlx.Array) map[string]*mlx.Array {
	scaleBaseNames := make(map[string]bool)
	allTensors := make(map[string]*mlx.Array, len(rawTensors))
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scales") {
			baseName := strings.TrimSuffix(name, ".scales")
			allTensors[baseName+".weight_scale"] = arr
			scaleBaseNames[baseName+".weight"] = true
			continue
		}
		if strings.HasSuffix(name, ".scale") {
			// Affine quant companions are `*.weight.scale`. DeepSeek-V4 HC mix
			// tensors are literally named `attn_hc.scale` / `hc_head.scale`.
			if strings.HasSuffix(name, ".weight.scale") {
				baseName := strings.TrimSuffix(name, ".scale")
				allTensors[baseName+"_scale"] = arr
				scaleBaseNames[baseName] = true
			} else {
				allTensors[name] = arr
			}
		}
	}

	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".scales") {
			continue
		}
		if strings.HasSuffix(name, ".biases") {
			baseName := strings.TrimSuffix(name, ".biases")
			if scaleBaseNames[baseName+".weight"] {
				allTensors[baseName+".weight_qbias"] = arr
				continue
			}
		}
		if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				allTensors[baseName+"_qbias"] = arr
			} else {
				allTensors[name] = arr
			}
		} else {
			allTensors[name] = arr
		}
	}
	return allTensors
}

// loadDraftCompanion attaches an in-manifest or inline draft head. A broken
// companion logs and falls back to PLD/AR unless ZEROLLAMA_MLX_MTP=require
// (mlx-serve: say so instead of a silent slower path).
func loadDraftCompanion(root *model.Root, m base.Model, tensors map[string]*mlx.Array) (base.DraftModel, error) {
	draft, err := base.NewDraft(root, m)
	if err != nil {
		if MTPRequire() {
			return nil, err
		}
		slog.Warn("mlx draft companion not loaded; PLD/AR", "error", err)
		draft = nil
	}
	if draft != nil {
		if err := draft.LoadWeights(tensors); err != nil {
			if MTPRequire() {
				return nil, err
			}
			slog.Warn("mlx draft weights failed; PLD/AR", "error", err)
			draft = nil
		}
	}
	if draft == nil {
		if sd, ok := m.(base.SelfDraft); ok {
			draft = sd.SelfDraft()
		}
	}
	if draft != nil {
		if n := quantizeDraftCompanion(draft); n > 0 {
			slog.Info("mlx draft weights quantized to 4-bit", "layers", n)
		}
	}
	return draft, nil
}

func (r *Runner) Run(host, port string, mux http.Handler) error {
	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				err := r.runRequest(request)
				if err != nil {
					slog.Info("Request terminated", "error", err)
					var statusErr api.StatusError
					if !errors.As(err, &statusErr) {
						statusErr = api.StatusError{
							StatusCode:   http.StatusInternalServerError,
							ErrorMessage: err.Error(),
						}
					}
					select {
					case request.Responses <- CompletionResponse{Error: &statusErr}:
					case <-request.Ctx.Done():
					}
				}

				close(request.Responses)
			case erequest := <-r.EmbedRequests:
				run := func() error { return r.runEmbed(erequest.Ctx, erequest) }
				var err error
				if r.mlxThread == nil {
					err = run()
				} else {
					err = r.mlxThread.Do(erequest.Ctx, run)
				}
				if err != nil {
					slog.Info("Embedding request terminated", "error", err)
				}
			}
		}
	})

	g.Go(func() error {
		slog.Info("Starting HTTP server", "host", host, "port", port)
		return http.ListenAndServe(net.JoinHostPort(host, port), mux)
	})

	return g.Wait()
}

func (r *Runner) runRequest(request Request) error {
	if r.mlxThread == nil {
		return request.Pipeline(request.Ctx, request)
	}

	return r.mlxThread.Do(request.Ctx, func() error {
		return request.Pipeline(request.Ctx, request)
	})
}
