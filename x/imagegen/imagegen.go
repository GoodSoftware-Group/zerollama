package imagegen

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/flux2"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
	"github.com/ollama/ollama/x/imagegen/size"
)

// ImageModel is the interface for image generation models.
type ImageModel interface {
	GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64, progress func(step, total int)) (*mlx.Array, error)
}

var imageGenMu sync.Mutex

// imageGenMu serializes completions in the MLX runner process.
// WHY: peak VRAM on 16GB already equals one full pipeline; overlapping generates OOM.

// loadImageModel loads an image generation model.
func (s *server) loadImageModel() error {
	// Check memory requirements before loading
	var requiredMemory uint64
	if modelManifest, err := manifest.LoadManifest(s.modelName); err == nil {
		requiredMemory = uint64(modelManifest.TotalTensorSize())
	}
	availableMemory := mlx.GetMemoryLimit()
	if availableMemory > 0 && requiredMemory > 0 && availableMemory < requiredMemory {
		return fmt.Errorf("insufficient memory for image generation: need %d GB, have %d GB",
			requiredMemory/(1024*1024*1024), availableMemory/(1024*1024*1024))
	}

	// Detect model type and load appropriate model
	modelType := DetectModelType(s.modelName)
	slog.Info("detected image model type", "type", modelType)

	var model ImageModel
	switch modelType {
	case "Flux2KleinPipeline":
		m := &flux2.Model{}
		if err := m.Load(s.modelName); err != nil {
			return fmt.Errorf("failed to load flux2 model: %w", err)
		}
		model = m
	default:
		// Default to Z-Image for ZImagePipeline, FluxPipeline, etc.
		m := &zimage.Model{}
		if err := m.Load(s.modelName); err != nil {
			return fmt.Errorf("failed to load zimage model: %w", err)
		}
		model = m
	}

	s.imageModel = model
	return nil
}

type completionOutcome struct {
	imageData string
	err       error
}

func (s *server) handleImageCompletion(w http.ResponseWriter, r *http.Request, req Request) {
	imageGenMu.Lock()
	defer imageGenMu.Unlock()

	if req.Seed <= 0 {
		req.Seed = time.Now().UnixNano()
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	enc := json.NewEncoder(w)
	streamProgress := func(step, total int) {
		resp := Response{Step: step, Total: total}
		_ = enc.Encode(resp)
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	var outcome completionOutcome
	if err := s.mlxThread.Do(ctx, func() error {
		outcome = s.generateOnMLXThread(ctx, req, streamProgress)
		return nil
	}); err != nil {
		if ctx.Err() == nil {
			resp := Response{Content: fmt.Sprintf("error: %v", err), Done: true}
			data, _ := json.Marshal(resp)
			w.Write(data)
			w.Write([]byte("\n"))
		}
		return
	}

	if outcome.err != nil {
		if ctx.Err() != nil {
			resp := Response{Content: fmt.Sprintf("error: %v", ctx.Err()), Done: true}
			data, _ := json.Marshal(resp)
			w.Write(data)
			w.Write([]byte("\n"))
			return
		}
		resp := Response{Content: fmt.Sprintf("error: %v", outcome.err), Done: true}
		data, _ := json.Marshal(resp)
		w.Write(data)
		w.Write([]byte("\n"))
		return
	}

	resp := Response{Image: outcome.imageData, Done: true}
	data, _ := json.Marshal(resp)
	w.Write(data)
	w.Write([]byte("\n"))
	flusher.Flush()
}

func (s *server) generateOnMLXThread(ctx context.Context, req Request, progress func(step, total int)) completionOutcome {

	w, h := req.Width, req.Height
	maxSide := size.MaxSide(mlx.GPUIsAvailable())
	var resolveErr error
	w, h, resolveErr = size.Resolve(w, h, req.AspectRatio, maxSide)
	if resolveErr != nil {
		return completionOutcome{err: resolveErr}
	}
	if w != req.Width || h != req.Height {
		slog.Info("image dimensions", "width", w, "height", h, "requested_w", req.Width, "requested_h", req.Height, "aspect", req.AspectRatio)
	}

	img, err := s.imageModel.GenerateImage(ctx, req.Prompt, w, h, req.Steps, req.Seed, progress)
	if err != nil {
		return completionOutcome{err: err}
	}

	imageData, err := EncodeImageBase64(img)
	img.Free()
	mlx.ClearCache()
	mlx.MetalResetPeakMemory()
	if err != nil {
		return completionOutcome{err: fmt.Errorf("encoding: %w", err)}
	}
	return completionOutcome{imageData: imageData}
}
