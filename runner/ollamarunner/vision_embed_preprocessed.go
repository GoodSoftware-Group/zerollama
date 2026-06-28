package ollamarunner

import (
	"errors"
	"log/slog"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/ml"
)

// GetOrEncodePrecomputed returns materialized vision tensors from SGLang precomputed rows.
// WHY: SGLang MultiModalStaticCache skips ViT when the same feature hash was seen on this
// thread; global LRU shares hits across sessions on the same runner.
func (c *VisionEmbedCache) GetOrEncodePrecomputed(
	ingest model.PrecomputedMultimodalIngest,
	backend ml.Backend,
	ctx ml.Context,
	img llm.ImageData,
	sessionKey string,
	sessionOverlay bool,
) ([]input.Multimodal, error) {
	if c == nil {
		return ingest.MultimodalFromPrecomputed(ctx, img.PrecomputedFeature, img.GridTHW)
	}
	if len(img.PrecomputedFeature) == 0 {
		return nil, errors.New("precomputed feature is empty")
	}

	hash := hashPrecomputedFeature(img.PrecomputedFeature)
	sessionKey = normalizeSessionKey(sessionKey)

	if mm, ok := c.lookupCached(ctx, hash, sessionKey, sessionOverlay, "precomputed_embedding"); ok {
		return mm, nil
	}

	encodeCtx := backend.NewContext()
	mm, err := ingest.MultimodalFromPrecomputed(encodeCtx, img.PrecomputedFeature, img.GridTHW)
	if err != nil {
		encodeCtx.Close()
		return nil, err
	}
	cached, err := materializeMultimodal(backend, mm)
	encodeCtx.Close()
	if err != nil {
		return nil, err
	}

	c.storeCached(hash, sessionKey, sessionOverlay, cached)
	slog.Info("precomputed_embedding runner inject",
		"rows", len(img.PrecomputedFeature),
		"engine", "ollama",
	)
	return restoreMultimodal(ctx, cached), nil
}

// GetOrEncodeProcessorOutput runs vision tower on HF pixels with the same cache layers as PNG encode.
func (c *VisionEmbedCache) GetOrEncodeProcessorOutput(
	ingest model.ProcessorOutputMultimodalIngest,
	backend ml.Backend,
	ctx ml.Context,
	img llm.ImageData,
	sessionKey string,
	sessionOverlay bool,
) ([]input.Multimodal, error) {
	if c == nil {
		return ingest.MultimodalFromProcessorOutput(ctx, img.ProcessorPixelValues, img.GridTHW)
	}
	if len(img.ProcessorPixelValues) == 0 {
		return nil, errors.New("processor pixel_values is empty")
	}

	hash := hashProcessorPixelValues(img.ProcessorPixelValues)
	sessionKey = normalizeSessionKey(sessionKey)

	if mm, ok := c.lookupCached(ctx, hash, sessionKey, sessionOverlay, "processor_output"); ok {
		return mm, nil
	}

	encodeCtx := backend.NewContext()
	mm, err := ingest.MultimodalFromProcessorOutput(encodeCtx, img.ProcessorPixelValues, img.GridTHW)
	if err != nil {
		encodeCtx.Close()
		return nil, err
	}
	cached, err := materializeMultimodal(backend, mm)
	encodeCtx.Close()
	if err != nil {
		return nil, err
	}

	c.storeCached(hash, sessionKey, sessionOverlay, cached)
	slog.Info("processor_output runner inject",
		"pixel_values", len(img.ProcessorPixelValues),
		"grid_thw", img.GridTHW,
		"engine", "ollama",
	)
	return restoreMultimodal(ctx, cached), nil
}

func (c *VisionEmbedCache) lookupCached(
	ctx ml.Context,
	hash uint64,
	sessionKey string,
	sessionOverlay bool,
	kind string,
) ([]input.Multimodal, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if sessionOverlay && sessionKey != "" {
		if cached, ok := c.findSessionEmbedLocked(sessionKey, hash); ok {
			return restoreMultimodal(ctx, cached), true
		}
	}
	if cached, err := c.findGlobalLocked(hash); err == nil {
		if sessionOverlay && sessionKey != "" {
			c.storeSessionEmbedLocked(sessionKey, hash, cached)
		}
		slog.Info(kind+" global cache hit", "engine", "ollama")
		return restoreMultimodal(ctx, cached), true
	}
	return nil, false
}

func (c *VisionEmbedCache) storeCached(hash uint64, sessionKey string, sessionOverlay bool, cached cachedMultimodal) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.addGlobalLocked(hash, cached)
	if sessionOverlay && sessionKey != "" {
		c.storeSessionEmbedLocked(sessionKey, hash, cached)
	}
}
