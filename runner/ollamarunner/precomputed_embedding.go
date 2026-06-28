// ollama-engine ingest for SGLang precomputed_embedding and processor_output on padded inject paths.
//
// WHY hashes on inject: input-cache and session ViT overlay key embeds by content;
// precomputed rows and pixel tensors must participate in the same dedup story as PNG bytes.
package ollamarunner

import (
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"log/slog"
	"math"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/ml"
)

func (s *Server) multimodalFromPrecomputed(ctx ml.Context, img llm.ImageData) ([]input.Multimodal, error) {
	ingest, ok := s.model.(model.PrecomputedMultimodalIngest)
	if !ok {
		return nil, fmt.Errorf("precomputed_embedding is not supported for this model on ollama-engine")
	}
	return ingest.MultimodalFromPrecomputed(ctx, img.PrecomputedFeature, img.GridTHW)
}

func (s *Server) multimodalFromProcessorOutput(ctx ml.Context, img llm.ImageData) ([]input.Multimodal, error) {
	ingest, ok := s.model.(model.ProcessorOutputMultimodalIngest)
	if !ok {
		return nil, fmt.Errorf("processor_output is not supported for this model on ollama-engine")
	}
	return ingest.MultimodalFromProcessorOutput(ctx, img.ProcessorPixelValues, img.GridTHW)
}

func hashPrecomputedFeature(rows [][]float32) uint64 {
	var h maphash.Hash
	var buf [4]byte
	for _, row := range rows {
		for _, v := range row {
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			_, _ = h.Write(buf[:])
		}
	}
	return h.Sum64()
}

func hashProcessorPixelValues(pixelValues []float32) uint64 {
	var h maphash.Hash
	var buf [4]byte
	for _, v := range pixelValues {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func (s *Server) appendPaddedPrecomputedImage(
	raw *[]*input.Input,
	mmStore *multimodalStore,
	ctxs *[]ml.Context,
	img llm.ImageData,
	consume string,
	sessionKey string,
	sessionOverlay bool,
) error {
	ctx := s.model.Backend().NewContext()
	ingest, ok := s.model.(model.PrecomputedMultimodalIngest)
	if !ok {
		ctx.Close()
		return fmt.Errorf("precomputed_embedding is not supported for this model on ollama-engine")
	}
	var mm []input.Multimodal
	var err error
	if s.visionCache != nil {
		mm, err = s.visionCache.GetOrEncodePrecomputed(ingest, s.model.Backend(), ctx, img, sessionKey, sessionOverlay)
	} else {
		mm, err = ingest.MultimodalFromPrecomputed(ctx, img.PrecomputedFeature, img.GridTHW)
		if err == nil {
			slog.Info("precomputed_embedding runner inject",
				"image", img.ID,
				"rows", len(img.PrecomputedFeature),
				"engine", "ollama",
				"consume", consume,
			)
		}
	}
	if err != nil {
		ctx.Close()
		return err
	}
	hash := hashPrecomputedFeature(img.PrecomputedFeature)
	*raw = append(*raw, &input.Input{Multimodal: mm, MultimodalHash: hash})
	mmStore.addMultimodal(mm)
	*ctxs = append(*ctxs, ctx)
	return nil
}

func (s *Server) appendPaddedProcessorOutputImage(
	raw *[]*input.Input,
	mmStore *multimodalStore,
	ctxs *[]ml.Context,
	img llm.ImageData,
	consume string,
	sessionKey string,
	sessionOverlay bool,
) error {
	ctx := s.model.Backend().NewContext()
	ingest, ok := s.model.(model.ProcessorOutputMultimodalIngest)
	if !ok {
		ctx.Close()
		return fmt.Errorf("processor_output is not supported for this model on ollama-engine")
	}
	var mm []input.Multimodal
	var err error
	if s.visionCache != nil {
		mm, err = s.visionCache.GetOrEncodeProcessorOutput(ingest, s.model.Backend(), ctx, img, sessionKey, sessionOverlay)
	} else {
		mm, err = ingest.MultimodalFromProcessorOutput(ctx, img.ProcessorPixelValues, img.GridTHW)
		if err == nil {
			slog.Info("processor_output runner inject",
				"image", img.ID,
				"pixel_values", len(img.ProcessorPixelValues),
				"grid_thw", img.GridTHW,
				"engine", "ollama",
				"consume", consume,
			)
		}
	}
	if err != nil {
		ctx.Close()
		return err
	}
	hash := hashProcessorPixelValues(img.ProcessorPixelValues)
	*raw = append(*raw, &input.Input{Multimodal: mm, MultimodalHash: hash})
	mmStore.addMultimodal(mm)
	*ctxs = append(*ctxs, ctx)
	return nil
}
