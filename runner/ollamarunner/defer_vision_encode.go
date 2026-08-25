// Defer ViT encode until after input-cache lookup (vLLM #52041 analog).
//
// WHY: NewSequence used to run GetOrEncode for every image before LoadCacheSlot
// discovered a prefix hit — wasted ViT when KV/input cache already covered those
// spans. We build PostTokenize-ready stubs from GridTHW, match prefix, then
// encode only multimodal rows in the uncached tail.
package ollamarunner

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/ml"
)

type deferredVisionImage struct {
	hash uint64
	img  llm.ImageData
}

var deferVisionStubMarker = &struct{ deferVisionStub bool }{}

func isDeferVisionStub(mm []input.Multimodal) bool {
	return len(mm) == 1 && mm[0].Data == deferVisionStubMarker
}

func stubVisionMultimodal(ctx ml.Context, tokensPerGrid int) []input.Multimodal {
	if tokensPerGrid <= 0 {
		tokensPerGrid = 1
	}
	// PostTokenize only needs Tensor.Dim(1) for SameBatch sizing.
	t := ctx.Input().FromFloats(make([]float32, tokensPerGrid), 1, tokensPerGrid, 1)
	return []input.Multimodal{{Tensor: t, Data: deferVisionStubMarker}}
}

func estimateVisionTokenSpan(img llm.ImageData) int {
	if n := visionTokensFromGridTHW(img.GridTHW, defaultVisionSpatialMerge); n > 0 {
		return n
	}
	return 0
}

func (s *Server) hydrateDeferredVision(
	seq *Sequence,
	tail []*input.Input,
	numCached int,
	sessionKey string,
	sessionOverlay bool,
) error {
	if seq == nil || len(seq.deferredImages) == 0 {
		return nil
	}
	byHash := make(map[uint64]llm.ImageData, len(seq.deferredImages))
	for _, d := range seq.deferredImages {
		byHash[d.hash] = d.img
	}

	hydrated := 0
	for i, inp := range tail {
		if inp == nil || !isDeferVisionStub(inp.Multimodal) {
			continue
		}
		img, ok := byHash[inp.MultimodalHash]
		if !ok {
			return fmt.Errorf("deferred vision: missing image for hash %x", inp.MultimodalHash)
		}
		ctx := s.model.Backend().NewContext()
		var mm []input.Multimodal
		var err error
		switch {
		case img.HasPrecomputedEmbedding():
			if s.visionCache != nil {
				ingest, ok := s.model.(model.PrecomputedMultimodalIngest)
				if !ok {
					ctx.Close()
					return fmt.Errorf("precomputed_embedding unsupported")
				}
				mm, err = s.visionCache.GetOrEncodePrecomputed(ingest, s.model.Backend(), ctx, img, sessionKey, sessionOverlay)
			} else {
				mm, err = s.multimodalFromPrecomputed(ctx, img)
			}
		case img.HasProcessorOutput():
			if s.visionCache != nil {
				ingest, ok := s.model.(model.ProcessorOutputMultimodalIngest)
				if !ok {
					ctx.Close()
					return fmt.Errorf("processor_output unsupported")
				}
				mm, err = s.visionCache.GetOrEncodeProcessorOutput(ingest, s.model.Backend(), ctx, img, sessionKey, sessionOverlay)
			} else {
				mm, err = s.multimodalFromProcessorOutput(ctx, img)
			}
		default:
			mm, err = s.encodeMultimodalCached(ctx, img.Data, img.GridTHW, sessionKey, sessionOverlay)
		}
		if err != nil {
			ctx.Close()
			return err
		}
		seq.mmStore.addMultimodal(mm)
		tail[i].Multimodal = mm
		seq.ctxs = append(seq.ctxs, ctx)
		hydrated++
	}
	if hydrated > 0 {
		slog.Info("deferred vision encode after input-cache lookup",
			"hydrated", hydrated,
			"cached_prefix_inputs", numCached,
			"engine", "ollama",
		)
	}
	return nil
}
