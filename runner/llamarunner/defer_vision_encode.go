// Defer mtmd/ViT encode until after input-cache lookup (vLLM #52041 analog).
//
// WHY: NewSequence used to run MultimodalTokenize for every image before
// LoadCacheSlot discovered a prefix hit. Stubs sized from GridTHW let prefix
// matching proceed; only the uncached tail is encoded.
package llamarunner

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/llm"
)

type deferredVisionImage struct {
	hash uint64
	img  llm.ImageData
}

func isVisionStub(in input) bool {
	return in.embedHash != 0 && len(in.embed) == 0 && in.token == 0
}

func stubVisionInputs(n int, hash uint64) []input {
	if n <= 0 {
		return nil
	}
	out := make([]input, n)
	for i := range out {
		out[i] = input{embedHash: hash}
	}
	return out
}

func stubVisionChunks(n int, hash uint64) []visionChunk {
	if n <= 0 {
		return nil
	}
	out := make([]visionChunk, n)
	for i := range out {
		out[i] = visionChunk{hash: hash}
	}
	return out
}

func estimateVisionTokenSpan(img llm.ImageData) int {
	if n := visionTokensFromGridTHW(img.GridTHW, defaultVisionSpatialMerge); n > 0 {
		return n
	}
	return 0
}

func (s *Server) imageChunksMaybeDeferred(
	images []llm.ImageData,
	sessionKey string,
	sessionOverlay bool,
	deferEncode bool,
) ([][]visionChunk, []deferredVisionImage, error) {
	var imageChunks [][]visionChunk
	var deferred []deferredVisionImage
	var stats visionGridHintStats
	for _, img := range images {
		hash := s.imageContentHash(img)
		if deferEncode {
			if span := estimateVisionTokenSpan(img); span > 0 {
				deferred = append(deferred, deferredVisionImage{hash: hash, img: img})
				imageChunks = append(imageChunks, stubVisionChunks(span, hash))
				continue
			}
		}
		vc, err := s.encodeOneImage(img, sessionKey, sessionOverlay)
		if err != nil {
			return nil, nil, err
		}
		st := logVisionGridHint(img.ID, img.GridTHW, vc)
		stats.Hinted += st.Hinted
		stats.Matched += st.Matched
		stats.Mismatched += st.Mismatched
		imageChunks = append(imageChunks, vc)
	}
	logVisionGridHintSummary(stats)
	return imageChunks, deferred, nil
}

func (s *Server) hydrateDeferredVision(
	seq *Sequence,
	tail []input,
	numCached int,
	sessionKey string,
	sessionOverlay bool,
) ([]input, error) {
	if seq == nil || len(seq.deferredImages) == 0 {
		return tail, nil
	}
	byHash := make(map[uint64]llm.ImageData, len(seq.deferredImages))
	for _, d := range seq.deferredImages {
		byHash[d.hash] = d.img
	}

	var out []input
	hydrated := 0
	for i := 0; i < len(tail); {
		if !isVisionStub(tail[i]) {
			out = append(out, tail[i])
			i++
			continue
		}
		hash := tail[i].embedHash
		j := i + 1
		for j < len(tail) && isVisionStub(tail[j]) && tail[j].embedHash == hash {
			j++
		}
		img, ok := byHash[hash]
		if !ok {
			return nil, fmt.Errorf("deferred vision: missing image for hash %x", hash)
		}
		vc, err := s.encodeOneImage(img, sessionKey, sessionOverlay)
		if err != nil {
			return nil, err
		}
		st := logVisionGridHint(img.ID, img.GridTHW, vc)
		logVisionGridHintSummary(st)
		for _, c := range vc {
			out = appendVisionChunk(out, c)
		}
		hydrated++
		i = j
	}
	if hydrated > 0 {
		slog.Info("deferred vision encode after input-cache lookup",
			"hydrated", hydrated,
			"cached_prefix_inputs", numCached,
			"engine", "llama",
		)
	}
	return out, nil
}
