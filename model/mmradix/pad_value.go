// Package mmradix implements SGLang-style multimodal pad_value sentinels for KV prefix matching.
//
// WHY: vision soft/pad token IDs are identical across different images, so a token-id-only
// radix / input-cache key cannot tell two clips apart. SGLang replaces each image's pad run
// with pad_value = 1_000_000 + (content_hash % 2^30). Zerollama already stores MultimodalHash
// on the first vision slot; this package also stamps Token (and trailing pads) so Token-only
// and MultimodalHash paths agree.
//
// Embeddings must clamp pad_values into [0, vocab_size) before TokenEmbedding — vision
// tensors overwrite those positions in Forward.
package mmradix

import (
	"github.com/ollama/ollama/model/input"
)

// MMPadShiftValue matches SGLang schedule_batch.MM_PAD_SHIFT_VALUE.
const MMPadShiftValue = 1_000_000

// PadValueFromHash maps a content hash to a sentinel token id above real vocab.
func PadValueFromHash(hash uint64) int32 {
	return int32(MMPadShiftValue + (hash % (1 << 30)))
}

// IsPadValue reports whether id is in the multimodal sentinel band.
func IsPadValue(id int32) bool {
	return id >= MMPadShiftValue
}

// ClampForEmbed maps a possibly-sentinel token into the embedding vocab range.
// Matches SGLang input_ids.clamp_(0, vocab_size-1) before embed_tokens.
func ClampForEmbed(id int32, vocabSize int32) int32 {
	if vocabSize <= 0 {
		return id
	}
	if id < 0 {
		return 0
	}
	if id >= vocabSize {
		return vocabSize - 1
	}
	return id
}

// ApplyToInputs rewrites vision pad Token ids to content-hash pad_values.
//
// For each input with MultimodalHash != 0, sets Token = PadValueFromHash(hash) and
// stamps the following SameBatch-1 text-only siblings (PostTokenize expansion) with the
// same Token and MultimodalHash so the full vision span is content-addressed.
func ApplyToInputs(inputs []*input.Input) int {
	if len(inputs) == 0 {
		return 0
	}
	n := 0
	for i := 0; i < len(inputs); i++ {
		inp := inputs[i]
		if inp == nil || inp.MultimodalHash == 0 {
			continue
		}
		pad := PadValueFromHash(inp.MultimodalHash)
		if inp.Token != pad {
			inp.Token = pad
			n++
		}
		span := inp.SameBatch
		if span <= 0 {
			span = 1
		}
		for j := 1; j < span && i+j < len(inputs); j++ {
			sib := inputs[i+j]
			if sib == nil {
				continue
			}
			// Trailing pads from PostTokenize: no Multimodal payload, hash unset.
			if len(sib.Multimodal) > 0 {
				break
			}
			if sib.MultimodalHash != 0 && sib.MultimodalHash != inp.MultimodalHash {
				break
			}
			sib.Token = pad
			sib.MultimodalHash = inp.MultimodalHash
			n++
		}
	}
	return n
}
