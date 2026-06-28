package llamarunner

import (
	"encoding/binary"
	"hash/maphash"
	"math"
)

func hashPrecomputedRows(rows [][]float32) uint64 {
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

func cloneVisionChunks(vc []visionChunk) []visionChunk {
	out := make([]visionChunk, len(vc))
	for i, c := range vc {
		if len(c.embed) != 0 {
			out[i].embed = append([]float32(nil), c.embed...)
		}
		if len(c.tokens) != 0 {
			out[i].tokens = append([]int(nil), c.tokens...)
		}
	}
	return out
}
