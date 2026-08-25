package create

import (
	"strconv"
	"strings"
)

// layerIndex returns the transformer layer index encoded in name, or -1.
func layerIndex(name string) int {
	m := layerIndexRe.FindStringSubmatch(name)
	if m == nil {
		return -1
	}
	idx, err := strconv.Atoi(m[1])
	if err != nil {
		return -1
	}
	return idx
}

// eightBit returns the 8-bit quantization type in base's family.
func eightBit(base string) string {
	if base == "int4" || base == "int8" {
		return "int8"
	}
	return "mxfp8"
}

// promoteEmbedding returns the 8-bit type when the embedding shape fits it.
func promoteEmbedding(shape []int32, base string) string {
	if e := eightBit(base); isAligned(shape, e) {
		return e
	}
	return ""
}

// sensitiveType resolves a quantization-sensitive projection type.
func sensitiveType(promote bool, shape []int32, base string) string {
	if promote {
		if e := eightBit(base); isAligned(shape, e) {
			return e
		}
	}
	if isAligned(shape, base) {
		return base
	}
	return ""
}

func isVision(name string) bool {
	return strings.Contains(name, "vision") ||
		strings.Contains(name, "visual") ||
		strings.HasPrefix(name, "vision_tower.") ||
		strings.HasPrefix(name, "multi_modal_projector.")
}
