package llamarunner

// visionChunksFromPrecomputed maps SGLang precomputed_embedding rows to mtmd-style embed chunks.
func visionChunksFromPrecomputed(rows [][]float32) []visionChunk {
	var vc []visionChunk
	for _, row := range rows {
		if len(row) == 0 {
			continue
		}
		vc = append(vc, visionChunk{embed: append([]float32(nil), row...)})
	}
	return vc
}
