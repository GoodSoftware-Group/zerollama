package modality

// visionGridParams resolves manifest/env defaults for grid_thw layout estimates.
func (p VideoSamplingPolicy) visionPatchSize() int {
	if p.VisionPatchSize > 0 {
		return p.VisionPatchSize
	}
	return qwenVLGridPatchSize
}

func (p VideoSamplingPolicy) visionSpatialMergeSize() int {
	if p.VisionSpatialMergeSize > 0 {
		return p.VisionSpatialMergeSize
	}
	return defaultSpatialMergeSize
}

func (p VideoSamplingPolicy) visionGridFactor() int {
	return p.visionPatchSize() * p.visionSpatialMergeSize()
}
