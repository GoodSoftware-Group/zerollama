package llm

// Gemma4PaddedMediaSchedule mirrors server/modality schedule for runner/llama-server inject.
type Gemma4PaddedMediaSchedule struct {
	StillImageCount  int   `json:"still_image_count,omitempty"`
	VideoFrameCounts []int `json:"video_frame_counts,omitempty"`
	AudioClipCount   int   `json:"audio_clip_count,omitempty"`
}
