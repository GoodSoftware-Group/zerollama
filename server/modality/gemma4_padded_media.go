package modality

import "github.com/ollama/ollama/api"

// Gemma4PaddedMediaSchedule describes raster order for Gemma4 padded_input_ids inject.
// Flat images[] order: still rasters, then video frames per clip (VideoSpans order), then audio clips.
type Gemma4PaddedMediaSchedule struct {
	StillImageCount  int   `json:"still_image_count,omitempty"`
	VideoFrameCounts []int `json:"video_frame_counts,omitempty"`
	AudioClipCount   int   `json:"audio_clip_count,omitempty"`
}

// Gemma4PaddedMediaScheduleFromMessage derives media slot counts from a user message.
func Gemma4PaddedMediaScheduleFromMessage(msg api.Message) Gemma4PaddedMediaSchedule {
	videoFrames := 0
	var videoCounts []int
	for _, sp := range msg.VideoSpans {
		videoFrames += sp.FrameCount
		if sp.FrameCount > 0 {
			videoCounts = append(videoCounts, sp.FrameCount)
		}
	}
	still := len(msg.Images) - videoFrames
	if still < 0 {
		still = 0
	}
	return Gemma4PaddedMediaSchedule{
		StillImageCount:  still,
		VideoFrameCounts: videoCounts,
		AudioClipCount:   len(msg.AudioClips),
	}
}

// Gemma4PaddedMediaScheduleForChat returns schedule for the latest user turn (inject target).
func Gemma4PaddedMediaScheduleForChat(msgs []api.Message) Gemma4PaddedMediaSchedule {
	idx := lastUserMessageIndex(msgs)
	if idx < 0 {
		return Gemma4PaddedMediaSchedule{}
	}
	return Gemma4PaddedMediaScheduleFromMessage(msgs[idx])
}
