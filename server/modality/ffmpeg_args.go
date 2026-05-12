package modality

import (
	"fmt"

	"github.com/ollama/ollama/types/model"
)

// BuildFFmpegVideoFilter returns the -vf argument for frame sampling (fps vs stride).
//
// Why one builder: keeps mode-specific ffmpeg math in one place and allows argv-only tests
// without installing ffmpeg. Stride uses input-frame select + setpts so output timestamps stay
// sane when the filter drops frames.
func BuildFFmpegVideoFilter(policy VideoSamplingPolicy) string {
	if policy.Mode == model.VideoSampleModeStride {
		n := policy.Stride
		if n < 1 {
			n = 1
		}
		// Every Nth decoded frame; setpts keeps timestamps sane for variable output rate.
		return fmt.Sprintf("select='not(mod(n\\,%d))',setpts=N/FRAME_RATE/TB", n)
	}
	fps := policy.FPS
	if fps <= 0 {
		fps = 1
	}
	return fmt.Sprintf("fps=%g", fps)
}
