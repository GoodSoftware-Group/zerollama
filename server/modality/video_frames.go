package modality

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// ExpandVideosInChatRequest turns each raw video blob into sampled PNG frames and appends them to
// Images in order, sets VideoSpans for each clip, then clears Videos. Run before prompt/render.
//
// Why VideoSpans: the runner still sees a flat image list; spans record which images came from
// which clip so renderers can group placeholders when a model cares (see docs/video-understanding.md).
func ExpandVideosInChatRequest(ctx context.Context, policy VideoSamplingPolicy, req *api.ChatRequest) error {
	if req == nil {
		return nil
	}
	for i := range req.Messages {
		if len(req.Messages[i].Videos) == 0 {
			continue
		}
		maxV := policy.MaxVideosPerMessage
		if len(req.Messages[i].Videos) > maxV {
			return fmt.Errorf("too many videos in one message (max %d)", maxV)
		}
		var spans []api.VideoSpan
		for _, vid := range req.Messages[i].Videos {
			frames, err := sampleVideoToPNGs(ctx, policy, vid)
			if err != nil {
				return err
			}
			spans = append(spans, api.VideoSpan{FrameCount: len(frames)})
			for _, f := range frames {
				req.Messages[i].Images = append(req.Messages[i].Images, f)
			}
		}
		req.Messages[i].Videos = nil
		req.Messages[i].VideoSpans = spans
		if len(req.Messages[i].Images) > policy.MaxImagesAfterExpand {
			return fmt.Errorf("too many images after video expansion (max %d)", policy.MaxImagesAfterExpand)
		}
	}
	return nil
}

func sampleVideoToPNGs(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
	// Enforce empty/size invariants before ExternalVideoDecodeHook so custom decoders match ffmpeg’s contract.
	if len(data) == 0 {
		return nil, errors.New("empty video data")
	}
	if int64(len(data)) > policy.MaxBytes {
		return nil, fmt.Errorf("video exceeds max size (%d bytes)", policy.MaxBytes)
	}
	if ExternalVideoDecodeHook != nil {
		return ExternalVideoDecodeHook(ctx, policy, data)
	}
	tmp, err := os.CreateTemp("", "ollama-vid-*."+sniffVideoExt(data))
	if err != nil {
		return nil, err
	}
	path := tmp.Name()
	defer os.Remove(path)

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return nil, err
	}
	if err := tmp.Close(); err != nil {
		return nil, err
	}

	outDir, err := os.MkdirTemp("", "ollama-vframes-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(outDir)

	ctx, cancel := context.WithTimeout(ctx, policy.FFmpegTimeout)
	defer cancel()

	ffmpeg := envconfig.FFmpegBin()
	vf := BuildFFmpegVideoFilter(policy)
	maxFrames := policy.MaxFrames
	outPattern := filepath.Join(outDir, "frame_%04d.png")

	args := []string{
		"-hide_banner", "-loglevel", "error", "-y",
		"-i", path,
		"-vf", vf,
		"-frames:v", strconv.Itoa(maxFrames),
	}
	if policy.Mode == model.VideoSampleModeStride {
		args = append(args, "-vsync", "vfr")
	}
	args = append(args, outPattern)
	cmd := exec.CommandContext(ctx, ffmpeg, args...)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg != "" {
			return nil, fmt.Errorf("ffmpeg: %w: %s", err, msg)
		}
		return nil, fmt.Errorf("ffmpeg failed: %w (is ffmpeg installed and on PATH?)", err)
	}

	entries, err := os.ReadDir(outDir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".png") {
			continue
		}
		names = append(names, e.Name())
	}
	if len(names) == 0 {
		return nil, errors.New("ffmpeg produced no frames (unsupported or empty video?)")
	}
	sort.Strings(names)

	var out []api.ImageData
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			return nil, err
		}
		out = append(out, b)
	}

	// Info (not Debug): operators need to see effective sampling after deploy without enabling debug.
	slog.Info("video sample",
		"mode", policy.Mode,
		"fps", policy.FPS,
		"stride", policy.Stride,
		"max_frames", policy.MaxFrames,
		"frame_count", len(out),
		"manifest_override", policy.ManifestOverride,
	)
	return out, nil
}

func sniffVideoExt(data []byte) string {
	if len(data) >= 12 && string(data[4:8]) == "ftyp" {
		return "mp4"
	}
	if len(data) >= 4 && data[0] == 0x1a && data[1] == 0x45 && data[2] == 0xdf && data[3] == 0xa3 {
		return "webm"
	}
	if len(data) >= 4 && string(data[0:4]) == "OggS" {
		return "ogg"
	}
	return "mp4"
}
