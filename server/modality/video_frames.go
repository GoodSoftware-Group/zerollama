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
	sessionKey := ExtractPromptCacheKey(req.Options)
	lastUser := lastUserMessageIndex(req.Messages)
	for i := range req.Messages {
		if len(req.Messages[i].Videos) == 0 {
			if err := validatePreexpandedVideoMessage(req.Messages[i]); err != nil {
				return err
			}
			maybeRestorePreprocessedLayout(sessionKey, i, lastUser, &req.Messages[i])
			continue
		}
		maxV := policy.MaxVideosPerMessage
		if len(req.Messages[i].Videos) > maxV {
			return fmt.Errorf("too many videos in one message (max %d)", maxV)
		}
		var spans []api.VideoSpan
		singleClip := len(req.Messages[i].Videos) == 1
		paddedForClip := req.Messages[i].PaddedInputIDs
		if !singleClip {
			paddedForClip = nil // layout cache is per single-clip agent messages only
		}
		var cachedPadded []int
		for _, vid := range req.Messages[i].Videos {
			sample, err := sampleVideoToPNGs(ctx, policy, sessionKey, vid, paddedForClip)
			if err != nil {
				return err
			}
			if len(sample.paddedInputIDs) > 0 {
				cachedPadded = sample.paddedInputIDs
			}
			spans = append(spans, videoSpanFromExpand(sample.frames, sample.gridTHW, policy))
			for _, f := range sample.frames {
				req.Messages[i].Images = append(req.Messages[i].Images, f)
			}
		}
		req.Messages[i].Videos = nil
		req.Messages[i].VideoSpans = spans
		if singleClip && sessionKey != "" && len(req.Messages[i].PaddedInputIDs) == 0 && len(cachedPadded) > 0 {
			req.Messages[i].PaddedInputIDs = append([]int(nil), cachedPadded...)
			slog.Info("video layout session cache hit",
				"session_key", sessionKey,
				"padded_input_ids_len", len(cachedPadded),
			)
		}
		if len(req.Messages[i].Images) > policy.MaxImagesAfterExpand {
			return fmt.Errorf("too many images after video expansion (max %d)", policy.MaxImagesAfterExpand)
		}
	}
	return nil
}

func validatePreexpandedVideoMessage(msg api.Message) error {
	// SGLang-style fast path: client already expanded video into images + spans.
	// Why validate: inconsistent spans would mis-render or blow num_ctx silently.
	if len(msg.VideoSpans) == 0 {
		return nil
	}
	frames := 0
	for _, sp := range msg.VideoSpans {
		if err := validateVideoSpanGridTHW(sp); err != nil {
			return err
		}
		frames += sp.FrameCount
	}
	if err := validatePaddedInputIDs(msg.PaddedInputIDs); err != nil {
		return err
	}
	if frames > len(msg.Images) {
		return fmt.Errorf("video_spans claim %d frames but message has %d images", frames, len(msg.Images))
	}
	return nil
}

func sampleVideoToPNGs(ctx context.Context, policy VideoSamplingPolicy, sessionKey string, data []byte, paddedInputIDs []int) (videoExpandEntry, error) {
	var empty videoExpandEntry
	if err := validatePaddedInputIDs(paddedInputIDs); err != nil {
		return empty, err
	}
	// Enforce empty/size invariants before ExternalVideoDecodeHook so custom decoders match ffmpeg’s contract.
	if len(data) == 0 {
		return empty, errors.New("empty video data")
	}
	if int64(len(data)) > policy.MaxBytes {
		return empty, fmt.Errorf("video exceeds max size (%d bytes)", policy.MaxBytes)
	}
	// Lookup order: session (agent thread) → global (any client) → ffmpeg.
	// Why session first: global LRU may have evicted this clip under fleet load.
	if entry, ok := lookupSessionVideoExpand(sessionKey, policy, data); ok {
		slog.Info("video sample session cache hit",
			"session_key", sessionKey,
			"mode", policy.Mode,
			"frame_count", len(entry.frames),
		)
		return entry, nil
	}
	if entry, ok := lookupVideoExpandCache(policy, data); ok {
		slog.Info("video sample global cache hit",
			"session_key", sessionKey,
			"mode", policy.Mode,
			"frame_count", len(entry.frames),
		)
		// Promote frames+grid to session cache but do not clobber any layout
		// the session already has for this clip (paddedInputIDs is session-specific).
		rememberSessionVideoExpand(sessionKey, policy, data, entry.frames, entry.gridTHW, nil)
		if ids, ok := lookupSessionVideoLayout(sessionKey, policy, data); ok {
			entry.paddedInputIDs = ids
		}
		return entry, nil
	}
	if ExternalVideoDecodeHook != nil {
		frames, err := ExternalVideoDecodeHook(ctx, policy, data)
		if err != nil {
			return empty, err
		}
		grid := computeVideoGridTHWFromFrames(frames, policy)
		rememberVideoExpandCache(policy, data, frames, grid)
		rememberSessionVideoExpand(sessionKey, policy, data, frames, grid, paddedInputIDs)
		return videoExpandEntry{frames: frames, gridTHW: grid, paddedInputIDs: cloneIntSlice(paddedInputIDs)}, nil
	}
	tmp, err := os.CreateTemp("", "ollama-vid-*."+sniffVideoExt(data))
	if err != nil {
		return empty, err
	}
	path := tmp.Name()
	defer os.Remove(path)

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return empty, err
	}
	if err := tmp.Close(); err != nil {
		return empty, err
	}

	outDir, err := os.MkdirTemp("", "ollama-vframes-*")
	if err != nil {
		return empty, err
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
			return empty, fmt.Errorf("ffmpeg: %w: %s", err, msg)
		}
		return empty, fmt.Errorf("ffmpeg failed: %w (is ffmpeg installed and on PATH?)", err)
	}

	entries, err := os.ReadDir(outDir)
	if err != nil {
		return empty, err
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".png") {
			continue
		}
		names = append(names, e.Name())
	}
	if len(names) == 0 {
		return empty, errors.New("ffmpeg produced no frames (unsupported or empty video?)")
	}
	sort.Strings(names)

	var out []api.ImageData
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			return empty, err
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
	grid := computeVideoGridTHWFromFrames(out, policy)
	rememberVideoExpandCache(policy, data, out, grid)
	rememberSessionVideoExpand(sessionKey, policy, data, out, grid, paddedInputIDs)
	return videoExpandEntry{frames: out, gridTHW: grid, paddedInputIDs: cloneIntSlice(paddedInputIDs)}, nil
}

func cloneIntSlice(in []int) []int {
	if len(in) == 0 {
		return nil
	}
	return append([]int(nil), in...)
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
