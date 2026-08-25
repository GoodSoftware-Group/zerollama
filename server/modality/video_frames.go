package modality

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// ExpandVideosInChatRequest turns each raw video blob into sampled PNG frames and appends them to
// Images in order, sets VideoSpans for each clip, then clears Videos. Run before prompt/render.
//
// Why VideoSpans: the runner still sees a flat image list; spans record which images came from
// which clip so renderers can group placeholders when a model cares (see docs/video-understanding.md).
//
// Multi-clip turns sample in parallel up to OLLAMA_MM_IO_WORKERS (SGLang #31438); span/image order
// is preserved.
func ExpandVideosInChatRequest(ctx context.Context, policy VideoSamplingPolicy, req *api.ChatRequest) error {
	if req == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	sessionKey := ExtractPromptCacheKey(req.Options)
	lastUser := lastUserMessageIndex(req.Messages)
	for i := range req.Messages {
		if len(req.Messages[i].Videos) == 0 {
			if err := validatePreexpandedVideoMessage(&req.Messages[i]); err != nil {
				return err
			}
			maybeRestorePreprocessedLayout(sessionKey, i, lastUser, &req.Messages[i])
			continue
		}
		maxV := policy.MaxVideosPerMessage
		if len(req.Messages[i].Videos) > maxV {
			return ClientMediaf("too many videos in one message (max %d)", maxV)
		}
		videos := req.Messages[i].Videos
		singleClip := len(videos) == 1
		paddedForClip := req.Messages[i].PaddedInputIDs
		if !singleClip {
			paddedForClip = nil // layout cache is per single-clip agent messages only
		}

		samples, err := sampleVideosParallel(ctx, policy, sessionKey, videos, paddedForClip)
		if err != nil {
			return err
		}

		var spans []api.VideoSpan
		var cachedPadded []int
		for _, sample := range samples {
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
			return ClientMediaf("too many images after video expansion (max %d)", policy.MaxImagesAfterExpand)
		}
	}
	return nil
}

func sampleVideosParallel(ctx context.Context, policy VideoSamplingPolicy, sessionKey string, videos []api.VideoData, paddedForClip []int) ([]videoExpandEntry, error) {
	n := len(videos)
	if n == 0 {
		return nil, nil
	}
	out := make([]videoExpandEntry, n)
	if n == 1 {
		sample, err := sampleVideoToPNGs(ctx, policy, sessionKey, videos[0], paddedForClip)
		if err != nil {
			return nil, err
		}
		out[0] = sample
		return out, nil
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(envconfig.MMIOWorkers())
	for i := range videos {
		i, vid := i, videos[i]
		g.Go(func() error {
			sample, err := sampleVideoToPNGs(gctx, policy, sessionKey, vid, paddedForClip)
			if err != nil {
				return err
			}
			out[i] = sample
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}

func validatePreexpandedVideoMessage(msg *api.Message) error {
	// SGLang-style fast path: client already expanded video into images + spans.
	// Why validate: inconsistent spans would mis-render or blow num_ctx silently.
	if msg == nil || len(msg.VideoSpans) == 0 {
		return nil
	}
	frames := 0
	for i := range msg.VideoSpans {
		if err := validateVideoSpanGridTHW(msg.VideoSpans[i]); err != nil {
			return WrapClientMedia(err, "invalid video_spans")
		}
		if len(msg.VideoSpans[i].GridTHW) == 3 {
			// Client sent grid_thw with pre-expanded frames — safe to forward to mtmd.
			msg.VideoSpans[i].GridTHWExplicit = true
		}
		frames += msg.VideoSpans[i].FrameCount
	}
	if err := validatePaddedInputIDs(msg.PaddedInputIDs); err != nil {
		return WrapClientMedia(err, "invalid padded_input_ids")
	}
	if frames > len(msg.Images) {
		return ClientMediaf("video_spans claim %d frames but message has %d images", frames, len(msg.Images))
	}
	// SGLang #31957: reject empty/zero-frame spans and one-sided mismatches instead of
	// silently truncating with min(...) downstream (Moss-VL metadata vs tokens).
	if frames == 0 {
		return ClientMedia("video_spans present but total frame_count is 0")
	}
	for i, sp := range msg.VideoSpans {
		if sp.FrameCount <= 0 {
			return ClientMediaf("video_spans[%d] frame_count must be positive, got %d", i, sp.FrameCount)
		}
	}
	return nil
}

func sampleVideoToPNGs(ctx context.Context, policy VideoSamplingPolicy, sessionKey string, data []byte, paddedInputIDs []int) (videoExpandEntry, error) {
	var empty videoExpandEntry
	if err := validatePaddedInputIDs(paddedInputIDs); err != nil {
		return empty, WrapClientMedia(err, "invalid padded_input_ids")
	}
	// Enforce empty/size invariants before ExternalVideoDecodeHook so custom decoders match ffmpeg’s contract.
	if len(data) == 0 {
		return empty, ClientMedia("empty video data")
	}
	if int64(len(data)) > policy.MaxBytes {
		return empty, ClientMediaf("video exceeds max size (%d bytes)", policy.MaxBytes)
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
			return empty, WrapClientMedia(err, "video decode")
		}
		grid := computeVideoGridTHWFromFrames(frames, policy)
		rememberVideoExpandCache(policy, data, frames, grid)
		rememberSessionVideoExpand(sessionKey, policy, data, frames, grid, paddedInputIDs)
		return videoExpandEntry{frames: frames, gridTHW: grid, paddedInputIDs: cloneIntSlice(paddedInputIDs)}, nil
	}

	ffmpeg := envconfig.FFmpegBin()
	if _, err := exec.LookPath(ffmpeg); err != nil {
		return empty, WrapServerMedia(err, "ffmpeg not found for video sampling (is ffmpeg installed and on PATH?)")
	}

	tmp, err := os.CreateTemp("", "ollama-vid-*."+sniffVideoExt(data))
	if err != nil {
		return empty, WrapServerMedia(err, "temp video file")
	}
	path := tmp.Name()
	defer os.Remove(path)

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return empty, WrapServerMedia(err, "write temp video")
	}
	if err := tmp.Close(); err != nil {
		return empty, WrapServerMedia(err, "close temp video")
	}

	outDir, err := os.MkdirTemp("", "ollama-vframes-*")
	if err != nil {
		return empty, WrapServerMedia(err, "temp frame dir")
	}
	defer os.RemoveAll(outDir)

	ctx, cancel := context.WithTimeout(ctx, policy.FFmpegTimeout)
	defer cancel()

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
		if isFFmpegMissingError(err) {
			return empty, WrapServerMedia(err, "ffmpeg failed (is ffmpeg installed and on PATH?)")
		}
		if msg != "" {
			return empty, WrapClientMedia(err, "ffmpeg: "+msg)
		}
		return empty, WrapClientMedia(err, "ffmpeg failed to decode video")
	}

	entries, err := os.ReadDir(outDir)
	if err != nil {
		return empty, WrapServerMedia(err, "read frame dir")
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".png") {
			continue
		}
		names = append(names, e.Name())
	}
	if len(names) == 0 {
		return empty, ClientMedia("ffmpeg produced no frames (unsupported or empty video?)")
	}
	sort.Strings(names)

	var out []api.ImageData
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			return empty, WrapServerMedia(err, "read sampled frame")
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

func isFFmpegMissingError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, exec.ErrNotFound) {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "executable file not found") ||
		strings.Contains(s, "no such file or directory")
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
