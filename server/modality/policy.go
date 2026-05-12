package modality

import (
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// VideoSamplingPolicy is the fully resolved native video sampling policy (env merged with manifest).
// It is a plain value type so expansion and tests never depend on a live server.Model (keeps import
// graphs clean and matches the “resolve once at the HTTP boundary” design).
type VideoSamplingPolicy struct {
	Mode      string // model.VideoSampleModeFPS or model.VideoSampleModeStride
	FPS       float64
	Stride    int
	MaxFrames int

	MaxBytes             int64
	FFmpegTimeout        time.Duration
	MaxVideosPerMessage  int
	MaxImagesAfterExpand int

	// TokensPerImage is used for context preflight only (default 768 until projector metadata exists;
	// see server/prompt.go TODO). Manifest may override to match a family’s real vision cost.
	TokensPerImage int

	// ManifestOverride is true when the model manifest influenced sampling or tokens_per_image—used
	// for observability (whether tuning came from env defaults or the published model config).
	ManifestOverride bool
}

// ResolveVideoPolicy merges envconfig defaults with non-zero manifest fields.
//
// Why env first, then manifest: operators set global guardrails; the model author optionally
// aligns sampling with what that checkpoint was trained or evaluated on. Stride without an
// explicit mode implies stride mode so a minimal manifest can set only stride.
//
// Why warn invalid mode: a typo should not silently change behavior; falling back to fps is
// predictable and keeps the server running while surfacing a fixable config error.
func ResolveVideoPolicy(cfg model.ConfigV2) VideoSamplingPolicy {
	p := VideoSamplingPolicy{
		Mode:                 normalizeMode(envconfig.VideoSampleMode()),
		FPS:                  envconfig.VideoSampleFPS(),
		Stride:               envconfig.VideoStride(),
		MaxFrames:            envconfig.VideoMaxFrames(),
		MaxBytes:             envconfig.VideoMaxBytes(),
		FFmpegTimeout:        envconfig.VideoFFmpegTimeout(),
		MaxVideosPerMessage:  envconfig.VideoMaxVideosPerMessage(),
		MaxImagesAfterExpand: envconfig.VideoMaxImagesPerMessage(),
		TokensPerImage:       768,
	}
	if cfg.TokensPerImage > 0 {
		p.TokensPerImage = cfg.TokensPerImage
		p.ManifestOverride = true
	}
	if cfg.VideoSampling == nil {
		return p
	}
	vs := cfg.VideoSampling
	touched := false
	explicitMode := strings.TrimSpace(vs.Mode)
	if explicitMode != "" {
		lm := strings.ToLower(explicitMode)
		switch lm {
		case model.VideoSampleModeFPS:
			p.Mode = model.VideoSampleModeFPS
		case model.VideoSampleModeStride:
			p.Mode = model.VideoSampleModeStride
		default:
			slog.Warn("invalid video_sampling.mode in model manifest, using fps", "mode", explicitMode)
			p.Mode = model.VideoSampleModeFPS
		}
		touched = true
	}
	if vs.FPS > 0 {
		p.FPS = vs.FPS
		touched = true
	}
	if vs.Stride > 0 {
		p.Stride = vs.Stride
		touched = true
		if explicitMode == "" {
			p.Mode = model.VideoSampleModeStride
		}
	}
	if vs.MaxFrames > 0 {
		p.MaxFrames = vs.MaxFrames
		touched = true
	}
	p.ManifestOverride = p.ManifestOverride || touched
	if p.Mode == model.VideoSampleModeStride && p.Stride < 1 {
		p.Stride = 1
	}
	return p
}

func normalizeMode(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case model.VideoSampleModeStride:
		return model.VideoSampleModeStride
	default:
		return model.VideoSampleModeFPS
	}
}
