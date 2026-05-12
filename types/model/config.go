package model

// Native video frame sampling modes (manifest video_sampling.mode and env OLLAMA_VIDEO_SAMPLE_MODE).
const (
	VideoSampleModeFPS    = "fps"
	VideoSampleModeStride = "stride"
)

// VideoSampling holds optional per-model overrides for native ffmpeg frame sampling.
// Zero values mean “use the merged server env defaults” after ResolveVideoPolicy merges env + manifest.
//
// Why JSON on ConfigV2: model authors tune sampling per published artifact (evals, paper recipe)
// without forcing every deployment to set env vars; env remains the global safety net.
type VideoSampling struct {
	// Mode is VideoSampleModeFPS (time-uniform fps filter) or VideoSampleModeStride (every Nth frame).
	Mode string `json:"mode,omitempty"`
	// FPS is the ffmpeg fps filter rate when Mode is fps; set > 0 to override env.
	FPS float64 `json:"fps,omitempty"`
	// Stride is “every Nth frame” when Mode is stride (N >= 1).
	Stride int `json:"stride,omitempty"`
	// MaxFrames caps sampled frames per video; set > 0 to override env.
	MaxFrames int `json:"max_frames,omitempty"`
}

// ConfigV2 represents the configuration metadata for a model.
type ConfigV2 struct {
	ModelFormat   string   `json:"model_format"`
	ModelFamily   string   `json:"model_family"`
	ModelFamilies []string `json:"model_families"`
	ModelType     string   `json:"model_type"` // shown as Parameter Size
	FileType      string   `json:"file_type"`  // shown as Quantization Level
	Renderer      string   `json:"renderer,omitempty"`
	Parser        string   `json:"parser,omitempty"`
	Requires      string   `json:"requires,omitempty"`

	RemoteHost  string `json:"remote_host,omitempty"`
	RemoteModel string `json:"remote_model,omitempty"`

	// used for remotes
	Capabilities []string `json:"capabilities,omitempty"`
	ContextLen   int      `json:"context_length,omitempty"`
	EmbedLen     int      `json:"embedding_length,omitempty"`
	BaseName     string   `json:"base_name,omitempty"`

	// required by spec
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`

	// ModalityBackends selects which subprocess or built-in driver handles each modality.
	// Keys (see model.Modality* constants): "image", "speech" (TTS), "transcribe" (STT),
	// "video_understanding" (VLM: "native" default, or "sglang" with OLLAMA_SGLANG_URL).
	// Empty or omitted value means the default built-in path for that modality.
	ModalityBackends map[string]string `json:"modality_backends,omitempty"`
	// BackendPaths passes filesystem paths to subprocess adapters (e.g. Whisper GGML, Piper ONNX).
	// Keys include "whisper_model", "piper_model" (and optionally "piper_config").
	BackendPaths map[string]string `json:"backend_paths,omitempty"`

	// VideoSampling overrides native ffmpeg sampling for video_understanding=native (see docs/video-parity.md).
	VideoSampling *VideoSampling `json:"video_sampling,omitempty"`
	// TokensPerImage is an optional vision-token budget per raster frame for context preflight only.
	// Default 768 matches server/prompt.go until projector metadata supplies a real per-image cost.
	TokensPerImage int `json:"tokens_per_image,omitempty"`
}

// RootFS represents the root filesystem configuration for a model.
type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}
