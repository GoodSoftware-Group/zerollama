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

// VideoGenerationConfig holds per-model defaults for text-to-video (Wan and future runners).
type VideoGenerationConfig struct {
	Runner       string `json:"runner,omitempty"`    // wan-cli; later diffusers | comfy-headless
	Profile      string `json:"profile,omitempty"`   // wan2.1-t2v-1.3b | wan2.2-ti2v-5b
	VRAMTier     string `json:"vram_tier,omitempty"` // 16g | 24g | 32g
	Size         string `json:"size,omitempty"`      // 832x480
	Frames       int    `json:"frames,omitempty"`
	Steps        int    `json:"steps,omitempty"`
	Precision    string `json:"precision,omitempty"` // bf16 | fp16 | fp8
	Quant        string `json:"quant,omitempty"`     // none | gguf | fp8
	BatchSize    int    `json:"batch_size,omitempty"`
	OffloadModel bool   `json:"offload_model,omitempty"`
	T5CPU        bool   `json:"t5_cpu,omitempty"`
	VAETiling    bool   `json:"vae_tiling,omitempty"`
	TimeoutSec   int    `json:"timeout_sec,omitempty"`
}

// ImageGenerationConfig holds per-model defaults for subprocess image backends
// (stable-diffusion.cpp and OpenVINO GenAI).
//
// WHY manifest fields not env-only: Intel Mesa ANV requires diffusion_fa for sd.cpp;
// SD-Turbo uses ~4 steps and cfg≈1; SDXL may need vae_tiling — these vary per tag.
// Server maps these to OLLAMA_SD_* / OLLAMA_OV_* at the subprocess boundary.
type ImageGenerationConfig struct {
	Width       int     `json:"width,omitempty"`
	Height      int     `json:"height,omitempty"`
	Steps       int     `json:"steps,omitempty"`
	CFGScale    float64 `json:"cfg_scale,omitempty"`
	Sampler     string  `json:"sampler,omitempty"`
	DiffusionFA *bool   `json:"diffusion_fa,omitempty"`
	VAEOnCPU    *bool   `json:"vae_on_cpu,omitempty"`
	VAETiling   *bool   `json:"vae_tiling,omitempty"`
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
	// ConcurrencyGroups declare mutual exclusion with other loaded models (LocalAI pattern).
	// Why: on tight GPUs, imagegen + chat must not stay resident together.
	ConcurrencyGroups []string `json:"concurrency_groups,omitempty"`
	Requires          string   `json:"requires,omitempty"`

	RemoteHost  string `json:"remote_host,omitempty"`
	RemoteModel string `json:"remote_model,omitempty"`

	// used for remotes
	Capabilities []string `json:"capabilities,omitempty"`
	ContextLen   int      `json:"context_length,omitempty"`
	EmbedLen     int      `json:"embedding_length,omitempty"`
	BaseName     string   `json:"base_name,omitempty"`
	Draft        *Draft   `json:"draft,omitempty"`

	// required by spec
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`

	// ModalityBackends selects which subprocess or built-in driver handles each modality.
	// Keys (see model.Modality* constants): "image", "speech" (TTS), "transcribe" (STT),
	// "video_understanding" (VLM: "native" default, or "sglang" with OLLAMA_SGLANG_URL),
	// "video_generation" (T2V: "wan" with scripts/wan_video_generate.py).
	// Empty or omitted value means the default built-in path for that modality.
	ModalityBackends map[string]string `json:"modality_backends,omitempty"`
	// BackendPaths passes filesystem paths / URLs to subprocess adapters (e.g. Whisper GGML, Piper ONNX).
	// Keys include "whisper_model", "piper_model", "piper_config", "piper_voice_<name>",
	// "tts_url", "tts_upstream_model", "tts_default_voice", "tts_voices_file", "tts_ref_audio",
	// "wan_repo", "wan_ckpt_dir", "wan_venv", "wan_gguf_path",
	// "sd_cli", "sd_model" (stable-diffusion.cpp binary and GGUF weights),
	// "ov_model_dir", "ov_python", "external_image_bin" (OpenVINO GenAI; see docs/sd-openvino-a380.md),
	// "comfy_workflow_dir" (Comfy template directory; relative paths need OLLAMA_COMFYUI_WORKFLOWS_ROOT
	// or an absolute/~/ path — see docs/comfyui-image-backend.md), "comfy_default_workflow" (name, not a path).
	BackendPaths map[string]string `json:"backend_paths,omitempty"`

	// VideoGeneration presets for models with capability video_gen (see docs/wan-t2v.md).
	VideoGeneration *VideoGenerationConfig `json:"video_generation,omitempty"`

	// ImageGeneration presets for models with capability image and modality_backends.image=external-image
	// or openvino-image (see docs/sd-vulkan-a380.md, docs/sd-openvino-a380.md).
	ImageGeneration *ImageGenerationConfig `json:"image_generation,omitempty"`

	// VideoSampling overrides native ffmpeg sampling for video_understanding=native (see docs/video-parity.md).
	VideoSampling *VideoSampling `json:"video_sampling,omitempty"`
	// TokensPerImage is an optional vision-token budget per raster frame for context preflight only.
	// Default 768 matches server/prompt.go until projector metadata supplies a real per-image cost.
	TokensPerImage int `json:"tokens_per_image,omitempty"`
	// VisionPatchSize / VisionSpatialMergeSize tune native grid_thw estimates (Qwen/SGLang layout).
	VisionPatchSize        int `json:"vision_patch_size,omitempty"`
	VisionSpatialMergeSize int `json:"vision_spatial_merge_size,omitempty"`
}

// Draft describes an auxiliary draft model stored in the same manifest.
type Draft struct {
	ModelFormat  string `json:"model_format,omitempty"`
	Architecture string `json:"architecture,omitempty"`
	TensorPrefix string `json:"tensor_prefix,omitempty"`
	Config       string `json:"config,omitempty"`
}

// RootFS represents the root filesystem configuration for a model.
type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}
