package model

// Keys for [ConfigV2.ModalityBackends] and [ConfigV2.BackendPaths].
const (
	ModalityImage      = "image"
	ModalitySpeech     = "speech"
	ModalityTranscribe = "transcribe"
	// ModalityVideoUnderstanding selects native ffmpeg sampling vs forwarding chat to SGLang when
	// operators already run a separate VLM stack—see docs/video-understanding.md.
	ModalityVideoUnderstanding = "video_understanding"
	// ModalityInference selects the local GPU inference driver (default: built-in ggml runner).
	ModalityInference = "inference"
	// ModalityVideoGeneration selects the text-to-video driver (e.g. Wan via run_script).
	ModalityVideoGeneration = "video_generation"
)

// Backend driver names for [ConfigV2.ModalityBackends].
const (
	BackendMLXImagegen   = "mlx-imagegen"   // default MLX pipeline in Ollama
	BackendWhisper       = "whisper"        // whisper.cpp / compatible CLI
	BackendPiper         = "piper"          // Piper TTS
	BackendExternalImage = "external-image" // user-provided command (see docs)
	BackendOpenVINOImage = "openvino-image" // OpenVINO GenAI Text2ImagePipeline (see docs/sd-openvino-a380.md)
	BackendVideoNative   = "native"         // ffmpeg frame sampling inside Ollama (default when unset)
	BackendSGLang           = "sglang"              // forward OpenAI chat to SGLang HTTP API
	BackendZerollamaRuntime = "zerollama-runtime" // Python GGUF runtime sidecar (see runtime/)
	BackendWan              = "wan"               // Wan2.x via scripts/wan_video_generate.py (see docs/wan-t2v.md)
)
