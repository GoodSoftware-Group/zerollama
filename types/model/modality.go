package model

// Keys for [ConfigV2.ModalityBackends] and [ConfigV2.BackendPaths].
const (
	ModalityImage      = "image"
	ModalitySpeech     = "speech"
	ModalityTranscribe = "transcribe"
	// ModalityVideoUnderstanding selects native ffmpeg sampling vs forwarding chat to SGLang when
	// operators already run a separate VLM stack—see docs/video-understanding.md.
	ModalityVideoUnderstanding = "video_understanding"
)

// Backend driver names for [ConfigV2.ModalityBackends].
const (
	BackendMLXImagegen   = "mlx-imagegen"   // default MLX pipeline in Ollama
	BackendWhisper       = "whisper"        // whisper.cpp / compatible CLI
	BackendPiper         = "piper"          // Piper TTS
	BackendExternalImage = "external-image" // user-provided command (see docs)
	BackendVideoNative   = "native"         // ffmpeg frame sampling inside Ollama (default when unset)
	BackendSGLang        = "sglang"         // forward OpenAI chat to SGLang HTTP API
)
