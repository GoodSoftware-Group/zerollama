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
	BackendMLXImagegen      = "mlx-imagegen"      // default MLX pipeline in Ollama
	BackendWhisper          = "whisper"           // whisper.cpp / compatible CLI
	BackendPiper            = "piper"             // Piper TTS (CPU ONNX)
	BackendMusic3           = "music3"            // MiniMax Music 3 (mlx-audio / later music-cli). Not Piper. Not H3 AudioVAE.
	BackendRemoteTTS        = "remote-tts"        // OpenAI-compatible HTTP TTS (Chatterbox/Orpheus/Kokoro/…)
	BackendExternalImage    = "external-image"    // user-provided command (see docs)
	BackendOpenVINOImage    = "openvino-image"    // OpenVINO GenAI Text2ImagePipeline (see docs/sd-openvino-a380.md)
	BackendVideoNative      = "native"            // ffmpeg frame sampling inside Ollama (default when unset)
	BackendSGLang           = "sglang"            // forward OpenAI chat to SGLang HTTP API
	BackendZerollamaRuntime = "zerollama-runtime" // Python GGUF runtime sidecar (see runtime/)
	BackendWan              = "wan"               // Wan2.x: Python generate.py, or video-cli when backend_paths.video_cli / ZEROLLAMA_VIDEO_CLI is set
	// BackendLTX is Wan2GP LTXV (13B distilled quanto or 2B distilled FP8) via scripts/video/ltx_video_generate.py.
	// See docs/ltx-t2v.md — not LTX-2/Gemma on ≤24 GiB hosts.
	BackendLTX = "ltx"
	// BackendH3 is Darwin video-c MiniMax-H3 (`--family h3 --generate`). Tiny T2VA
	// (5×32²) and 768-canvas T2VA (1 layer); not 50-layer host. See docs/video-c.md.
	BackendH3 = "h3"
	// BackendRIFE is reserved for classical optical-flow inbetweens (not shipped yet).
	// WHY reserve the name now: same /v1/videos + /v1/media contracts as Wan so agents
	// and OpenAPI do not learn a second upload protocol when classical inbetweens ship.
	// See docs/media-uploads.md + docs/wan-t2v.md.
	BackendRIFE = "rife"
	// BackendComfyUI orchestrates a running ComfyUI server for agent-max image utility
	// (edit/img2img/ControlNet/LoRA on Qwen/FLUX/GLM graphs). WHY not mlx-imagegen:
	// porting each HF DiT into x/imagegen costs months; Comfy already packs those
	// workflows. WHY not external-image: that hook has no named workflows or discovery.
	// See docs/comfyui-image-backend.md.
	BackendComfyUI = "comfyui"
)
