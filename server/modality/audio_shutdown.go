package modality

import (
	"github.com/ollama/ollama/server/modality/kokoro"
	"github.com/ollama/ollama/server/modality/neuralvad"
	"github.com/ollama/ollama/server/modality/qwen3asr"
)

// CloseAudioShared stops resident audio sidecars started for serve.
// Whisper-server is per duplex socket (WhisperLive.Close); the CLI path is
// one-shot per request.
func CloseAudioShared() {
	ClosePiperShared()
	kokoro.CloseShared()
	qwen3asr.CloseShared()
	neuralvad.CloseShared()
}
