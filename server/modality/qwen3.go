package modality

import (
	"context"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/qwen3asr"
	"github.com/ollama/ollama/types/model"
)

// UsesQwen3ASR reports modality_backends.transcribe=qwen3-asr.
func UsesQwen3ASR(cfg model.ConfigV2) bool {
	return BackendFor(cfg, model.ModalityTranscribe) == model.BackendQwen3ASR
}

// Qwen3ASRPath is backend_paths.qwen3_asr, else OLLAMA_QWEN3_ASR.
func Qwen3ASRPath(cfg model.ConfigV2) string {
	p := strings.TrimSpace(PathFor(cfg, "qwen3_asr"))
	if p == "" {
		p = strings.TrimSpace(envconfig.Qwen3ASRModelPath())
	}
	return p
}

// TranscribeQwen3 posts wav to the resident Qwen3-ASR process.
// language is a whisper-style code (en) or empty. ErrUnavailable means
// the process could not start and the caller should use Whisper.
func TranscribeQwen3(ctx context.Context, cfg model.ConfigV2, wav []byte, language string) (string, error) {
	return qwen3asr.Transcribe(ctx, Qwen3ASRPath(cfg), wav, language)
}
