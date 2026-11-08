package modality

import (
	"context"
	"log/slog"

	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/types/model"
)

// PiperFillersReady reports whether the short acknowledgement clips for this
// voice are already in memory. The length scale is Piper's default.
func PiperFillersReady(cfg model.ConfigV2, voice string) bool {
	modelPath := resolvePiperModelPath(cfg, voice)
	if modelPath == "" {
		return false
	}
	return piperphrase.FillersReady(modelPath, resolvePiperConfigPath(cfg, voice, modelPath), sanitizeVoiceKey(voice), "", "")
}

// WarmPiperPhrases synthesizes the short acknowledgements for one voice.
// A phrase already in memory is skipped. The first Piper error stops the pass.
func WarmPiperPhrases(ctx context.Context, cfg model.ConfigV2, voice string) {
	if PiperFillersReady(cfg, voice) {
		return
	}
	for _, phrase := range piperphrase.Fillers() {
		if ctx.Err() != nil {
			return
		}
		if _, _, _, err := SpeechPiper(ctx, cfg, phrase, voice, nil, "", "", "", "", ""); err != nil {
			slog.Debug("piper phrase warm stopped", "error", err)
			return
		}
	}
}
