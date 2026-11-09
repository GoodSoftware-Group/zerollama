package modality

import (
	"context"
	"log/slog"
	"strings"

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

// PiperHoldsReady reports whether the duplex hesitation and HoldMore clips
// for this voice are already in memory.
func PiperHoldsReady(cfg model.ConfigV2, voice string) bool {
	modelPath := resolvePiperModelPath(cfg, voice)
	if modelPath == "" {
		return false
	}
	key := piperphrase.Key{
		Model:  modelPath,
		Config: resolvePiperConfigPath(cfg, voice, modelPath),
		Voice:  sanitizeVoiceKey(voice),
	}
	for _, phrase := range []string{piperphrase.Hesitation(), piperphrase.HoldMore()} {
		k := key
		k.Text = phrase
		if _, ok := piperphrase.Get(k); !ok {
			return false
		}
	}
	return true
}

// WarmPiperPhrases synthesizes the short acknowledgements for one voice.
// A phrase already in memory is skipped. The first Piper error stops the pass.
func WarmPiperPhrases(ctx context.Context, cfg model.ConfigV2, voice string) {
	if PiperFillersReady(cfg, voice) && PiperHoldsReady(cfg, voice) {
		return
	}
	for _, phrase := range append(piperphrase.Fillers(), piperphrase.Hesitation()) {
		if ctx.Err() != nil {
			return
		}
		if _, _, _, err := SpeechPiper(ctx, cfg, phrase, voice, nil, "", "", "", "", ""); err != nil {
			slog.Debug("piper phrase warm stopped", "error", err)
			return
		}
	}
}

// PiperCachedWAV returns a cached WAV for text when the phrase is already warm.
func PiperCachedWAV(cfg model.ConfigV2, voice, text string) ([]byte, bool) {
	modelPath := resolvePiperModelPath(cfg, voice)
	if modelPath == "" || strings.TrimSpace(text) == "" {
		return nil, false
	}
	wav, ok := piperphrase.Get(piperphrase.Key{
		Model:  modelPath,
		Config: resolvePiperConfigPath(cfg, voice, modelPath),
		Voice:  sanitizeVoiceKey(voice),
		Text:   text,
	})
	return wav, ok
}
