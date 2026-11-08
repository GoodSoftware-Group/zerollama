package modality

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/types/model"
)

// SpeechPiper runs Piper TTS. Resolves ONNX path from backend_paths.piper_model, or
// backend_paths.piper_voice_<voice> when voice is set (voice name sanitized to [a-z0-9]+).
// Optional speed maps to Piper --length_scale (inverse: faster speech => smaller scale).
// silence is --sentence_silence. Empty leaves 0.2 seconds.
// noise is --noise_scale. Empty leaves the voice JSON.
// width is --noise_w. Empty leaves the voice JSON.
// speaker is --speaker. Empty leaves speaker 0.
// phonemes is --phoneme_silence pairs. Empty adds none.
// A configured libtashkeel model is passed as --tashkeel_model. An Arabic voice
// uses it to add diacritics before phonemization.
// A configured espeak-ng data directory is passed as --espeak_data. An eSpeak
// voice uses it to turn text into phonemes.
// A short phrase that was spoken before is returned from memory. cached is true on that hit.
// The Piper process stays up for the next request with the same voice and tuning.
func SpeechPiper(ctx context.Context, cfg model.ConfigV2, text, voice string, speed *float64, silence, noise, width, speaker, phonemes string) ([]byte, string, bool, error) {
	text = strings.TrimSpace(text)
	modelPath := resolvePiperModelPath(cfg, voice)
	if modelPath == "" {
		return nil, "", false, fmt.Errorf("set backend_paths.piper_model or piper_voice_<name> for the requested voice")
	}
	configPath := resolvePiperConfigPath(cfg, voice, modelPath)
	scale := piperphrase.LengthScale(speed)
	tashkeel := TashkeelModelPath(cfg)
	espeak := ESpeakDataPath(cfg)
	key := piperphrase.Key{Model: modelPath, Config: configPath, Voice: sanitizeVoiceKey(voice), Scale: scale, Silence: silence, Noise: noise, Width: width, Speaker: speaker, Phonemes: phonemes, Tashkeel: tashkeel, ESpeak: espeak, Text: text}
	if wav, ok := piperphrase.Get(key); ok {
		return wav, "audio/wav", true, nil
	}
	ctx, cancel := piperCtx(ctx)
	defer cancel()
	synth := func() ([]byte, string, error) {
		return execPiper(ctx, text, modelPath, configPath, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak)
	}
	if !piperphrase.Cacheable(text) {
		data, ct, err := synth()
		return data, ct, false, err
	}
	return piperphrase.Flight(key, synth)
}

// postPiper is the process for POST /v1/audio/speech. The live socket keeps its own.
var postPiper piperphrase.Shared

func execPiper(ctx context.Context, text, modelPath, configPath, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak string) ([]byte, string, error) {
	bin := envconfig.PiperBin()
	if rate, ok := piperConfigRate(configPath); ok {
		var pcm []byte
		err := postPiper.SpeakPCM(ctx, bin, piperJSONArgs(modelPath, configPath, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak, true), text, func(chunk []byte) error {
			pcm = append(pcm, chunk...)
			return nil
		})
		if err != nil {
			return nil, "", piperErr(ctx, bin, err)
		}
		wav := piperphrase.PCM16WAV(rate, pcm)
		if wav == nil {
			return nil, "", fmt.Errorf("piper returned no audio")
		}
		return wav, "audio/wav", nil
	}

	tmpDir, err := os.MkdirTemp("", "ollama-piper-*")
	if err != nil {
		return nil, "", err
	}
	defer os.RemoveAll(tmpDir)
	outPath := filepath.Join(tmpDir, "out.wav")
	data, err := postPiper.Speak(ctx, bin, piperJSONArgs(modelPath, configPath, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak, false), text, outPath)
	if err != nil {
		return nil, "", piperErr(ctx, bin, err)
	}
	return data, "audio/wav", nil
}

func piperErr(ctx context.Context, bin string, err error) error {
	if ctx.Err() != nil {
		return fmt.Errorf("piper: deadline exceeded or cancelled: %w", ctx.Err())
	}
	return fmt.Errorf("piper (%s): %w", bin, err)
}

func piperJSONArgs(modelPath, configPath, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak string, raw bool) []string {
	args := []string{"--model", modelPath, "--json-input"}
	if configPath != "" {
		args = append(args, "--config", configPath)
	}
	args = appendPiperTuning(args, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak)
	if raw {
		args = append(args, "--output_raw")
	}
	return args
}

// TashkeelModelPath is the libtashkeel ONNX file for an Arabic Piper voice.
// backend_paths.tashkeel_model wins. OLLAMA_PIPER_TASHKEEL_MODEL is the fallback.
func TashkeelModelPath(cfg model.ConfigV2) string {
	if p := strings.TrimSpace(PathFor(cfg, "tashkeel_model")); p != "" {
		return p
	}
	return strings.TrimSpace(envconfig.PiperTashkeelModel())
}

// ESpeakDataPath is the espeak-ng data directory for an eSpeak Piper voice.
// backend_paths.espeak_data wins. OLLAMA_PIPER_ESPEAK_DATA is the fallback.
func ESpeakDataPath(cfg model.ConfigV2) string {
	if p := strings.TrimSpace(PathFor(cfg, "espeak_data")); p != "" {
		return p
	}
	return strings.TrimSpace(envconfig.PiperESpeakData())
}

func appendPiperTuning(args []string, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak string) []string {
	if scale != "" {
		args = append(args, "--length_scale", scale)
	}
	if silence != "" {
		args = append(args, "--sentence_silence", silence)
	}
	if noise != "" {
		args = append(args, "--noise_scale", noise)
	}
	if width != "" {
		args = append(args, "--noise_w", width)
	}
	if speaker != "" {
		args = append(args, "--speaker", speaker)
	}
	if extra, err := piperphrase.PhonemeSilenceArgs(phonemes); err == nil && len(extra) > 0 {
		args = append(args, extra...)
	}
	if extra := piperphrase.TashkeelArgs(tashkeel); len(extra) > 0 {
		args = append(args, extra...)
	}
	if extra := piperphrase.ESpeakArgs(espeak); len(extra) > 0 {
		args = append(args, extra...)
	}
	return args
}

func resolvePiperModelPath(cfg model.ConfigV2, voice string) string {
	base := PathFor(cfg, "piper_model")
	voice = strings.TrimSpace(strings.ToLower(voice))
	if voice == "" {
		return base
	}
	key := "piper_voice_" + sanitizeVoiceKey(voice)
	if p := PathFor(cfg, key); p != "" {
		return p
	}
	return base
}

// resolvePiperConfigPath prefers piper_config_<voice>, then <model>.json beside the ONNX, then piper_config.
func resolvePiperConfigPath(cfg model.ConfigV2, voice, modelPath string) string {
	voice = strings.TrimSpace(strings.ToLower(voice))
	if voice != "" {
		if p := PathFor(cfg, "piper_config_"+sanitizeVoiceKey(voice)); p != "" {
			return p
		}
	}
	if modelPath != "" {
		sibling := modelPath + ".json"
		if st, err := os.Stat(sibling); err == nil && !st.IsDir() {
			return sibling
		}
	}
	return PathFor(cfg, "piper_config")
}

func sanitizeVoiceKey(v string) string {
	v = strings.ToLower(v)
	var b strings.Builder
	for _, r := range v {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_':
			b.WriteRune(r)
		}
	}
	return b.String()
}
