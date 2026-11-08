package modality

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/kokoro"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/types/model"
)

// KokoroModelPath is backend_paths.kokoro_model or OLLAMA_KOKORO_MODEL.
func KokoroModelPath(cfg model.ConfigV2) string {
	if p := PathFor(cfg, "kokoro_model"); p != "" {
		return p
	}
	return envconfig.KokoroModel()
}

// KokoroVoicesPath is backend_paths.kokoro_voices or OLLAMA_KOKORO_VOICES.
func KokoroVoicesPath(cfg model.ConfigV2) string {
	if p := PathFor(cfg, "kokoro_voices"); p != "" {
		return p
	}
	return envconfig.KokoroVoices()
}

// KokoroDefaultVoice is tts_default_voice or af_heart.
func KokoroDefaultVoice(cfg model.ConfigV2) string {
	if v := strings.TrimSpace(PathFor(cfg, "tts_default_voice")); v != "" {
		return v
	}
	return kokoro.DefaultVoice
}

// KokoroPathsSet reports that this tag names both Kokoro files in backend_paths.
// OLLAMA_KOKORO_MODEL and OLLAMA_KOKORO_VOICES do not count: those fill paths
// only after the tag is already kokoro.
func KokoroPathsSet(cfg model.ConfigV2) bool {
	return PathFor(cfg, "kokoro_model") != "" && PathFor(cfg, "kokoro_voices") != ""
}

// KokoroAvailable is true when both local weight files exist.
// Path keys win; the env vars are the fallback for a tag that is already kokoro.
func KokoroAvailable(cfg model.ConfigV2) bool {
	return kokoro.Available(KokoroModelPath(cfg), KokoroVoicesPath(cfg))
}

// ResolveSpeechBackend picks the speech driver.
// An explicit modality_backends.speech wins. An empty backend uses piper_model,
// then both kokoro path keys, then tts_url or OLLAMA_TTS_URL.
// Fleet Kokoro env vars never select a backend.
func ResolveSpeechBackend(cfg model.ConfigV2) string {
	backend := BackendFor(cfg, model.ModalitySpeech)
	if backend != "" {
		return backend
	}
	if PathFor(cfg, "piper_model") != "" {
		return model.BackendPiper
	}
	if KokoroPathsSet(cfg) {
		return model.BackendKokoro
	}
	if PathFor(cfg, "tts_url") != "" || envconfig.TTSURL() != "" {
		return model.BackendRemoteTTS
	}
	return ""
}

// IsKokoroRequest is true for empty text, a speed outside [0.5, 2], or a voice
// the library rejected. Those are HTTP 400. A down process is not.
func IsKokoroRequest(err error) bool {
	return errors.Is(err, kokoro.ErrBadRequest)
}

func prepareKokoro(cfg model.ConfigV2, voice, lang string) (string, string, error) {
	if strings.TrimSpace(voice) == "" {
		voice = KokoroDefaultVoice(cfg)
	}
	resolved, err := kokoro.ResolveLang(voice, lang)
	if err != nil {
		return "", "", err
	}
	return voice, resolved, nil
}

func kokoroSpeed(speed *float64) (float64, error) {
	if speed == nil {
		return 1, nil
	}
	s := *speed
	if math.IsNaN(s) || s < 0.5 || s > 2 {
		return 0, &kokoro.BadRequestError{Msg: "speed must be between 0.5 and 2"}
	}
	return s, nil
}

// WarmKokoro starts the shared process in the background.
func WarmKokoro(cfg model.ConfigV2) {
	kokoro.Warm(KokoroModelPath(cfg), KokoroVoicesPath(cfg), KokoroDefaultVoice(cfg))
}

// KokoroFillersReady reports whether the short acknowledgement clips for this
// voice are already in memory. Speed is 1 and emotion is omitted.
func KokoroFillersReady(cfg model.ConfigV2, voice string) bool {
	voice, lang, err := prepareKokoro(cfg, voice, "")
	if err != nil || KokoroModelPath(cfg) == "" || KokoroVoicesPath(cfg) == "" {
		return false
	}
	return piperphrase.FillersReady(KokoroModelPath(cfg), KokoroVoicesPath(cfg), voice, kokoroSpeedKey(1), lang)
}

// WarmKokoroPhrases synthesizes the short acknowledgements for one voice.
// A phrase already in memory is skipped. The first Kokoro error stops the pass.
func WarmKokoroPhrases(ctx context.Context, cfg model.ConfigV2, voice string) {
	if KokoroFillersReady(cfg, voice) {
		return
	}
	for _, phrase := range piperphrase.Fillers() {
		if ctx.Err() != nil {
			return
		}
		if _, _, _, err := SpeechKokoro(ctx, cfg, phrase, voice, "", nil, "", ""); err != nil {
			return
		}
	}
}

// SpeechKokoro synthesizes a WAV. An omitted speed is 1. A speed outside
// [0.5, 2] is a request error. phonemes is the canonical phoneme_silence
// value. emotion scales speed when it is one of the known words.
// A short phrase that was spoken before is returned from memory.
func SpeechKokoro(ctx context.Context, cfg model.ConfigV2, text, voice, lang string, speed *float64, phonemes, emotion string) ([]byte, string, bool, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, "", false, &kokoro.BadRequestError{Msg: "input is required"}
	}
	voice, lang, err := prepareKokoro(cfg, voice, lang)
	if err != nil {
		return nil, "", false, err
	}
	spd, err := kokoroSpeed(speed)
	if err != nil {
		return nil, "", false, err
	}
	if _, emotion, err = kokoro.ApplyEmotion(spd, emotion); err != nil {
		return nil, "", false, err
	}
	key := kokoroCacheKey(cfg, voice, lang, phonemes, emotion, text, spd)
	if wav, ok := piperphrase.Get(key); ok {
		return wav, "audio/wav", true, nil
	}
	ctx, cancel := context.WithTimeout(ctx, envconfig.ModalityKokoroTimeout())
	defer cancel()
	synth := func() ([]byte, string, error) {
		wav, _, err := kokoro.Speak(ctx, KokoroModelPath(cfg), KokoroVoicesPath(cfg), text, voice, lang, phonemes, emotion, spd, KokoroDefaultVoice(cfg))
		if err != nil {
			return nil, "", err
		}
		return wav, "audio/wav", nil
	}
	if !piperphrase.Cacheable(text) {
		wav, ct, err := synth()
		return wav, ct, false, err
	}
	return piperphrase.Flight(key, synth)
}

// SpeechKokoroStream synthesizes and calls emit for each int16 PCM chunk.
// A cached short phrase is one chunk. A new short phrase is stored when it finishes.
func SpeechKokoroStream(ctx context.Context, cfg model.ConfigV2, text, voice, lang string, speed *float64, phonemes, emotion string, emit func(rate int, pcm []byte) error) error {
	text = strings.TrimSpace(text)
	if text == "" {
		return &kokoro.BadRequestError{Msg: "input is required"}
	}
	voice, lang, err := prepareKokoro(cfg, voice, lang)
	if err != nil {
		return err
	}
	spd, err := kokoroSpeed(speed)
	if err != nil {
		return err
	}
	if _, emotion, err = kokoro.ApplyEmotion(spd, emotion); err != nil {
		return err
	}
	key := kokoroCacheKey(cfg, voice, lang, phonemes, emotion, text, spd)
	if wav, ok := piperphrase.Get(key); ok {
		pcm := wavPCM(wav)
		if len(pcm) == 0 {
			return nil
		}
		return emit(kokoro.SampleRate, pcm)
	}
	ctx, cancel := context.WithTimeout(ctx, envconfig.ModalityKokoroTimeout())
	defer cancel()
	var pcm []byte
	cache := piperphrase.Cacheable(text)
	err = kokoro.SpeakStream(ctx, KokoroModelPath(cfg), KokoroVoicesPath(cfg), text, voice, lang, phonemes, emotion, spd, KokoroDefaultVoice(cfg), func(rate int, chunk []byte) error {
		if cache {
			pcm = append(pcm, chunk...)
		}
		return emit(rate, chunk)
	})
	if err != nil {
		return err
	}
	if cache {
		if wav := piperphrase.PCM16WAV(kokoro.SampleRate, pcm); wav != nil {
			piperphrase.Store(key, wav)
		}
	}
	return nil
}

// SpeechKokoroStreamReader returns a reader of raw int16 PCM for HTTP streaming.
// The first chunk, or a request error, settles before the reader is returned
// so the HTTP handler can still send 400. cached is true when the clip was
// already in memory.
func SpeechKokoroStreamReader(ctx context.Context, cfg model.ConfigV2, text, voice, lang string, speed *float64, phonemes, emotion string) (contentType string, rate int, r io.ReadCloser, cached bool, err error) {
	if strings.TrimSpace(text) == "" {
		return "", 0, nil, false, &kokoro.BadRequestError{Msg: "input is required"}
	}
	if _, _, err := prepareKokoro(cfg, voice, lang); err != nil {
		return "", 0, nil, false, err
	}
	spd, err := kokoroSpeed(speed)
	if err != nil {
		return "", 0, nil, false, err
	}
	if _, _, err := kokoro.ApplyEmotion(spd, emotion); err != nil {
		return "", 0, nil, false, err
	}
	pr, pw := io.Pipe()
	rate = kokoro.SampleRate
	first := make(chan error, 1)
	var once sync.Once
	go func() {
		cerr := SpeechKokoroStream(ctx, cfg, text, voice, lang, speed, phonemes, emotion, func(_ int, pcm []byte) error {
			once.Do(func() { first <- nil })
			_, werr := pw.Write(pcm)
			return werr
		})
		once.Do(func() { first <- cerr })
		if cerr != nil {
			_ = pw.CloseWithError(cerr)
			return
		}
		_ = pw.Close()
	}()
	select {
	case err := <-first:
		if err != nil {
			_ = pr.Close()
			return "", 0, nil, false, err
		}
	case <-ctx.Done():
		_ = pr.CloseWithError(ctx.Err())
		return "", 0, nil, false, ctx.Err()
	}
	cached = piperphrase.Cacheable(strings.TrimSpace(text)) && kokoroCached(cfg, voice, lang, phonemes, emotion, strings.TrimSpace(text), spd)
	return fmt.Sprintf("audio/pcm;rate=%d;bits=16", rate), rate, pr, cached, nil
}

func kokoroSpeedKey(speed float64) string {
	return strconv.FormatFloat(speed, 'f', 4, 64)
}

func kokoroCacheKey(cfg model.ConfigV2, voice, lang, phonemes, emotion, text string, speed float64) piperphrase.Key {
	return piperphrase.Key{
		Model:    KokoroModelPath(cfg),
		Config:   KokoroVoicesPath(cfg),
		Voice:    voice,
		Scale:    kokoroSpeedKey(speed),
		Phonemes: phonemes,
		Lang:     lang,
		Emotion:  emotion,
		Text:     text,
	}
}

func kokoroCached(cfg model.ConfigV2, voice, lang, phonemes, emotion, text string, speed float64) bool {
	voice, lang, err := prepareKokoro(cfg, voice, lang)
	if err != nil {
		return false
	}
	if _, emotion, err = kokoro.ApplyEmotion(speed, emotion); err != nil {
		return false
	}
	_, ok := piperphrase.Get(kokoroCacheKey(cfg, voice, lang, phonemes, emotion, text, speed))
	return ok
}
