package modality

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/types/model"
)

// ErrPiperStreamUnavailable means this voice JSON has no audio.sample_rate,
// so the clip is returned as one WAV instead of a stream.
var ErrPiperStreamUnavailable = errors.New("piper voice json has no audio.sample_rate")

// SpeechPiperStream runs Piper with --output_raw and returns a WAV reader.
// The header is sent first. PCM follows as Piper synthesizes each line.
// The process stays up for the next request with the same voice and tuning.
// A phrase already in memory is the cached WAV. cached is true on that hit.
func SpeechPiperStream(ctx context.Context, cfg model.ConfigV2, text, voice string, speed *float64, silence, noise, width, speaker, phonemes string) (string, io.ReadCloser, bool, error) {
	text = strings.TrimSpace(text)
	modelPath := resolvePiperModelPath(cfg, voice)
	if modelPath == "" {
		return "", nil, false, errors.New("set backend_paths.piper_model or piper_voice_<name> for the requested voice")
	}
	configPath := resolvePiperConfigPath(cfg, voice, modelPath)
	scale := piperphrase.LengthScale(speed)
	tashkeel := TashkeelModelPath(cfg)
	espeak := ESpeakDataPath(cfg)
	key := piperphrase.Key{Model: modelPath, Config: configPath, Voice: sanitizeVoiceKey(voice), Scale: scale, Silence: silence, Noise: noise, Width: width, Speaker: speaker, Phonemes: phonemes, Tashkeel: tashkeel, ESpeak: espeak, Text: text}
	if wav, ok := piperphrase.Get(key); ok {
		return "audio/wav", io.NopCloser(bytes.NewReader(wav)), true, nil
	}
	rate, ok := piperConfigRate(configPath)
	if !ok {
		return "", nil, false, ErrPiperStreamUnavailable
	}
	ctx, cancel := piperCtx(ctx)
	pr, pw := io.Pipe()
	args := piperJSONArgs(modelPath, configPath, scale, silence, noise, width, speaker, phonemes, tashkeel, espeak, true)
	go func() {
		err := postPiper.SpeakPCM(ctx, envconfig.PiperBin(), args, text, func(chunk []byte) error {
			_, werr := pw.Write(chunk)
			return werr
		})
		_ = pw.CloseWithError(err)
	}()
	return "audio/wav", &piperRawStream{
		stdout: pr,
		cancel: cancel,
		header: piperphrase.StreamHeader(rate),
		key:    key,
		rate:   rate,
		cache:  piperphrase.Cacheable(text),
	}, false, nil
}

func piperConfigRate(path string) (int, bool) {
	if path == "" {
		return 0, false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	return piperphrase.SampleRate(raw)
}

type piperRawStream struct {
	stdout   io.ReadCloser
	cancel   context.CancelFunc
	header   []byte
	pcm      bytes.Buffer
	key      piperphrase.Key
	rate     int
	cache    bool
	complete bool
	once     sync.Once
	waitErr  error
}

func (p *piperRawStream) Read(b []byte) (int, error) {
	if len(p.header) > 0 {
		n := copy(b, p.header)
		p.header = p.header[n:]
		return n, nil
	}
	n, err := p.stdout.Read(b)
	if n > 0 && p.cache {
		p.pcm.Write(b[:n])
	}
	if err == io.EOF {
		p.complete = true
	}
	return n, err
}

func (p *piperRawStream) Close() error {
	p.once.Do(func() {
		if p.cancel != nil {
			p.cancel()
		}
		if p.stdout != nil {
			p.waitErr = p.stdout.Close()
		}
		if p.complete && p.cache {
			if wav := piperphrase.PCM16WAV(p.rate, p.pcm.Bytes()); wav != nil {
				piperphrase.Store(p.key, wav)
			}
		}
	})
	return p.waitErr
}
