package modality

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/types/model"
)

// PiperSession is one live Piper process. The ONNX voice loads when the
// session is created, and later sentences reuse it.
type PiperSession struct {
	cfg    model.ConfigV2
	voice  string
	bin    string
	args   []string
	dir    string
	ready  chan struct{}
	mu     sync.Mutex
	proc   *piperphrase.Proc
	n      int
	rate   int
	closed bool
}

// Voice is the speaker id this process loaded.
func (s *PiperSession) Voice() string {
	if s == nil {
		return ""
	}
	return s.voice
}

// SampleRate is the voice JSON audio.sample_rate. Zero means this process
// writes a whole WAV per sentence.
func (s *PiperSession) SampleRate() int {
	if s == nil {
		return 0
	}
	return s.rate
}

// NewPiperSession starts Piper in the background. A missing model returns nil.
func NewPiperSession(cfg model.ConfigV2, voice string) *PiperSession {
	voice = strings.TrimSpace(voice)
	modelPath := resolvePiperModelPath(cfg, voice)
	if modelPath == "" {
		return nil
	}
	dir, err := os.MkdirTemp("", "piper-live-*")
	if err != nil {
		return nil
	}
	configPath := resolvePiperConfigPath(cfg, voice, modelPath)
	tashkeel := TashkeelModelPath(cfg)
	espeak := ESpeakDataPath(cfg)
	args := []string{"--model", modelPath, "--json-input"}
	if configPath != "" {
		args = append(args, "--config", configPath)
	}
	args = appendPiperTuning(args, "", "", "", "", "", "", tashkeel, espeak)
	rate, _ := piperConfigRate(configPath)
	if rate > 0 {
		args = append(args, "--output_raw")
	}
	s := &PiperSession{
		cfg:   cfg,
		voice: voice,
		bin:   envconfig.PiperBin(),
		args:  args,
		dir:   dir,
		ready: make(chan struct{}),
		rate:  rate,
	}
	go s.warm()
	return s
}

func (s *PiperSession) warm() {
	proc, err := piperphrase.Start(s.bin, s.args)
	s.mu.Lock()
	if err == nil {
		s.proc = proc
	} else if proc != nil {
		proc.Close()
	}
	s.mu.Unlock()
	close(s.ready)
}

// Speak synthesizes one sentence. A voice other than the one this process
// loaded uses a one-shot Piper. A cancelled ctx stops the loaded process.
func (s *PiperSession) Speak(ctx context.Context, text, voice string) ([]byte, error) {
	if s == nil {
		return nil, errors.New("piper is not running")
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, errors.New("nothing to speak")
	}
	voice = strings.TrimSpace(voice)
	if voice != "" && s.voice != "" && voice != s.voice {
		data, _, _, err := SpeechPiper(ctx, s.cfg, text, voice, nil, "", "", "", "", "")
		return data, err
	}
	key := piperphrase.Key{
		Model:    resolvePiperModelPath(s.cfg, s.voice),
		Config:   resolvePiperConfigPath(s.cfg, s.voice, resolvePiperModelPath(s.cfg, s.voice)),
		Voice:    sanitizeVoiceKey(s.voice),
		Tashkeel: TashkeelModelPath(s.cfg),
		ESpeak:   ESpeakDataPath(s.cfg),
		Text:     text,
	}
	if wav, ok := piperphrase.Get(key); ok {
		return wav, nil
	}
	data, err := s.synth(ctx, text)
	if err != nil {
		s.reset()
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		data, err = s.synth(ctx, text)
	}
	if err != nil {
		return nil, err
	}
	if piperphrase.Cacheable(text) {
		piperphrase.Store(key, data)
	}
	return data, nil
}

func (s *PiperSession) synth(ctx context.Context, text string) ([]byte, error) {
	<-s.ready
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, errors.New("piper is not running")
	}
	if s.proc == nil {
		proc, err := piperphrase.Start(s.bin, s.args)
		if err != nil {
			return nil, err
		}
		s.proc = proc
	}
	s.n++
	if s.rate > 0 {
		var pcm []byte
		err := s.proc.SpeakPCM(ctx, text, func(chunk []byte) error {
			pcm = append(pcm, chunk...)
			return nil
		})
		if err != nil {
			return nil, err
		}
		wav := piperphrase.PCM16WAV(s.rate, pcm)
		if wav == nil {
			return nil, errors.New("piper returned no audio")
		}
		return wav, nil
	}
	path := filepath.Join(s.dir, strconv.Itoa(s.n)+".wav")
	return s.proc.Speak(ctx, text, path)
}

// SpeakStream emits PCM as Piper synthesizes it. A voice other than the one
// this process loaded is spoken as one clip. rate is SampleRate.
func (s *PiperSession) SpeakStream(ctx context.Context, text, voice string, emit func([]byte) error) error {
	if s == nil {
		return errors.New("piper is not running")
	}
	if s.rate <= 0 {
		data, err := s.Speak(ctx, text, voice)
		if err != nil {
			return err
		}
		return emit(wavPCM(data))
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return errors.New("nothing to speak")
	}
	voice = strings.TrimSpace(voice)
	if voice != "" && s.voice != "" && voice != s.voice {
		data, _, _, err := SpeechPiper(ctx, s.cfg, text, voice, nil, "", "", "", "", "")
		if err != nil {
			return err
		}
		return emit(wavPCM(data))
	}
	key := piperphrase.Key{
		Model:    resolvePiperModelPath(s.cfg, s.voice),
		Config:   resolvePiperConfigPath(s.cfg, s.voice, resolvePiperModelPath(s.cfg, s.voice)),
		Voice:    sanitizeVoiceKey(s.voice),
		Tashkeel: TashkeelModelPath(s.cfg),
		ESpeak:   ESpeakDataPath(s.cfg),
		Text:     text,
	}
	if wav, ok := piperphrase.Get(key); ok {
		return emit(wavPCM(wav))
	}
	var pcm []byte
	err := s.synthStream(ctx, text, func(chunk []byte) error {
		pcm = append(pcm, chunk...)
		if emit == nil {
			return nil
		}
		return emit(chunk)
	})
	if err != nil {
		s.reset()
		if len(pcm) == 0 && ctx.Err() == nil {
			pcm = nil
			err = s.synthStream(ctx, text, func(chunk []byte) error {
				pcm = append(pcm, chunk...)
				if emit == nil {
					return nil
				}
				return emit(chunk)
			})
		}
	}
	if err != nil {
		return err
	}
	if wav := piperphrase.PCM16WAV(s.rate, pcm); wav != nil && piperphrase.Cacheable(text) {
		piperphrase.Store(key, wav)
	}
	return nil
}

func (s *PiperSession) synthStream(ctx context.Context, text string, emit func([]byte) error) error {
	<-s.ready
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errors.New("piper is not running")
	}
	if s.proc == nil {
		proc, err := piperphrase.Start(s.bin, s.args)
		if err != nil {
			return err
		}
		s.proc = proc
	}
	return s.proc.SpeakPCM(ctx, text, emit)
}

func wavPCM(data []byte) []byte {
	if len(data) > 44 && string(data[:4]) == "RIFF" {
		return data[44:]
	}
	return data
}

func (s *PiperSession) reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.proc != nil {
		s.proc.Close()
		s.proc = nil
	}
}

// Close stops the process and removes its WAV directory.
func (s *PiperSession) Close() {
	if s == nil {
		return
	}
	<-s.ready
	s.mu.Lock()
	s.closed = true
	proc := s.proc
	s.proc = nil
	dir := s.dir
	s.mu.Unlock()
	if proc != nil {
		proc.Close()
	}
	_ = os.RemoveAll(dir)
}
