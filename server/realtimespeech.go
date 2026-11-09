package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/server/modality/neuralvad"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/server/modality/qwen3asr"
	"github.com/ollama/ollama/server/voicepipe"
	"github.com/ollama/ollama/server/wsproxy"
	"github.com/ollama/ollama/types/model"
)

func (s *Server) serveSpeech(c *gin.Context, modelName string, m *Model) {
	conn, src, err := wsproxy.Accept(c.Writer, c.Request)
	if err != nil {
		if !c.Writer.Written() {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		}
		return
	}
	piper := openLivePiper(m, strings.TrimSpace(c.Query("voice")))
	defer piper.Close()
	speak, speakMore := s.liveSpeaker(m, modelName, piper)
	eng := voicepipe.Engine{
		OutputRate: genSampleRate(m),
		Language:   strings.TrimSpace(c.Query("language")),
		Voice:      strings.TrimSpace(c.Query("voice")),
		Emotion:    strings.TrimSpace(c.Query("emotion")),
		Speak:      speak,
		SpeakMore:  speakMore,
	}
	_ = voicepipe.Serve(c.Request.Context(), conn, src, eng)
}

func (s *Server) serveDuplex(c *gin.Context, chatName string, chat *Model) {
	if hostSpeechOnly(chat) || modelMTMDRole(chat).SpeechOnly() || hostTranscribeOnly(chat) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "duplex model must be a chat model; set voice and asr for the other steps"})
		return
	}
	voiceName := strings.TrimSpace(c.Query("voice"))
	asrName := strings.TrimSpace(c.Query("asr"))
	if voiceName == "" || asrName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "duplex requires voice and asr"})
		return
	}
	rate, err := strconv.Atoi(strings.TrimSpace(c.Query("rate")))
	if err != nil || rate <= 0 || rate > 192000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "duplex requires rate for the input pcm16"})
		return
	}
	voiceModel, voiceTag, err := loadLocalModel(voiceName)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if model.SpeechBlock(modelMTMDRole(voiceModel)) != "" && !hostSpeechOnly(voiceModel) && !modelMTMDRole(voiceModel).SpeechOnly() {
		c.JSON(http.StatusBadRequest, gin.H{"error": "voice must be a speech model"})
		return
	}
	if !canSpeak(voiceModel) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "voice must be a speech model"})
		return
	}
	asrModel, asrTag, err := loadLocalModel(asrName)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if !canHear(asrModel) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "asr must be a whisper or llama-server ASR model"})
		return
	}
	var turn *voicepipe.Turn
	var speechScore voicepipe.SpeechScorer
	var onTurnReset func()
	if on, parsed, err := parseDuplexTurnQuery(c.Query("turn")); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else if on {
		turn = &parsed
		// Silero windows are 8 kHz / 16 kHz only. Other rates stay on amplitude
		// with the client's amplitude threshold (not a neural 0.5 rewrite).
		if rate == 8000 || rate == 16000 {
			if sid, score := neuralvad.Attach(); score != nil {
				turn.ApplyNeuralThreshold()
				speechScore = score
				onTurnReset = func() { neuralvad.Reset(sid) }
			}
		}
	}
	conn, src, err := wsproxy.Accept(c.Writer, c.Request)
	if err != nil {
		if !c.Writer.Written() {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		}
		return
	}
	speaker := strings.TrimSpace(c.Query("speaker"))
	piper := openLivePiper(voiceModel, speaker)
	defer piper.Close()
	live := openLiveWhisper(c.Request.Context(), asrModel)
	defer live.Close()
	if !voiceHoldsRunner(voiceModel) {
		pinCtx, pinCancel := context.WithCancel(c.Request.Context())
		defer pinCancel()
		go s.pinDuplexChat(pinCtx, chatName)
	}
	var chatMu sync.Mutex
	stages := &voicepipe.StageReporter{}
	eng := voicepipe.Engine{
		Duplex:      true,
		InputRate:   rate,
		OutputRate:  genSampleRate(voiceModel),
		Language:    strings.TrimSpace(c.Query("language")),
		Voice:       speaker,
		Emotion:     strings.TrimSpace(c.Query("emotion")),
		Turn:        turn,
		SpeechScore: speechScore,
		OnTurnReset: onTurnReset,
		Stages:      stages,
		ASR:         asrTag,
		ASRBackend:  duplexASRBackend(asrModel),
		PrepareHolds: func(ctx context.Context) (voicepipe.Clip, voicepipe.Clip) {
			return duplexHoldClips(ctx, voiceModel, speaker, piper)
		},
		HearEx: func(ctx context.Context, wav []byte, language string) (string, string, error) {
			return s.hearClip(ctx, asrModel, asrTag, wav, language)
		},
		WarmHearEx: warmASR(asrModel, live),
		StreamReply: func(ctx context.Context, instructions, user string, prior []voicepipe.Exchange, emit func(string) error) error {
			chatMu.Lock()
			defer chatMu.Unlock()
			return s.replyTextStream(ctx, chatName, instructions, user, prior, emit, stages)
		},
		WarmReply: func(ctx context.Context, instructions string, prior []voicepipe.Exchange) error {
			chatMu.Lock()
			defer chatMu.Unlock()
			return s.warmDuplexPrefix(ctx, chatName, instructions, prior)
		},
	}
	eng.Speak, eng.SpeakMore = s.liveSpeaker(voiceModel, voiceTag, piper)
	_ = voicepipe.Serve(c.Request.Context(), conn, src, eng)
}

func parseDuplexTurnQuery(q string) (bool, voicepipe.Turn, error) {
	switch strings.ToLower(strings.TrimSpace(q)) {
	case "", "none":
		return false, voicepipe.Turn{}, nil
	case "server_vad":
		return true, voicepipe.DuplexDefaultTurn(), nil
	default:
		return false, voicepipe.Turn{}, errors.New("turn must be server_vad or none")
	}
}

// duplexHoldClips warms and loads the hesitation and "One moment." clips for
// the live voice. Missing weights leave both empty so the socket skips the hold.
func duplexHoldClips(ctx context.Context, m *Model, voice string, piper *modality.PiperSession) (hold, more voicepipe.Clip) {
	if m == nil {
		return voicepipe.Clip{}, voicepipe.Clip{}
	}
	switch speechBackend(m.Config) {
	case model.BackendKokoro:
		modality.WarmKokoroPhrases(ctx, m.Config, voice)
		if wav, ok := modality.KokoroCachedWAV(m.Config, voice, piperphrase.Hesitation()); ok {
			hold = decodeHoldWAV(wav)
		}
		if wav, ok := modality.KokoroCachedWAV(m.Config, voice, piperphrase.HoldMore()); ok {
			more = decodeHoldWAV(wav)
		}
		return hold, more
	case model.BackendPiper:
		modality.WarmPiperPhrases(ctx, m.Config, voice)
		if piper != nil {
			if wav, err := piper.Speak(ctx, piperphrase.Hesitation(), voice); err == nil {
				hold = decodeHoldWAV(wav)
			}
			if wav, err := piper.Speak(ctx, piperphrase.HoldMore(), voice); err == nil {
				more = decodeHoldWAV(wav)
			}
		}
		if len(hold.PCM) == 0 {
			if wav, ok := modality.PiperCachedWAV(m.Config, voice, piperphrase.Hesitation()); ok {
				hold = decodeHoldWAV(wav)
			}
		}
		if len(more.PCM) == 0 {
			if wav, ok := modality.PiperCachedWAV(m.Config, voice, piperphrase.HoldMore()); ok {
				more = decodeHoldWAV(wav)
			}
		}
		return hold, more
	default:
		return voicepipe.Clip{}, voicepipe.Clip{}
	}
}

func decodeHoldWAV(wav []byte) voicepipe.Clip {
	if len(wav) == 0 {
		return voicepipe.Clip{}
	}
	rate, pcm, err := voicepipe.ParseWAV(wav)
	if err != nil || rate <= 0 || len(pcm) < 2 {
		return voicepipe.Clip{}
	}
	return voicepipe.Clip{
		MIME: "audio/pcm",
		Rate: rate,
		PCM:  voicepipe.TrimSilencePCM16(pcm, 200),
	}
}

func canSpeak(m *Model) bool {
	if m == nil {
		return false
	}
	if hostSpeechOnly(m) || modelMTMDRole(m).SpeechOnly() {
		return true
	}
	backend := modality.BackendFor(m.Config, model.ModalitySpeech)
	return backend == model.BackendPiper || backend == model.BackendKokoro || backend == model.BackendRemoteTTS || backend == model.BackendMusic3
}

func canHear(m *Model) bool {
	if m == nil {
		return false
	}
	if hostTranscribeOnly(m) || modelUsesLlamaASR(m) || modality.UsesQwen3ASR(m.Config) {
		return true
	}
	return modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendWhisper || modality.PathFor(m.Config, "whisper_model") != ""
}

func loadLocalModel(raw string) (*Model, string, error) {
	modelRef, err := parseAndValidateModelRef(raw)
	if err != nil {
		return nil, "", err
	}
	if modelRef.Source == modelSourceCloud {
		return nil, "", errors.New("realtime is a local socket")
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		return nil, "", errors.New("model not found")
	}
	m, err := GetModel(name.String())
	if err != nil {
		return nil, "", errors.New("model not found")
	}
	return m, name.String(), nil
}

// voiceHoldsRunner is true when speaking needs the same llama runner as the chat model.
func voiceHoldsRunner(m *Model) bool {
	if m == nil {
		return false
	}
	role := modelMTMDRole(m)
	return role.Speech && (m.ModelPath != "" || len(m.ProjectorPaths) > 0)
}

// pinDuplexChat loads the chat model for the life of the socket, so the first
// transcript does not wait on weights. The scheduler drops the ref when ctx ends.
func (s *Server) pinDuplexChat(ctx context.Context, name string) {
	if m, err := GetModel(name); err == nil && deferInferenceToRuntime(m) {
		// One token loads the runtime GGUF before the first transcript.
		// The spoken reply is a later /api/chat on the same sidecar.
		_ = s.replyRuntimeChat(ctx, name, m, "", ".", nil, func(string) error { return nil }, 1, nil)
		return
	}
	_, _, _, _, release, err := s.scheduleRunner(ctx, name, []model.Capability{model.CapabilityCompletion}, nil, nil, nil, nil, nil)
	if err != nil {
		return
	}
	release()
}

func openLivePiper(m *Model, voice string) *modality.PiperSession {
	if m == nil {
		return nil
	}
	role := modelMTMDRole(m)
	if role.Speech && (m.ModelPath != "" || len(m.ProjectorPaths) > 0) {
		return nil
	}
	backend := modality.BackendFor(m.Config, model.ModalitySpeech)
	if backend == "" && modality.PathFor(m.Config, "piper_model") != "" {
		backend = model.BackendPiper
	}
	if backend != model.BackendPiper {
		return nil
	}
	return modality.NewPiperSession(m.Config, voice)
}

func (s *Server) liveSpeaker(m *Model, modelName string, piper *modality.PiperSession) (func(context.Context, string, string, string, string) (voicepipe.Clip, error), func(context.Context, string, string, string, string, func(voicepipe.Clip) error) error) {
	if m != nil && speechBackend(m.Config) == model.BackendKokoro {
		modality.WarmKokoro(m.Config)
	}
	speak := func(ctx context.Context, text, voice, language, emotion string) (voicepipe.Clip, error) {
		return s.voiceClip(ctx, m, modelName, text, voice, language, emotion, piper)
	}
	more := func(ctx context.Context, text, voice, language, emotion string, emit func(voicepipe.Clip) error) error {
		if piper != nil && piper.SampleRate() > 0 && (strings.TrimSpace(voice) == "" || strings.TrimSpace(voice) == strings.TrimSpace(piperVoice(piper))) {
			rate := piper.SampleRate()
			return piper.SpeakStream(ctx, text, voice, func(pcm []byte) error {
				if len(pcm) == 0 {
					return nil
				}
				return emit(voicepipe.Clip{MIME: "audio/pcm", Rate: rate, PCM: pcm})
			})
		}
		if m != nil && speechBackend(m.Config) == model.BackendKokoro {
			return modality.SpeechKokoroStream(ctx, m.Config, text, voice, language, nil, "", emotion, func(rate int, pcm []byte) error {
				if len(pcm) == 0 {
					return nil
				}
				if rate <= 0 {
					rate = 24000
				}
				return emit(voicepipe.Clip{MIME: "audio/pcm", Rate: rate, PCM: pcm})
			})
		}
		clip, err := speak(ctx, text, voice, language, emotion)
		if err != nil {
			return err
		}
		return emit(clip)
	}
	return speak, more
}

func piperVoice(p *modality.PiperSession) string {
	if p == nil {
		return ""
	}
	return p.Voice()
}

func (s *Server) voiceClip(ctx context.Context, m *Model, modelName, text, voice, language, emotion string, piper *modality.PiperSession) (voicepipe.Clip, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return voicepipe.Clip{}, errors.New("nothing to speak")
	}
	if voice == "" {
		voice = strings.TrimSpace(modality.PathFor(m.Config, "tts_default_voice"))
	}
	role := modelMTMDRole(m)
	if role.Speech && (m.ModelPath != "" || len(m.ProjectorPaths) > 0) {
		return s.llamaVoiceClip(ctx, modelName, text, voice, language, genSampleRate(m))
	}
	backend := speechBackend(m.Config)
	switch backend {
	case model.BackendMusic3:
		return voicepipe.Clip{}, errors.New("this model queues audio on POST /v1/audio/speech")
	case model.BackendKokoro:
		data, contentType, _, err := modality.SpeechKokoro(ctx, m.Config, text, voice, language, nil, "", emotion)
		if err != nil {
			return voicepipe.Clip{}, err
		}
		return clipFromAudio(contentType, data, 0)
	case model.BackendPiper:
		if piper != nil {
			data, err := piper.Speak(ctx, text, voice)
			if err != nil {
				return voicepipe.Clip{}, err
			}
			return clipFromAudio("audio/wav", data, 0)
		}
		data, contentType, _, err := modality.SpeechPiper(ctx, m.Config, text, voice, nil, "", "", "", "", "")
		if err != nil {
			return voicepipe.Clip{}, err
		}
		return clipFromAudio(contentType, data, 0)
	case model.BackendRemoteTTS:
		data, contentType, _, err := modality.SpeechRemote(ctx, m.Config, modelName, text, voice, "wav", emotion, nil)
		if err != nil {
			return voicepipe.Clip{}, err
		}
		return clipFromAudio(contentType, data, 0)
	default:
		return voicepipe.Clip{}, fmt.Errorf("unsupported speech backend %q", backend)
	}
}

func (s *Server) llamaVoiceClip(ctx context.Context, modelName, text, voice, language string, rate int) (voicepipe.Clip, error) {
	runner, _, _, _, release, err := s.scheduleRunner(ctx, modelName, []model.Capability{model.CapabilitySpeech}, nil, nil, nil, nil, nil)
	if err != nil {
		return voicepipe.Clip{}, err
	}
	defer release()
	speaker, ok := runner.(llm.LlamaSpeaker)
	if !ok {
		return voicepipe.Clip{}, errors.New("speech requires llama-server")
	}
	contentType, audio, err := speaker.Speak(ctx, text, voice, "wav", language, nil, false, nil, nil, llm.SpeechSample{})
	if err != nil {
		return voicepipe.Clip{}, err
	}
	defer audio.Close()
	buf, err := io.ReadAll(audio)
	if err != nil {
		return voicepipe.Clip{}, err
	}
	return clipFromAudio(contentType, buf, rate)
}

func clipFromAudio(contentType string, data []byte, fallbackRate int) (voicepipe.Clip, error) {
	ct := strings.ToLower(contentType)
	if strings.Contains(ct, "wav") || (len(data) > 12 && string(data[0:4]) == "RIFF") {
		rate, pcm, err := voicepipe.ParseWAV(data)
		if err != nil {
			return voicepipe.Clip{}, err
		}
		return voicepipe.Clip{MIME: "audio/pcm", Rate: rate, PCM: pcm}, nil
	}
	if strings.Contains(ct, "pcm") {
		if fallbackRate <= 0 {
			return voicepipe.Clip{}, errors.New("pcm reply has no sample rate")
		}
		return voicepipe.Clip{MIME: "audio/pcm", Rate: fallbackRate, PCM: data}, nil
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	return voicepipe.Clip{MIME: contentType, Raw: data, Rate: fallbackRate}, nil
}

func openLiveWhisper(ctx context.Context, m *Model) *modality.WhisperLive {
	if m == nil || modelUsesLlamaASR(m) || modality.UsesQwen3ASR(m.Config) {
		return nil
	}
	whisper := hostTranscribeOnly(m) || modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendWhisper || modality.PathFor(m.Config, "whisper_model") != ""
	if !whisper {
		return nil
	}
	return modality.StartWhisperLive(ctx, m.Config)
}

func duplexASRBackend(m *Model) string {
	if m == nil {
		return ""
	}
	if modality.UsesQwen3ASR(m.Config) {
		return "qwen3-asr"
	}
	if modelUsesLlamaASR(m) {
		return "llama-asr"
	}
	return "whisper"
}

func warmASR(m *Model, live *modality.WhisperLive) func(context.Context, string) (func(context.Context, []byte) (string, string, error), error) {
	if m == nil || modelUsesLlamaASR(m) {
		return nil
	}
	if modality.UsesQwen3ASR(m.Config) {
		return func(ctx context.Context, language string) (func(context.Context, []byte) (string, string, error), error) {
			return func(ctx context.Context, wav []byte) (string, string, error) {
				text, err := modality.TranscribeQwen3(ctx, m.Config, wav, language)
				if errors.Is(err, qwen3asr.ErrUnavailable) {
					slog.Warn("duplex asr: qwen3-asr unavailable, using whisper fallback", "err", err)
					tr, werr := modality.TranscribeWhisper(ctx, m.Config, wav, "clip.wav", language, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
					if werr != nil {
						return "", "", fmt.Errorf("%w (whisper fallback: %v)", err, werr)
					}
					return tr.Text, "whisper_fallback", nil
				}
				if err != nil {
					return "", "", err
				}
				return text, "qwen3-asr", nil
			}, nil
		}
	}
	whisper := hostTranscribeOnly(m) || modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendWhisper || modality.PathFor(m.Config, "whisper_model") != ""
	if !whisper {
		return nil
	}
	return func(ctx context.Context, language string) (func(context.Context, []byte) (string, string, error), error) {
		fin, err := modality.WarmLive(ctx, live, m.Config, language)
		if err != nil || fin == nil {
			return nil, err
		}
		return func(ctx context.Context, wav []byte) (string, string, error) {
			text, err := fin(ctx, wav)
			return text, "whisper", err
		}, nil
	}
}

func (s *Server) hearClip(ctx context.Context, m *Model, modelName string, wav []byte, language string) (string, string, error) {
	var qwenErr error
	if modality.UsesQwen3ASR(m.Config) {
		text, err := modality.TranscribeQwen3(ctx, m.Config, wav, language)
		if err == nil {
			return text, "qwen3-asr", nil
		}
		if !errors.Is(err, qwen3asr.ErrUnavailable) {
			return "", "", err
		}
		qwenErr = err
		slog.Warn("duplex asr: qwen3-asr unavailable, using whisper fallback", "err", err)
		// Fall through to Whisper CLI when the resident process cannot start.
	}
	whisper := hostTranscribeOnly(m) || modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendWhisper || modality.PathFor(m.Config, "whisper_model") != "" || modality.UsesQwen3ASR(m.Config)
	if whisper && !modelUsesLlamaASR(m) {
		tr, err := modality.TranscribeWhisper(ctx, m.Config, wav, "clip.wav", language, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
		if err != nil {
			if qwenErr != nil {
				return "", "", fmt.Errorf("%w (whisper fallback: %v)", qwenErr, err)
			}
			return "", "", err
		}
		backend := "whisper"
		if qwenErr != nil {
			backend = "whisper_fallback"
		}
		return tr.Text, backend, nil
	}
	runner, _, _, _, release, err := s.scheduleRunner(ctx, modelName, nil, nil, nil, nil, nil, nil)
	if err != nil {
		return "", "", err
	}
	defer release()
	asr, ok := runner.(llm.LlamaASRTranscriber)
	if !ok {
		return "", "", errors.New("asr requires a whisper model or llama-server ASR")
	}
	_, body, err := asr.TranscribeAudio(ctx, wav, "clip.wav", language, "", "", "", false)
	if err != nil {
		return "", "", err
	}
	defer body.Close()
	buf, err := io.ReadAll(body)
	if err != nil {
		return "", "", err
	}
	var parsed struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(buf, &parsed); err == nil && strings.TrimSpace(parsed.Text) != "" {
		return strings.TrimSpace(parsed.Text), "llama-asr", nil
	}
	return strings.TrimSpace(string(buf)), "llama-asr", nil
}

// duplexThink is off so a spoken reply starts with the words, not a thinking block.
func duplexThink() *api.ThinkValue {
	return &api.ThinkValue{Value: false}
}

// duplexPrompt renders the chat template with thinking off. The ggml runner
// has no native template, so the Go template is used. The plain user line is
// the fallback when neither render succeeds.
func duplexPrompt(ctx context.Context, runner llm.LlamaServer, m *Model, msgs []api.Message, user string) string {
	if runner != nil {
		if rendered, err := runner.ApplyChatTemplate(ctx, llm.ChatRequest{Messages: msgs, Think: duplexThink()}); err == nil {
			if text := strings.TrimSpace(rendered); text != "" {
				return text
			}
		}
	}
	if m != nil && m.Template != nil {
		if rendered, err := renderPrompt(m, msgs, nil, duplexThink()); err == nil {
			if text := strings.TrimSpace(rendered); text != "" {
				return text
			}
		}
	}
	return user
}

func duplexMessages(instructions, user string, prior []voicepipe.Exchange) []api.Message {
	msgs := []api.Message{}
	if text := strings.TrimSpace(instructions); text != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: text})
	}
	for _, ex := range prior {
		if text := strings.TrimSpace(ex.User); text != "" {
			msgs = append(msgs, api.Message{Role: "user", Content: text})
		}
		if text := strings.TrimSpace(ex.Assistant); text != "" {
			msgs = append(msgs, api.Message{Role: "assistant", Content: text})
		}
	}
	msgs = append(msgs, api.Message{Role: "user", Content: user})
	return msgs
}

func duplexPrefixMessages(instructions string, prior []voicepipe.Exchange) []api.Message {
	msgs := duplexMessages(instructions, "", prior)
	if len(msgs) == 0 {
		return nil
	}
	last := msgs[len(msgs)-1]
	if last.Role == "user" && strings.TrimSpace(last.Content) == "" {
		msgs = msgs[:len(msgs)-1]
	}
	return msgs
}

// warmDuplexPrefix runs one token on the instructions and earlier turns.
// The next reply reuses that prefix and prefills only the new utterance.
func (s *Server) warmDuplexPrefix(ctx context.Context, modelName, instructions string, prior []voicepipe.Exchange) error {
	msgs := duplexPrefixMessages(instructions, prior)
	if len(msgs) == 0 {
		return nil
	}
	if m, err := GetModel(modelName); err == nil && deferInferenceToRuntime(m) {
		// The runtime keeps its own cache. A llama-server prefix warm is the
		// call that returns "model uses zerollama-runtime inference".
		return nil
	}
	runner, m, opts, _, release, err := s.scheduleRunner(ctx, modelName, []model.Capability{model.CapabilityCompletion}, nil, nil, nil, nil, nil)
	if err != nil {
		return err
	}
	defer release()
	prompt := duplexPrompt(ctx, runner, m, msgs, "")
	if strings.TrimSpace(prompt) == "" {
		return nil
	}
	opt := duplexOptions(opts)
	opt.NumPredict = 1
	mtp := duplexMTP(m)
	return runner.Completion(ctx, llm.CompletionRequest{Prompt: prompt, Options: opt, EnableMTP: mtp}, func(llm.CompletionResponse) {})
}

func duplexOptions(opts *api.Options) *api.Options {
	base := api.DefaultOptions()
	if opts != nil {
		base = *opts
	}
	base.NumPredict = voicepipe.ReplyPredict(base.NumPredict)
	base.NumCtx = voicepipe.ReplyNumCtx(base.NumCtx)
	return &base
}

func (s *Server) replyTextStream(ctx context.Context, modelName, instructions, user string, prior []voicepipe.Exchange, emit func(string) error, stages *voicepipe.StageReporter) error {
	if m, err := GetModel(modelName); err == nil && deferInferenceToRuntime(m) {
		return s.replyRuntimeChat(ctx, modelName, m, instructions, user, prior, emit, 0, stages)
	}
	runner, m, opts, _, release, err := s.scheduleRunner(ctx, modelName, []model.Capability{model.CapabilityCompletion}, nil, nil, nil, nil, nil)
	if err != nil {
		return err
	}
	defer release()
	prompt := duplexPrompt(ctx, runner, m, duplexMessages(instructions, user, prior), user)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var stream voicepipe.SentenceStream
	var emitErr error
	first := true
	mtp := duplexMTP(m)
	err = runner.Completion(ctx, llm.CompletionRequest{Prompt: prompt, Options: duplexOptions(opts), EnableMTP: mtp}, func(res llm.CompletionResponse) {
		if emitErr != nil || res.Content == "" {
			return
		}
		if first {
			first = false
			stages.Report(voicepipe.StageFirstToken, nil)
		}
		if pushErr := stream.Push(res.Content, emit); pushErr != nil {
			emitErr = pushErr
			cancel()
		}
	})
	if emitErr != nil {
		return emitErr
	}
	if err != nil {
		return err
	}
	return stream.Flush(emit)
}

// duplexMTP sets the same EnableMTP on prefix warmup and reply so MLX MoE can
// draft. Accepted tokens reach the sentence splitter; draft tokens do not.
func duplexMTP(m *Model) *bool {
	if m == nil {
		return nil
	}
	return voicepipe.DuplexEnableMTP(m.EmbeddedMTP || m.HasMTPCompanion, m.PrimaryFamily())
}
