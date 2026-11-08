package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/server/modality/whispercli"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// SpeechHandler serves POST /v1/audio/speech for Piper, remote-tts, Music 3,
// and a llama-server gen-audio GGUF (LFM3, Qwen3-TTS, Pocket TTS, Soprano,
// Drama3, Chatterbox tower). Music 3 returns 202 JSON, not WAV bytes.
func (s *Server) SpeechHandler(c *gin.Context) {
	v, ok := c.Get(middleware.CtxKeySpeechRequest)
	if !ok {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "missing speech request"})
		return
	}
	req := v.(openai.SpeechCreateRequest)

	if served, err := applyModelAlias(c, req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		req.Model = served
	}
	if routed, rerr := s.routeForSurface(c, req.Model, speechRouteSignals(req, autoroute.SurfaceSpeech)); rerr != nil {
		writeRouterError(c, rerr.dec, rerr.err)
		return
	} else {
		req.Model = routed
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cloud speech is not supported on this endpoint"})
		return
	}

	name := modelRef.Name
	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist), os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "remote models are not supported on this endpoint"})
		return
	}

	role := modelMTMDRole(m)
	if msg := model.SpeechBlock(role); msg != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": msg})
		return
	}
	if hostTranscribeOnly(m) || modelUsesLlamaASR(m) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "this model transcribes on POST /v1/audio/transcriptions"})
		return
	}
	silence, serr := piperphrase.SentenceSilence(req.SentenceSilence)
	if serr != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": serr.Error()})
		return
	}
	if silence != "" && !piperSpeechBackend(m) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "sentence_silence requires a piper model"})
		return
	}
	noise, nerr := piperphrase.NoiseScale(req.NoiseScale)
	if nerr != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": nerr.Error()})
		return
	}
	if noise != "" && !piperSpeechBackend(m) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "noise_scale requires a piper model"})
		return
	}
	width, werr := piperphrase.NoiseW(req.NoiseW)
	if werr != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": werr.Error()})
		return
	}
	if width != "" && !piperSpeechBackend(m) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "noise_w requires a piper model"})
		return
	}
	speaker, spkerr := piperphrase.SpeakerID(req.SpeakerID)
	if spkerr != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": spkerr.Error()})
		return
	}
	if speaker != "" && !piperSpeechBackend(m) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "speaker_id requires a piper model"})
		return
	}
	phonemes, perr := piperphrase.PhonemeSilence(req.PhonemeSilence)
	if perr != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": perr.Error()})
		return
	}
	if phonemes != "" && !piperSpeechBackend(m) && !kokoroSpeechBackend(m) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "phoneme_silence requires a piper or kokoro model"})
		return
	}
	if role.Speech && (m.ModelPath != "" || len(m.ProjectorPaths) > 0) {
		s.speakLlamaTTS(c, name.String(), req, genSampleRate(m))
		return
	}

	if err := m.CheckCapabilities(model.CapabilitySpeech); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support speech (set modality_backends.speech=piper|kokoro|remote-tts and capability speech)", req.Model)})
		return
	}

	backend := speechBackend(m.Config)

	var data []byte
	var contentType string
	var upstreamSeconds string
	switch backend {
	case model.BackendMusic3:
		s.queueMusic3(c, req)
		return
	case model.BackendKokoro:
		if _, err := model.SpeechContainer(req.ResponseFormat, false); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if !modality.KokoroFillersReady(m.Config, req.Voice) {
			cfg, voice := m.Config, req.Voice
			go modality.WarmKokoroPhrases(context.WithoutCancel(c.Request.Context()), cfg, voice)
		} else {
			go modality.WarmKokoro(m.Config)
		}
		if req.Stream {
			ct, rate, r, cached, serr := modality.SpeechKokoroStreamReader(c.Request.Context(), m.Config, req.Input, req.Voice, req.Lang, req.Speed, phonemes, req.Emotion)
			if serr != nil {
				writeKokoroError(c, serr)
				return
			}
			defer r.Close()
			if cached {
				c.Header("X-Speech-Cache", "hit")
			}
			writeSpeechStream(c, ct, "pcm", r, rate)
			return
		}
		var cached bool
		data, contentType, cached, err = modality.SpeechKokoro(c.Request.Context(), m.Config, req.Input, req.Voice, req.Lang, req.Speed, phonemes, req.Emotion)
		if err == nil && cached {
			c.Header("X-Speech-Cache", "hit")
		}
	case model.BackendPiper:
		if _, err := model.SpeechContainer(req.ResponseFormat, false); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if !modality.PiperFillersReady(m.Config, req.Voice) {
			cfg, voice := m.Config, req.Voice
			go modality.WarmPiperPhrases(context.WithoutCancel(c.Request.Context()), cfg, voice)
		}
		if req.Stream {
			ct, r, cached, serr := modality.SpeechPiperStream(c.Request.Context(), m.Config, req.Input, req.Voice, req.Speed, silence, noise, width, speaker, phonemes)
			if serr == nil {
				defer r.Close()
				if cached {
					c.Header("X-Speech-Cache", "hit")
				}
				writeSpeechStream(c, ct, "wav", r, 0)
				return
			}
			if !errors.Is(serr, modality.ErrPiperStreamUnavailable) {
				c.JSON(http.StatusInternalServerError, gin.H{"error": serr.Error()})
				return
			}
		}
		var cached bool
		data, contentType, cached, err = modality.SpeechPiper(c.Request.Context(), m.Config, req.Input, req.Voice, req.Speed, silence, noise, width, speaker, phonemes)
		if err == nil && cached {
			c.Header("X-Speech-Cache", "hit")
		}
	case model.BackendRemoteTTS:
		if req.Stream {
			ct, seconds, r, serr := modality.SpeechRemoteStream(c.Request.Context(), m.Config, req.Model, req.Input, req.Voice, req.ResponseFormat, req.Emotion, req.Speed)
			if serr != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": serr.Error()})
				return
			}
			defer r.Close()
			if seconds != "" {
				c.Header("X-Audio-Seconds", seconds)
			}
			writeSpeechStream(c, ct, speechFileExt(ct), r, 0)
			return
		}
		data, contentType, upstreamSeconds, err = modality.SpeechRemote(c.Request.Context(), m.Config, req.Model, req.Input, req.Voice, req.ResponseFormat, req.Emotion, req.Speed)
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("unsupported speech backend %q (want piper, kokoro, remote-tts, or music3)", backend)})
		return
	}
	if err != nil {
		if backend == model.BackendKokoro {
			writeKokoroError(c, err)
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if req.ResponseFormat != "" && req.ResponseFormat != "wav" && !strings.Contains(contentType, req.ResponseFormat) {
		slog.Debug("speech response_format may differ from upstream content-type", "format", req.ResponseFormat, "content_type", contentType)
	}
	writeSpeechBody(c, contentType, speechFileExt(contentType), data, 0, upstreamSeconds)
}

func (s *Server) speakLlamaTTS(c *gin.Context, modelName string, req openai.SpeechCreateRequest, sampleRate int) {
	if strings.TrimSpace(req.Input) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "input is required"})
		return
	}
	if strings.TrimSpace(req.Voice) != "" && strings.TrimSpace(req.SpeakerRef) != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "voice and speaker_ref cannot be used together"})
		return
	}
	lang, err := openai.CleanSpeechLang(req.Lang)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	speakerRef, err := openai.DecodeSpeakerRef(req.SpeakerRef)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if req.Seed != nil && *req.Seed > math.MaxUint32 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "seed must fit in 32 bits"})
		return
	}
	if req.MaxNewTokens != nil && (*req.MaxNewTokens < -1 || *req.MaxNewTokens > math.MaxInt32) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "max_new_tokens must be between -1 and 2147483647"})
		return
	}
	if err := openai.CleanSpeechTopK(req.TopK); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := openai.CleanSpeechTopP(req.TopP); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := openai.CleanSpeechRepeatPenalty(req.RepeatPenalty); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	format, err := model.SpeechContainer(req.ResponseFormat, true)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	runner, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), modelName, []model.Capability{model.CapabilitySpeech}, nil, nil, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, modelName, err)
		return
	}
	defer releaseQoS()
	speaker, ok := runner.(llm.LlamaSpeaker)
	if !ok {
		c.JSON(http.StatusConflict, gin.H{"error": "speech requires llama-server; unload this model and retry (ZEROLLAMA_LLAMA_SERVER must not be 0)"})
		return
	}
	contentType, audio, err := speaker.Speak(c.Request.Context(), req.Input, req.Voice, format, lang, speakerRef, req.Stream, req.Seed, req.MaxNewTokens, llm.SpeechSample{
		TopK:          req.TopK,
		TopP:          req.TopP,
		RepeatPenalty: req.RepeatPenalty,
	})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	defer audio.Close()
	if format == "pcm" {
		contentType = model.PCMContentType(contentType, sampleRate)
	}
	ext := "wav"
	if format == "pcm" {
		ext = "pcm"
	}
	if req.Stream {
		writeSpeechStream(c, contentType, ext, audio, sampleRate)
		return
	}
	buf, err := io.ReadAll(audio)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	writeSpeechBody(c, contentType, ext, buf, sampleRate, "")
}

func speechFileExt(contentType string) string {
	if strings.Contains(contentType, "mpeg") || strings.Contains(contentType, "mp3") {
		return "mp3"
	}
	if strings.Contains(contentType, "ogg") || strings.Contains(contentType, "opus") {
		return "ogg"
	}
	return "wav"
}

func writeSpeechBody(c *gin.Context, contentType, ext string, data []byte, rate int, upstreamSeconds string) {
	c.Header("Content-Disposition", fmt.Sprintf(`attachment; filename="speech.%s"`, ext))
	if sec, ok := whispercli.AudioSeconds(contentType, data, rate); ok {
		c.Header("X-Audio-Seconds", strconv.FormatFloat(sec, 'f', -1, 64))
	} else if upstreamSeconds != "" {
		c.Header("X-Audio-Seconds", upstreamSeconds)
	}
	c.Data(http.StatusOK, contentType, data)
}

func writeSpeechStream(c *gin.Context, contentType, ext string, r io.Reader, rate int) {
	c.Header("Trailer", "X-Audio-Seconds")
	c.Header("Content-Type", contentType)
	c.Header("Content-Disposition", fmt.Sprintf(`attachment; filename="speech.%s"`, ext))
	c.Status(http.StatusOK)
	meter := whispercli.NewStreamMeter(contentType, rate)
	buf := make([]byte, 32*1024)
	flusher, _ := c.Writer.(http.Flusher)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			if _, werr := c.Writer.Write(buf[:n]); werr != nil {
				return
			}
			meter.Add(buf[:n])
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			if err != io.EOF {
				slog.Info("speech stream ended", "err", err)
				return
			}
			if sec, ok := meter.Seconds(); ok {
				c.Header("X-Audio-Seconds", strconv.FormatFloat(sec, 'f', -1, 64))
			}
			return
		}
	}
}

// speechBackend resolves modality_backends.speech with path heuristics.
// An empty backend selects kokoro only when both kokoro path keys are set
// and piper_model is not. OLLAMA_KOKORO_MODEL does not select a backend.
func speechBackend(cfg model.ConfigV2) string {
	return modality.ResolveSpeechBackend(cfg)
}

func writeKokoroError(c *gin.Context, err error) {
	status := http.StatusInternalServerError
	if modality.IsKokoroRequest(err) {
		status = http.StatusBadRequest
	}
	c.JSON(status, gin.H{"error": err.Error()})
}

// voiceRefuse is the 400 for GET /v1/audio/voices?model= when the tag cannot speak.
// A speech model never reaches this. Live-only, file-audio, Whisper, and
// Parakeet tags get the same route message as the other audio handlers.
func piperSpeechBackend(m *Model) bool {
	if m == nil {
		return false
	}
	role := modelMTMDRole(m)
	if role.Speech && (m.ModelPath != "" || len(m.ProjectorPaths) > 0) {
		return false
	}
	return speechBackend(m.Config) == model.BackendPiper
}

func kokoroSpeechBackend(m *Model) bool {
	if m == nil {
		return false
	}
	role := modelMTMDRole(m)
	if role.Speech && (m.ModelPath != "" || len(m.ProjectorPaths) > 0) {
		return false
	}
	return speechBackend(m.Config) == model.BackendKokoro
}

func voiceRefuse(m *Model) string {
	if m == nil {
		return ""
	}
	if msg := model.SpeechBlock(modelMTMDRole(m)); msg != "" {
		return msg
	}
	return textRouteBlock(m)
}

// VoicesHandler serves GET /v1/audio/voices — lists selectable voice ids for speech models.
// Query model= optional; when set, returns voices for that tag only.
func (s *Server) VoicesHandler(c *gin.Context) {
	q := strings.TrimSpace(c.Query("model"))
	type modelVoices struct {
		Model   string                 `json:"model"`
		Backend string                 `json:"backend,omitempty"`
		Voices  []modality.SpeechVoice `json:"voices"`
	}
	var models []modelVoices

	if q != "" {
		modelRef, err := parseAndValidateModelRef(q)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
			return
		}
		name, err := getExistingName(modelRef.Name)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", q)})
			return
		}
		m, err := GetModel(name.String())
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", q)})
			return
		}
		if err := m.CheckCapabilities(model.CapabilitySpeech); err != nil {
			if msg := voiceRefuse(m); msg != "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": msg})
				return
			}
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q is not a speech model", q)})
			return
		}
		backend, voices := speechCatalog(m)
		models = append(models, modelVoices{
			Model:   name.DisplayShortest(),
			Backend: backend,
			Voices:  voices,
		})
		c.JSON(http.StatusOK, gin.H{"object": "list", "data": models})
		return
	}

	ms, err := manifest.Manifests(true)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	for name := range ms {
		m, err := GetModel(name.String())
		if err != nil {
			continue
		}
		if err := m.CheckCapabilities(model.CapabilitySpeech); err != nil {
			continue
		}
		backend, voices := speechCatalog(m)
		models = append(models, modelVoices{
			Model:   name.DisplayShortest(),
			Backend: backend,
			Voices:  voices,
		})
	}
	c.JSON(http.StatusOK, gin.H{"object": "list", "data": models})
}
