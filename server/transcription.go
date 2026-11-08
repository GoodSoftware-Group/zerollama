package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/server/modality/qwen3asr"
	"github.com/ollama/ollama/server/modality/transcript"
	"github.com/ollama/ollama/server/modality/whispercli"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// TranscriptionHandler serves POST /v1/audio/transcriptions and POST /v1/audio/translations.
// A translations request asks a whisper model for English text.
// whisper.cpp runs when modality_backends.transcribe=whisper.
// Parakeet, Nemotron-transcribe, and mtmd file-audio projectors (Voxtral Small,
// Qwen2.5-Omni, Ultravox, Phi-4, LFM3-Audio) are scheduled on llama-server.
// voxtral_rt has no file transcription. Other audio models fall through to ChatHandler.
func (s *Server) TranscriptionHandler(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.Request.Body = io.NopCloser(bytes.NewReader(body))

	var chatReq api.ChatRequest
	if err := json.Unmarshal(body, &chatReq); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if served, err := applyModelAlias(c, chatReq.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else if routed, rerr := s.routeForSurface(c, served, autoroute.Signals{Surface: autoroute.SurfaceTranscribe, HasAudio: true}); rerr != nil {
		writeRouterError(c, rerr.dec, rerr.err)
		return
	} else if routed != chatReq.Model {
		// Rewrite the body too: non-whisper models fall through to ChatHandler with it.
		chatReq.Model = routed
		if rewritten, err := replaceJSONModelField(body, routed); err == nil {
			body = rewritten
			c.Request.Body = io.NopCloser(bytes.NewReader(body))
		}
	}

	modelRef, err := parseAndValidateModelRef(chatReq.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.Request.Body = io.NopCloser(bytes.NewReader(body))
		s.ChatHandler(c)
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
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", chatReq.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if transcriptionStream(c) && !modelUsesLlamaASR(m) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "streaming transcription requires a llama-server ASR model"})
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.Request.Body = io.NopCloser(bytes.NewReader(body))
		s.ChatHandler(c)
		return
	}

	role := modelMTMDRole(m)
	if msg := audioRouteBlock(m); msg != "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": msg})
		return
	}
	translate := strings.HasSuffix(c.Request.URL.Path, "/translations")
	llamaASR := modelUsesLlamaASR(m)
	if model.TranslationRejected(llamaASR, translate) || (translate && modality.UsesQwen3ASR(m.Config)) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "translation requires a whisper model or a chat model that can hear"})
		return
	}
	if transcriptionLogprobs(c) && (llamaASR || modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "logprobs require a whisper model"})
		return
	}
	if transcriptionDiarize(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "diarize requires a whisper model"})
		return
	}
	if transcriptionTinyDiarize(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "tinydiarize requires a whisper model"})
		return
	}
	if transcriptionSuppressNST(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "suppress_nst requires a whisper model"})
		return
	}
	if transcriptionNoFallback(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "no_fallback requires a whisper model"})
		return
	}
	if transcriptionTemperatureInc(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "temperature_inc requires a whisper model"})
		return
	}
	if transcriptionEntropyThold(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "entropy_thold requires a whisper model"})
		return
	}
	if transcriptionLogprobThold(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "logprob_thold requires a whisper model"})
		return
	}
	if transcriptionNoSpeechThold(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "no_speech_thold requires a whisper model"})
		return
	}
	if transcriptionBeamSize(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "beam_size requires a whisper model"})
		return
	}
	if transcriptionBestOf(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "best_of requires a whisper model"})
		return
	}
	if transcriptionCarryInitial(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "carry_initial_prompt requires a whisper model"})
		return
	}
	if transcriptionMaxLen(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "max_len requires a whisper model"})
		return
	}
	if transcriptionSplitOnWord(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "split_on_word requires a whisper model"})
		return
	}
	if transcriptionSplitOnWord(c) && transcriptionMaxLen(c) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "split_on_word requires max_len"})
		return
	}
	if transcriptionWordThold(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "word_thold requires a whisper model"})
		return
	}
	if transcriptionMaxContext(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "max_context requires a whisper model"})
		return
	}
	if transcriptionAudioCtx(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "audio_ctx requires a whisper model"})
		return
	}
	if transcriptionOffsetT(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "offset_t requires a whisper model"})
		return
	}
	if transcriptionDurationMS(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "duration_ms requires a whisper model"})
		return
	}
	if transcriptionSuppressRegex(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "suppress_regex requires a whisper model"})
		return
	}
	if transcriptionOffsetN(c) != 0 && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "offset_n requires a whisper model"})
		return
	}
	if transcriptionNoTimestamps(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "no_timestamps requires a whisper model"})
		return
	}
	if transcriptionGrammar(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "grammar requires a whisper model"})
		return
	}
	if transcriptionGrammarPenalty(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "grammar_penalty requires a whisper model"})
		return
	}
	if transcriptionPrintSpecial(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "print_special requires a whisper model"})
		return
	}
	if transcriptionDetectLanguage(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "detect_language requires a whisper model"})
		return
	}
	if transcriptionDTW(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "dtw requires a whisper model"})
		return
	}
	if transcriptionVAD(c) && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad requires a whisper model"})
		return
	}
	if transcriptionVADThreshold(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_threshold requires a whisper model"})
		return
	}
	if transcriptionVADThreshold(c) != "" && !transcriptionVAD(c) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_threshold requires vad=true"})
		return
	}
	if transcriptionVADMinSpeechMS(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_min_speech_duration_ms requires a whisper model"})
		return
	}
	if transcriptionVADMinSpeechMS(c) != "" && !transcriptionVAD(c) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_min_speech_duration_ms requires vad=true"})
		return
	}
	if transcriptionVADMinSilenceMS(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_min_silence_duration_ms requires a whisper model"})
		return
	}
	if transcriptionVADMinSilenceMS(c) != "" && !transcriptionVAD(c) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_min_silence_duration_ms requires vad=true"})
		return
	}
	if transcriptionVADMaxSpeechS(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_max_speech_duration_s requires a whisper model"})
		return
	}
	if transcriptionVADMaxSpeechS(c) != "" && !transcriptionVAD(c) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_max_speech_duration_s requires vad=true"})
		return
	}
	if transcriptionVADSpeechPadMS(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_speech_pad_ms requires a whisper model"})
		return
	}
	if transcriptionVADSpeechPadMS(c) != "" && !transcriptionVAD(c) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_speech_pad_ms requires vad=true"})
		return
	}
	if transcriptionVADSamplesOverlap(c) != "" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_samples_overlap requires a whisper model"})
		return
	}
	if transcriptionVADSamplesOverlap(c) != "" && !transcriptionVAD(c) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad_samples_overlap requires vad=true"})
		return
	}
	rf, _ := c.Get(middleware.CtxKeyTranscriptionResponseFormat)
	format, _ := rf.(string)
	if format == "diarized_json" && modality.BackendFor(m.Config, model.ModalityTranscribe) != model.BackendWhisper {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "diarized_json requires a whisper model"})
		return
	}
	if transcriptionTinyDiarize(c) && (transcriptionDiarize(c) || format == "diarized_json") {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "tinydiarize cannot be combined with diarize"})
		return
	}
	if _, capped := transcriptionMaxTokens(c); capped && modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendWhisper && !llamaASR {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "max_tokens requires a llama-server ASR model or a chat model that can hear"})
		return
	}
	if model.FileTranscriptionOnLlama(role, llamaASR, translate) {
		s.transcribeLlamaASR(c, m, &chatReq)
		return
	}

	if modality.UsesQwen3ASR(m.Config) {
		audio, err := audioBytesFromChat(&chatReq)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		langVal, _ := c.Get(middleware.CtxKeyTranscriptionLanguage)
		language, _ := langVal.(string)
		text, err := modality.TranscribeQwen3(c.Request.Context(), m.Config, audio, language)
		if errors.Is(err, qwen3asr.ErrUnavailable) {
			// Resident process could not start; use the Whisper CLI with the same clip.
			qwenErr := err
			orig, _ := c.Get(middleware.CtxKeyTranscriptionOriginalFilename)
			origName, _ := orig.(string)
			code, lerr := whispercli.LanguageCode(language)
			if lerr != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": lerr.Error()})
				return
			}
			tr, werr := modality.TranscribeWhisper(c.Request.Context(), m.Config, audio, origName, code, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
			if werr != nil {
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("%v (whisper fallback: %v)", qwenErr, werr)})
				return
			}
			text = tr.Text
			language = code
			if language == "" || language == "auto" {
				language = tr.Language
			}
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		usage := withWavSeconds(nil, audio)
		contentType, body, err := openai.TranscriptionResult(text, format, language, "transcribe", usage, nil, nil)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.Data(http.StatusOK, contentType, body)
		return
	}

	if modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendWhisper {
		audio, err := audioBytesFromChat(&chatReq)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		orig, _ := c.Get(middleware.CtxKeyTranscriptionOriginalFilename)
		origName, _ := orig.(string)
		langVal, _ := c.Get(middleware.CtxKeyTranscriptionLanguage)
		language, _ := langVal.(string)
		code, lerr := whispercli.LanguageCode(language)
		if lerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": lerr.Error()})
			return
		}
		language = code
		promptVal, _ := c.Get(middleware.CtxKeyTranscriptionPrompt)
		prompt, _ := promptVal.(string)
		tempVal, _ := c.Get(middleware.CtxKeyTranscriptionTemperature)
		temperature, _ := tempVal.(string)

		wordVal, _ := c.Get(middleware.CtxKeyTranscriptionWords)
		words, _ := wordVal.(bool)
		logprobs := transcriptionLogprobs(c)
		diarize := transcriptionDiarize(c) || format == "diarized_json"
		detectLanguage := transcriptionDetectLanguage(c)
		vadModel := ""
		if transcriptionVAD(c) {
			vadModel = modality.VADModelPath(m.Config)
			if vadModel == "" {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "vad requires backend_paths.vad_model or OLLAMA_WHISPER_VAD_MODEL"})
				return
			}
		}
		tinyDiarize := transcriptionTinyDiarize(c)
		timed := format == "verbose_json" || format == "srt" || format == "vtt" || format == "diarized_json" || logprobs || detectLanguage || language == "auto" || tinyDiarize
		// verbose_json and include=logprobs ask for -ojf so token probabilities are present.
		tr, err := modality.TranscribeWhisper(c.Request.Context(), m.Config, audio, origName, language, prompt, temperature, transcriptionTemperatureInc(c), transcriptionEntropyThold(c), transcriptionLogprobThold(c), transcriptionNoSpeechThold(c), transcriptionBeamSize(c), transcriptionBestOf(c), transcriptionMaxLen(c), transcriptionWordThold(c), transcriptionMaxContext(c), transcriptionAudioCtx(c), transcriptionOffsetT(c), transcriptionDurationMS(c), transcriptionSuppressRegex(c), transcriptionGrammar(c), transcriptionGrammarRule(c), transcriptionGrammarPenalty(c), transcriptionDTW(c), vadModel, transcriptionVADThreshold(c), transcriptionVADMinSpeechMS(c), transcriptionVADMinSilenceMS(c), transcriptionVADMaxSpeechS(c), transcriptionVADSpeechPadMS(c), transcriptionVADSamplesOverlap(c), timed, words || format == "verbose_json" || logprobs, translate, diarize, transcriptionSuppressNST(c), transcriptionNoFallback(c), transcriptionCarryInitial(c), transcriptionNoTimestamps(c), transcriptionPrintSpecial(c), detectLanguage, tinyDiarize, transcriptionSplitOnWord(c))
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if detectLanguage {
			language = tr.Language
		} else if translate {
			language = "en"
		} else if language == "" || language == "auto" {
			if tr.Language != "" {
				language = tr.Language
			}
		}
		task := "transcribe"
		if translate {
			task = "translate"
		}
		if format == "srt" || format == "vtt" {
			segs := tr.Segments
			if len(segs) == 0 && tr.Text != "" {
				sec, _ := whispercli.WavSeconds(audio)
				segs = []whispercli.Segment{whispercli.CueSpan(tr.Text, sec)}
			}
			payload := whispercli.WriteSRTOffset(segs, transcriptionOffsetN(c))
			contentType := "text/plain"
			if format == "vtt" {
				payload = whispercli.WriteVTT(segs)
				contentType = "text/vtt"
			}
			c.Data(http.StatusOK, contentType, []byte(payload))
			return
		}
		var contentType string
		var body []byte
		usage := withWavSeconds(nil, audio)
		langP := languageProbability(tr, language)
		if format == "diarized_json" {
			contentType, body, err = diarizedTranscription(tr, language, task, usage, langP)
		} else if format == "verbose_json" && len(tr.Segments) > 0 {
			segs := make([]openai.TranscriptionSegment, len(tr.Segments))
			var segmentEnd float64
			for i, seg := range tr.Segments {
				segs[i] = openai.TranscriptionSegment{
					ID: i, Start: seg.Start, End: seg.End, Text: seg.Text,
					Speaker:         seg.Speaker,
					SpeakerTurnNext: seg.SpeakerTurnNext,
					Tokens:          seg.Tokens,
					AvgLogprob:      seg.AvgLogprob, NoSpeechProb: seg.NoSpeechProb,
				}
				if seg.End > segmentEnd {
					segmentEnd = seg.End
				}
			}
			measured := 0.0
			if usage != nil {
				measured = usage.Seconds
			}
			duration := whispercli.KnownDuration(measured, segmentEnd)
			var wordOut []openai.TranscriptionWord
			if words {
				wordOut = make([]openai.TranscriptionWord, 0, len(tr.Words))
				for _, w := range tr.Words {
					wordOut = append(wordOut, openai.TranscriptionWord{Word: w.Word, Start: w.Start, End: w.End})
				}
			}
			contentType, body, err = openai.TranscriptionResultSegments(tr.Text, language, segs, duration, wordOut, task, usage, transcriptionLogprobList(logprobs, tr.Logprobs), langP)
		} else {
			contentType, body, err = openai.TranscriptionResult(tr.Text, format, language, task, usage, transcriptionLogprobList(logprobs, tr.Logprobs), langP)
		}
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.Data(http.StatusOK, contentType, body)
		return
	}

	c.Request.Body = io.NopCloser(bytes.NewReader(body))
	s.ChatHandler(c)
}

func languageProbability(tr whispercli.Transcript, language string) *float64 {
	if tr.LanguageProbability == nil || language == "" || language != tr.DetectedLanguage {
		return nil
	}
	return tr.LanguageProbability
}

func diarizedTranscription(tr whispercli.Transcript, language, task string, usage *openai.TranscriptionUsage, languageProbability *float64) (string, []byte, error) {
	var segmentEnd float64
	var spoken strings.Builder
	segs := make([]openai.TranscriptionDiarizedSegment, 0, len(tr.Segments))
	for _, seg := range tr.Segments {
		text := strings.TrimSpace(whispercli.StripSpeakerPrefix(seg.Text))
		if text == "" {
			continue
		}
		if spoken.Len() > 0 {
			spoken.WriteByte(' ')
		}
		spoken.WriteString(text)
		segs = append(segs, openai.TranscriptionDiarizedSegment{
			ID:      fmt.Sprintf("seg_%d", len(segs)),
			Type:    "transcript.text.segment",
			Speaker: seg.Speaker,
			Start:   seg.Start,
			End:     seg.End,
			Text:    text,
		})
		if seg.End > segmentEnd {
			segmentEnd = seg.End
		}
	}
	full := spoken.String()
	if full == "" {
		full = strings.TrimSpace(whispercli.StripSpeakerPrefix(tr.Text))
	}
	measured := 0.0
	if usage != nil {
		measured = usage.Seconds
	}
	duration := whispercli.KnownDuration(measured, segmentEnd)
	return openai.TranscriptionResultDiarized(full, language, task, segs, duration, usage, languageProbability)
}

func (s *Server) transcribeLlamaASR(c *gin.Context, m *Model, chatReq *api.ChatRequest) {
	if transcriptionLogprobs(c) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "logprobs require a whisper model"})
		return
	}
	if m.ModelPath == "" && len(m.ProjectorPaths) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "this audio model requires a GGUF; create it from the file"})
		return
	}
	audio, err := audioBytesFromChat(chatReq)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	orig, _ := c.Get(middleware.CtxKeyTranscriptionOriginalFilename)
	origName, _ := orig.(string)
	langVal, _ := c.Get(middleware.CtxKeyTranscriptionLanguage)
	language, _ := langVal.(string)
	promptVal, _ := c.Get(middleware.CtxKeyTranscriptionPrompt)
	prompt, _ := promptVal.(string)
	tempVal, _ := c.Get(middleware.CtxKeyTranscriptionTemperature)
	temperature, _ := tempVal.(string)
	rf, _ := c.Get(middleware.CtxKeyTranscriptionResponseFormat)
	format, _ := rf.(string)
	stream := transcriptionStream(c)
	if stream && format != "" && format != "json" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "stream requires response_format json"})
		return
	}

	runner, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), m.Name, []model.Capability{model.CapabilityAudio}, nil, nil, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, m.Name, err)
		return
	}
	defer releaseQoS()

	asr, ok := runner.(llm.LlamaASRTranscriber)
	if !ok {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "ASR model did not load on llama-server"})
		return
	}
	maxTokens := ""
	if n, ok := transcriptionMaxTokens(c); ok {
		maxTokens = strconv.Itoa(n)
	}
	contentType, audioBody, err := asr.TranscribeAudio(c.Request.Context(), audio, origName, language, prompt, temperature, maxTokens, stream)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	defer audioBody.Close()
	if stream {
		if contentType == "" {
			contentType = "text/event-stream"
		}
		sec, _ := whispercli.WavSeconds(audio)
		writeEventStream(c, contentType, audioBody, sec)
		return
	}
	body, err := io.ReadAll(audioBody)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	text, usage, err := asrTranscriptText(body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	usage = withWavSeconds(usage, audio)
	contentType, out, err := openai.TranscriptionResult(text, format, language, "transcribe", usage, nil, nil)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.Data(http.StatusOK, contentType, out)
}

// withWavSeconds records the WAV length on usage. A nil usage becomes type
// duration. An existing token usage keeps its type and gains seconds.
func withWavSeconds(usage *openai.TranscriptionUsage, audio []byte) *openai.TranscriptionUsage {
	sec, ok := whispercli.WavSeconds(audio)
	if !ok {
		return usage
	}
	if usage == nil {
		return &openai.TranscriptionUsage{Type: "duration", Seconds: sec}
	}
	usage.Seconds = sec
	return usage
}

func transcriptionMaxTokens(c *gin.Context) (int, bool) {
	v, ok := c.Get(middleware.CtxKeyTranscriptionMaxTokens)
	if !ok {
		return 0, false
	}
	n, ok := v.(int)
	if !ok || n < 1 {
		return 0, false
	}
	return n, true
}

func transcriptionGrammar(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionGrammar)
	s, _ := v.(string)
	return s
}

func transcriptionGrammarRule(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionGrammarRule)
	s, _ := v.(string)
	return s
}

func transcriptionVADSamplesOverlap(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionVADSamplesOverlap)
	s, _ := v.(string)
	return s
}

func transcriptionVADSpeechPadMS(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionVADSpeechPadMS)
	s, _ := v.(string)
	return s
}

func transcriptionVADMaxSpeechS(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionVADMaxSpeechS)
	s, _ := v.(string)
	return s
}

func transcriptionVADMinSilenceMS(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionVADMinSilenceMS)
	s, _ := v.(string)
	return s
}

func transcriptionVADMinSpeechMS(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionVADMinSpeechMS)
	s, _ := v.(string)
	return s
}

func transcriptionVADThreshold(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionVADThreshold)
	s, _ := v.(string)
	return s
}

func transcriptionVAD(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionVAD)
	on, _ := v.(bool)
	return on
}

func transcriptionDTW(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionDTW)
	s, _ := v.(string)
	return s
}

func transcriptionDetectLanguage(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionDetectLanguage)
	on, _ := v.(bool)
	return on
}

func transcriptionPrintSpecial(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionPrintSpecial)
	on, _ := v.(bool)
	return on
}

func transcriptionGrammarPenalty(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionGrammarPenalty)
	s, _ := v.(string)
	return s
}

func transcriptionNoTimestamps(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionNoTimestamps)
	on, _ := v.(bool)
	return on
}

func transcriptionOffsetN(c *gin.Context) int {
	v, _ := c.Get(middleware.CtxKeyTranscriptionOffsetN)
	n, _ := v.(int)
	return n
}

func transcriptionSuppressRegex(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionSuppressRegex)
	s, _ := v.(string)
	return s
}

func transcriptionDurationMS(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionDurationMS)
	s, _ := v.(string)
	return s
}

func transcriptionOffsetT(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionOffsetT)
	s, _ := v.(string)
	return s
}

func transcriptionAudioCtx(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionAudioCtx)
	s, _ := v.(string)
	return s
}

func transcriptionMaxContext(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionMaxContext)
	s, _ := v.(string)
	return s
}

func transcriptionWordThold(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionWordThold)
	s, _ := v.(string)
	return s
}

func transcriptionMaxLen(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionMaxLen)
	s, _ := v.(string)
	return s
}

func transcriptionCarryInitial(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionCarryInitial)
	on, _ := v.(bool)
	return on
}

func transcriptionBestOf(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionBestOf)
	s, _ := v.(string)
	return s
}

func transcriptionBeamSize(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionBeamSize)
	s, _ := v.(string)
	return s
}

func transcriptionNoSpeechThold(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionNoSpeechThold)
	s, _ := v.(string)
	return s
}

func transcriptionLogprobThold(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionLogprobThold)
	s, _ := v.(string)
	return s
}

func transcriptionEntropyThold(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionEntropyThold)
	s, _ := v.(string)
	return s
}

func transcriptionTemperatureInc(c *gin.Context) string {
	v, _ := c.Get(middleware.CtxKeyTranscriptionTemperatureInc)
	s, _ := v.(string)
	return s
}

func transcriptionNoFallback(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionNoFallback)
	on, _ := v.(bool)
	return on
}

func transcriptionSuppressNST(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionSuppressNST)
	on, _ := v.(bool)
	return on
}

func transcriptionSplitOnWord(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionSplitOnWord)
	on, _ := v.(bool)
	return on
}

func transcriptionTinyDiarize(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionTinyDiarize)
	on, _ := v.(bool)
	return on
}

func transcriptionDiarize(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionDiarize)
	on, _ := v.(bool)
	return on
}

func transcriptionLogprobs(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionLogprobs)
	on, _ := v.(bool)
	return on
}

func transcriptionLogprobList(on bool, src []whispercli.Logprob) []openai.TranscriptionLogprob {
	if !on {
		return nil
	}
	out := make([]openai.TranscriptionLogprob, len(src))
	for i, item := range src {
		b := make([]int, len(item.Token))
		for j := 0; j < len(item.Token); j++ {
			b[j] = int(item.Token[j])
		}
		out[i] = openai.TranscriptionLogprob{Token: item.Token, Logprob: item.Logprob, Bytes: b}
	}
	return out
}

func transcriptionStream(c *gin.Context) bool {
	v, _ := c.Get(middleware.CtxKeyTranscriptionStream)
	on, _ := v.(bool)
	return on
}

func writeEventStream(c *gin.Context, contentType string, r io.Reader, seconds float64) {
	c.Header("Content-Type", contentType)
	c.Header("Cache-Control", "no-cache")
	c.Status(http.StatusOK)
	flusher, _ := c.Writer.(http.Flusher)
	_ = transcript.NameEvents(&flushWriter{w: c.Writer, flush: flusher}, r, seconds)
}

type flushWriter struct {
	w     io.Writer
	flush http.Flusher
}

func (f *flushWriter) Write(p []byte) (int, error) {
	n, err := f.w.Write(p)
	if n > 0 && f.flush != nil {
		f.flush.Flush()
	}
	return n, err
}

func audioBytesFromChat(req *api.ChatRequest) ([]byte, error) {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" && len(req.Messages[i].Images) > 0 {
			return append([]byte(nil), req.Messages[i].Images[0]...), nil
		}
	}
	return nil, errors.New("no audio attachment in transcription request")
}
