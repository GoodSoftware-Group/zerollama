package server

import (
	"errors"

	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/server/modality/transcript"
	"github.com/ollama/ollama/types/model"
)

var errEmptyASRTranscript = errors.New("llama-server transcription returned no text")

// modelUsesLlamaASR reports whether transcription for m must go to llama-server
// (Parakeet or Nemotron-transcribe) instead of whisper.cpp or a chat fall-through.
func modelUsesLlamaASR(m *Model) bool {
	if m == nil {
		return false
	}
	if modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendLlamaASR {
		return true
	}
	if model.IsLlamaASRName(m.Config.ModelFamily) || model.IsLlamaASRName(m.PrimaryFamily()) {
		return true
	}
	for _, f := range m.Config.ModelFamilies {
		if model.IsLlamaASRName(f) {
			return true
		}
	}
	if m.ModelPath != "" {
		if flags, err := ggufCapabilityFlagsFor(m.ModelPath); err == nil && flags.llamaASR {
			return true
		}
	}
	for _, p := range m.ProjectorPaths {
		if flags, err := ggufCapabilityFlagsFor(p); err == nil && flags.llamaASR {
			return true
		}
	}
	return false
}

// asrTranscriptText reads the text field llama-server puts on
// transcript.text.done (and on an OpenAI-shaped {"text":"..."} body).
// usage is the token accounting on that event, when present.
func asrTranscriptText(body []byte) (string, *openai.TranscriptionUsage, error) {
	text, usage, err := transcript.ParseLlama(body)
	if err != nil {
		if errors.Is(err, transcript.ErrEmpty) {
			return "", nil, errEmptyASRTranscript
		}
		return "", nil, err
	}
	if usage == nil {
		return text, nil, nil
	}
	out := &openai.TranscriptionUsage{
		Type:         usage.Type,
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
		TotalTokens:  usage.TotalTokens,
	}
	if usage.CachedTokens != nil {
		out.InputTokensDetails = &openai.TranscriptionInputTokensDetails{CachedTokens: *usage.CachedTokens}
	}
	return text, out, nil
}
