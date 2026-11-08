package server

import (
	"net/http"
	"slices"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// modelMTMDRole unions projector metadata from the manifest families and the
// GGUF files. The text architecture (mistral3, lfm3) is not enough; the mmproj
// type is.
func modelMTMDRole(m *Model) model.MTMDRole {
	if m == nil {
		return model.MTMDRole{}
	}
	role := model.ClassifyMTMDName(m.Config.ModelFamily)
	role = role.Merge(model.ClassifyMTMDName(m.PrimaryFamily()))
	for _, f := range m.Config.ModelFamilies {
		role = role.Merge(model.ClassifyMTMDName(f))
	}
	if m.ModelPath != "" {
		if flags, err := ggufCapabilityFlagsFor(m.ModelPath); err == nil {
			role = role.Merge(flags.mtmd)
		}
	}
	for _, p := range m.ProjectorPaths {
		if flags, err := ggufCapabilityFlagsFor(p); err == nil {
			role = role.Merge(flags.mtmd)
		}
	}
	return role
}

// textRouteBlock is the 400 for a text route: chat, generate, embeddings,
// rerank, score, decisions, multidecode, cache warm, and the runtime and
// SGLang chat proxies. An empty string means that route may continue. A
// live-only model stays on GET /v1/realtime. A gen-only projector or a Piper,
// remote-tts, or Music 3 tag stays on POST /v1/audio/speech. A Whisper tag
// with no weight file, and a Parakeet or Nemotron-transcribe GGUF, stay on
// POST /v1/audio/transcriptions. Transcription uses audioRouteBlock so those
// models still transcribe. POST /api/load may still warm any of these models.
func textRouteBlock(m *Model) string {
	if msg := audioRouteBlock(m); msg != "" {
		return msg
	}
	if hostTranscribeOnly(m) || modelUsesLlamaASR(m) {
		return "this model transcribes on POST /v1/audio/transcriptions"
	}
	return ""
}

// audioRouteBlock is the 400 shared with transcription. It refuses live-only
// and speech-only models and leaves Whisper, Parakeet, and Nemotron-transcribe alone.
func audioRouteBlock(m *Model) string {
	role := modelMTMDRole(m)
	if role.StreamOnly {
		return "this model serves live audio on GET /v1/realtime"
	}
	if role.SpeechOnly() || hostSpeechOnly(m) {
		return "this model speaks on POST /v1/audio/speech"
	}
	return ""
}

// hostSpeechOnly is a Piper, kokoro, remote-tts, or Music 3 tag with no weight file.
// A GGUF path is left to the projector role so a text model is not refused.
func hostSpeechOnly(m *Model) bool {
	if m == nil || m.ModelPath != "" || len(m.ProjectorPaths) > 0 {
		return false
	}
	backend := speechBackend(m.Config)
	caps := make([]model.Capability, 0, len(m.Config.Capabilities))
	for _, c := range m.Config.Capabilities {
		caps = append(caps, model.Capability(c))
	}
	return model.HostSpeechOnly(backend, caps)
}

// hostTranscribeOnly is a Whisper tag with no weight file. A GGUF path is
// left to the projector role so a chat model that can hear is not refused.
func hostTranscribeOnly(m *Model) bool {
	if m == nil || m.ModelPath != "" || len(m.ProjectorPaths) > 0 {
		return false
	}
	backend := modality.BackendFor(m.Config, model.ModalityTranscribe)
	if backend == "" && modality.PathFor(m.Config, "whisper_model") != "" {
		backend = model.BackendWhisper
	}
	caps := make([]model.Capability, 0, len(m.Config.Capabilities))
	for _, c := range m.Config.Capabilities {
		caps = append(caps, model.Capability(c))
	}
	return model.HostTranscribeOnly(backend, caps)
}

// textRouteMessage is textRouteBlock for a tag that is already on disk.
// A missing model returns "" so the caller keeps its own not-found path.
func textRouteMessage(name string) string {
	m, err := GetModel(name)
	if err != nil || m == nil {
		return ""
	}
	return textRouteBlock(m)
}

// abortTextRoute writes 400 and stops the handler chain when modelName is a
// live-only, speech-only, or Whisper-only model. A name that does not resolve
// returns false so a proxy can fall through to the handler that reports not-found.
func abortTextRoute(c *gin.Context, modelName string) bool {
	modelRef, err := parseAndValidateModelRef(modelName)
	if err != nil || modelRef.Source == modelSourceCloud {
		return false
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		return false
	}
	msg := textRouteMessage(name.String())
	if msg == "" {
		return false
	}
	c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": msg})
	return true
}

// applyProjectorCapabilities records vision and audio from mmproj files.
// An audio-only projector (Voxtral, Parakeet, LFM3-Audio) is not vision.
// A projector we cannot read stays vision, which is what every mmproj used
// to advertise.
func applyProjectorCapabilities(caps []model.Capability, m *Model) []model.Capability {
	if m == nil || len(m.ProjectorPaths) == 0 {
		return caps
	}
	var role model.MTMDRole
	llamaASR := false
	sawVision := false
	readable := false
	for _, p := range m.ProjectorPaths {
		flags, err := ggufCapabilityFlagsFor(p)
		if err != nil {
			continue
		}
		readable = true
		role = role.Merge(flags.mtmd)
		llamaASR = llamaASR || flags.llamaASR
		sawVision = sawVision || flags.vision
	}
	audioOnly := readable && (role.Audio || llamaASR) && !role.Vision && !sawVision
	if !audioOnly {
		if !slices.Contains(caps, model.CapabilityVision) {
			caps = append(caps, model.CapabilityVision)
		}
		if !slices.Contains(caps, model.CapabilityVideo) {
			caps = append(caps, model.CapabilityVideo)
		}
	}
	if (role.Audio || llamaASR) && !slices.Contains(caps, model.CapabilityAudio) {
		caps = append(caps, model.CapabilityAudio)
	}
	if role.Speech && !slices.Contains(caps, model.CapabilitySpeech) {
		caps = append(caps, model.CapabilitySpeech)
	}
	return caps
}
