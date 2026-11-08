package server

import (
	"sort"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// speechCatalog is the backend name and voice ids for GET /v1/audio/voices.
// A gen-audio GGUF contributes the voice tensors in its mmproj. LFM3-Audio
// speaks as female or male even when those tensors are absent.
func speechCatalog(m *Model) (backend string, voices []modality.SpeechVoice) {
	if m == nil {
		return "", nil
	}
	voices = modality.ListSpeechVoices(m.Config)
	backend = modality.BackendFor(m.Config, model.ModalitySpeech)
	if !modelMTMDRole(m).Speech {
		return backend, voices
	}
	if backend == "" {
		backend = "llama-server"
	}
	seen := map[string]struct{}{}
	for _, v := range voices {
		seen[v.ID] = struct{}{}
	}
	for _, id := range genVoiceIDs(m) {
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		voices = append(voices, modality.SpeechVoice{ID: id, Name: id, Backend: "llama-server"})
	}
	if rate := genSampleRate(m); rate > 0 {
		for i := range voices {
			if voices[i].Backend == "llama-server" {
				voices[i].SampleRate = rate
			}
		}
	}
	sort.Slice(voices, func(i, j int) bool { return voices[i].ID < voices[j].ID })
	return backend, voices
}

func genSampleRate(m *Model) int {
	if m == nil {
		return 0
	}
	if rate := model.GenAudioSampleRate(m.Config.ModelFamily); rate > 0 {
		return rate
	}
	for _, family := range m.Config.ModelFamilies {
		if rate := model.GenAudioSampleRate(family); rate > 0 {
			return rate
		}
	}
	paths := make([]string, 0, len(m.ProjectorPaths)+1)
	paths = append(paths, m.ProjectorPaths...)
	if m.ModelPath != "" {
		paths = append(paths, m.ModelPath)
	}
	for _, path := range paths {
		if rate := sampleRateInFile(path); rate > 0 {
			return rate
		}
	}
	return 0
}

func sampleRateInFile(path string) int {
	if strings.TrimSpace(path) == "" {
		return 0
	}
	f, err := gguf.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	for _, kv := range f.KeyValues() {
		if rate := model.GenAudioSampleRate(kv.String()); rate > 0 && model.ClassifyMTMDKV(kv.Key, kv.String()).Speech {
			return rate
		}
	}
	return 0
}

func genVoiceIDs(m *Model) []string {
	if m == nil {
		return nil
	}
	seen := map[string]struct{}{}
	var out []string
	add := func(ids []string) {
		for _, id := range ids {
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			out = append(out, id)
		}
	}
	paths := make([]string, 0, len(m.ProjectorPaths)+1)
	paths = append(paths, m.ProjectorPaths...)
	if m.ModelPath != "" {
		paths = append(paths, m.ModelPath)
	}
	for _, path := range paths {
		add(genVoicesInFile(path))
	}
	// Family metadata can name the tower before the file is opened.
	if model.ClassifyMTMDName(m.Config.ModelFamily).Speech {
		add(model.GenVoiceNames(m.Config.ModelFamily, nil))
	}
	for _, family := range m.Config.ModelFamilies {
		if model.ClassifyMTMDName(family).Speech {
			add(model.GenVoiceNames(family, nil))
		}
	}
	sort.Strings(out)
	return out
}

func genVoicesInFile(path string) []string {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	f, err := gguf.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	projector := ""
	for _, kv := range f.KeyValues() {
		role := model.ClassifyMTMDKV(kv.Key, kv.String())
		if role.Speech {
			projector = kv.String()
		}
	}
	var names []string
	for _, ti := range f.TensorInfos() {
		names = append(names, ti.Name)
	}
	return model.GenVoiceNames(projector, names)
}
