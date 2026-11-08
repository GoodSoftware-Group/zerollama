package modality

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/server/modality/kokoro"
	"github.com/ollama/ollama/types/model"
)

// SpeechVoice is one selectable voice id for /v1/audio/speech (OpenAI voice field).
type SpeechVoice struct {
	ID          string `json:"id"`
	Name        string `json:"name,omitempty"`
	Backend     string `json:"backend,omitempty"`
	Description string `json:"description,omitempty"`
	// SampleRate is the playback rate in Hz for a llama-server gen projector.
	// PCM from that model is little-endian float32 at this rate.
	SampleRate int `json:"sample_rate,omitempty"`
}

// ListSpeechVoices enumerates voices from Piper piper_voice_* keys, Kokoro
// voices-v1.0.bin zip members, optional tts_voices_file JSON, tts_default_voice,
// and a Piper model basename fallback.
func ListSpeechVoices(cfg model.ConfigV2) []SpeechVoice {
	backend := BackendFor(cfg, model.ModalitySpeech)
	seen := map[string]struct{}{}
	var out []SpeechVoice

	add := func(v SpeechVoice) {
		id := strings.TrimSpace(v.ID)
		if id == "" {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		if v.Backend == "" {
			v.Backend = backend
		}
		if v.Name == "" {
			v.Name = id
		}
		out = append(out, v)
	}

	if cfg.BackendPaths != nil {
		for k, p := range cfg.BackendPaths {
			if !strings.HasPrefix(k, "piper_voice_") {
				continue
			}
			id := strings.TrimPrefix(k, "piper_voice_")
			desc := ""
			if p != "" {
				desc = filepath.Base(p)
			}
			add(SpeechVoice{ID: id, Backend: model.BackendPiper, Description: desc})
		}
	}

	kokoroList := backend == model.BackendKokoro || (backend == "" && PathFor(cfg, "piper_model") == "" && KokoroPathsSet(cfg))

	if f := PathFor(cfg, "tts_voices_file"); f != "" {
		for _, v := range loadVoicesFile(f) {
			if kokoroList && !kokoro.EnglishVoice(v.ID) {
				continue
			}
			add(v)
		}
	}

	if d := PathFor(cfg, "tts_default_voice"); d != "" && (!kokoroList || kokoro.EnglishVoice(d)) {
		add(SpeechVoice{ID: d, Backend: backend})
	}

	if backend == model.BackendPiper || backend == "" {
		if p := PathFor(cfg, "piper_model"); p != "" {
			base := strings.TrimSuffix(filepath.Base(p), ".onnx")
			if base != "" && base != filepath.Base(p) {
				add(SpeechVoice{ID: sanitizeVoiceKey(base), Name: base, Backend: model.BackendPiper, Description: filepath.Base(p)})
			}
		}
	}

	// sample_rate on this struct is the llama-server float32 rate. Kokoro
	// duplex clips carry their own int16 rate, so leave it unset.
	// Fleet Kokoro env vars do not attach voices to a tag that left speech empty.
	if kokoroList {
		raw := kokoro.VoiceIDsFromBin(KokoroVoicesPath(cfg))
		var ids []string
		for _, id := range raw {
			if kokoro.EnglishVoice(id) {
				ids = append(ids, id)
			}
		}
		for _, id := range ids {
			add(SpeechVoice{ID: id, Backend: model.BackendKokoro})
		}
		// A zip of only non-English ids is not an English catalog.
		// af_heart is the fallback when the file cannot be read.
		if len(ids) == 0 && len(raw) == 0 {
			if fallback := KokoroDefaultVoice(cfg); kokoro.EnglishVoice(fallback) {
				add(SpeechVoice{ID: fallback, Backend: model.BackendKokoro})
			}
		}
	}

	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// KokoroVoiceCatalogAuthoritative is true when the voices zip or tts_voices_file
// produced at least one English id. Non-English Kokoro prefixes do not count.
// The hardcoded af_heart fallback is not authoritative.
func KokoroVoiceCatalogAuthoritative(cfg model.ConfigV2) bool {
	if PathFor(cfg, "tts_voices_file") != "" {
		for _, v := range loadVoicesFile(PathFor(cfg, "tts_voices_file")) {
			if kokoro.EnglishVoice(v.ID) {
				return true
			}
		}
	}
	for _, id := range kokoro.VoiceIDsFromBin(KokoroVoicesPath(cfg)) {
		if kokoro.EnglishVoice(id) {
			return true
		}
	}
	return false
}

func loadVoicesFile(path string) []SpeechVoice {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var list []SpeechVoice
	if err := json.Unmarshal(raw, &list); err == nil && len(list) > 0 {
		return list
	}
	// Also accept {"voices":[...]}
	var wrap struct {
		Voices []SpeechVoice `json:"voices"`
	}
	if err := json.Unmarshal(raw, &wrap); err == nil {
		return wrap.Voices
	}
	return nil
}
