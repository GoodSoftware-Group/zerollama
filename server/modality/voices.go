package modality

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/types/model"
)

// SpeechVoice is one selectable voice id for /v1/audio/speech (OpenAI voice field).
type SpeechVoice struct {
	ID          string `json:"id"`
	Name        string `json:"name,omitempty"`
	Backend     string `json:"backend,omitempty"`
	Description string `json:"description,omitempty"`
}

// ListSpeechVoices enumerates voices from Piper piper_voice_* keys, optional
// tts_voices_file JSON, tts_default_voice, and a Piper model basename fallback.
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

	if f := PathFor(cfg, "tts_voices_file"); f != "" {
		for _, v := range loadVoicesFile(f) {
			add(v)
		}
	}

	if d := PathFor(cfg, "tts_default_voice"); d != "" {
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

	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
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
