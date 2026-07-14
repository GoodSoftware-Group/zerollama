package modality

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestNormalizeSpeechURL(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"http://host:8090", "http://host:8090/v1/audio/speech"},
		{"http://host:8090/", "http://host:8090/v1/audio/speech"},
		{"http://host:8090/v1/audio/speech", "http://host:8090/v1/audio/speech"},
	}
	for _, tt := range tests {
		if got := normalizeSpeechURL(tt.in); got != tt.want {
			t.Fatalf("normalizeSpeechURL(%q)=%q want %q", tt.in, got, tt.want)
		}
	}
}

func TestSpeechRemote(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/speech" {
			t.Errorf("path=%s", r.URL.Path)
		}
		var req remoteSpeechBody
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
		}
		if req.Input != "hello" || req.Voice != "puppet" || req.Emotion != "excited" {
			t.Errorf("req=%+v", req)
		}
		if req.Model != "chatterbox" {
			t.Errorf("upstream model=%q", req.Model)
		}
		if r.Header.Get("X-TTS-Ref-Audio") != "/tmp/ref.wav" {
			t.Errorf("ref header=%q", r.Header.Get("X-TTS-Ref-Audio"))
		}
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("RIFF"))
	}))
	defer srv.Close()

	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalitySpeech: model.BackendRemoteTTS},
		BackendPaths: map[string]string{
			"tts_url":            srv.URL,
			"tts_upstream_model": "chatterbox",
			"tts_ref_audio":      "/tmp/ref.wav",
		},
	}
	data, ct, err := SpeechRemote(t.Context(), cfg, "ignored", "hello", "puppet", "", "excited", nil)
	if err != nil {
		t.Fatal(err)
	}
	if ct != "audio/wav" || string(data) != "RIFF" {
		t.Fatalf("ct=%q data=%q", ct, data)
	}
}

func TestListSpeechVoices(t *testing.T) {
	dir := t.TempDir()
	voicesPath := filepath.Join(dir, "voices.json")
	if err := os.WriteFile(voicesPath, []byte(`{"voices":[{"id":"af_heart","name":"Heart"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalitySpeech: model.BackendRemoteTTS},
		BackendPaths: map[string]string{
			"tts_voices_file":   voicesPath,
			"tts_default_voice": "af_heart",
			"piper_voice_amy":   "/tmp/amy.onnx",
		},
	}
	got := ListSpeechVoices(cfg)
	ids := map[string]bool{}
	for _, v := range got {
		ids[v.ID] = true
	}
	if !ids["af_heart"] || !ids["amy"] {
		t.Fatalf("voices=%v", got)
	}
}
