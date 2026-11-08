package modality

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/types/model"
)

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
