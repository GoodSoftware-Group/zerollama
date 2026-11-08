package modality

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestResolveSpeechBackendKokoroPathsNotEnv(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "kokoro.onnx")
	voices := filepath.Join(dir, "voices.bin")
	if err := os.WriteFile(modelPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(voices)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	w, err := zw.Create("af_heart.npy")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = w.Write([]byte{1})
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_KOKORO_MODEL", modelPath)
	t.Setenv("OLLAMA_KOKORO_VOICES", voices)
	t.Setenv("OLLAMA_TTS_URL", "http://127.0.0.1:9/speech")

	remote := model.ConfigV2{}
	if got := ResolveSpeechBackend(remote); got != model.BackendRemoteTTS {
		t.Fatalf("env kokoro files selected %q", got)
	}
	if got := ListSpeechVoices(remote); len(got) != 0 {
		t.Fatalf("env attached voices %#v", got)
	}

	withURL := model.ConfigV2{BackendPaths: map[string]string{"tts_url": "http://127.0.0.1:9/speech"}}
	if got := ResolveSpeechBackend(withURL); got != model.BackendRemoteTTS {
		t.Fatalf("tts_url selected %q", got)
	}

	kokoroCfg := model.ConfigV2{BackendPaths: map[string]string{
		"kokoro_model":  modelPath,
		"kokoro_voices": voices,
	}}
	if got := ResolveSpeechBackend(kokoroCfg); got != model.BackendKokoro {
		t.Fatalf("paths selected %q", got)
	}
	listed := ListSpeechVoices(kokoroCfg)
	if len(listed) != 1 || listed[0].ID != "af_heart" || listed[0].SampleRate != 0 {
		t.Fatalf("voices %#v", listed)
	}

	both := model.ConfigV2{BackendPaths: map[string]string{
		"piper_model":   dir + "/voice.onnx",
		"kokoro_model":  modelPath,
		"kokoro_voices": voices,
	}}
	if got := ResolveSpeechBackend(both); got != model.BackendPiper {
		t.Fatalf("piper should win, got %q", got)
	}

	explicit := kokoroCfg
	explicit.ModalityBackends = map[string]string{model.ModalitySpeech: model.BackendRemoteTTS}
	if got := ResolveSpeechBackend(explicit); got != model.BackendRemoteTTS {
		t.Fatalf("explicit backend selected %q", got)
	}
}
