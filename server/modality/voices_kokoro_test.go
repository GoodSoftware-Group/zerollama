package modality

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestListSpeechVoicesKokoroZip(t *testing.T) {
	dir := t.TempDir()
	voices := filepath.Join(dir, "voices.bin")
	modelPath := filepath.Join(dir, "kokoro.onnx")
	_ = os.WriteFile(modelPath, []byte("x"), 0o644)
	f, err := os.Create(voices)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for _, name := range []string{"af_heart.npy", "bf_emma.npy", "zf_xiaobei.npy"} {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte{1})
	}
	_ = zw.Close()
	_ = f.Close()

	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalitySpeech: model.BackendKokoro},
		BackendPaths: map[string]string{
			"kokoro_model":  modelPath,
			"kokoro_voices": voices,
		},
	}
	got := ListSpeechVoices(cfg)
	if len(got) != 2 || got[0].ID != "af_heart" || got[1].ID != "bf_emma" {
		t.Fatalf("got %#v", got)
	}
	for _, v := range got {
		if v.SampleRate != 0 {
			t.Fatalf("sample_rate is the gen-projector field, got %#v", got)
		}
	}
	if !KokoroVoiceCatalogAuthoritative(cfg) {
		t.Fatal("expected authoritative")
	}
}

func TestKokoroFallbackNotAuthoritative(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "kokoro.onnx")
	voices := filepath.Join(dir, "voices.bin")
	_ = os.WriteFile(modelPath, []byte("x"), 0o644)
	_ = os.WriteFile(voices, []byte("not-a-zip"), 0o644)
	cfg := model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalitySpeech: model.BackendKokoro},
		BackendPaths: map[string]string{
			"kokoro_model":  modelPath,
			"kokoro_voices": voices,
		},
	}
	got := ListSpeechVoices(cfg)
	if len(got) != 1 || got[0].ID != "af_heart" {
		t.Fatalf("got %#v", got)
	}
	if KokoroVoiceCatalogAuthoritative(cfg) {
		t.Fatal("fallback must not be authoritative")
	}
}
