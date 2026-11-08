package server

import (
	"context"
	"log/slog"
	"os"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// warmKokoroPhrases synthesizes the short Kokoro acknowledgements for each
// installed Kokoro tag whose ONNX and voices files are present.
// ZEROLLAMA_KOKORO_WARM=0 skips the pass. A remote-tts kokoro tag is left alone.
func (s *Server) warmKokoroPhrases(ctx context.Context) {
	if !envconfig.KokoroPhraseWarm() {
		return
	}
	ms, err := manifest.Manifests(true)
	if err != nil {
		slog.Debug("kokoro phrase warm skipped", "error", err)
		return
	}
	n := 0
	for name := range ms {
		if ctx.Err() != nil {
			return
		}
		m, err := GetModel(name.String())
		if err != nil || m == nil || !kokoroSpeechConfig(m) {
			continue
		}
		modelPath := modality.KokoroModelPath(m.Config)
		voicesPath := modality.KokoroVoicesPath(m.Config)
		if !regularFile(modelPath) || !regularFile(voicesPath) {
			continue
		}
		modality.WarmKokoroPhrases(ctx, m.Config, "")
		n++
	}
	if n > 0 {
		slog.Info("kokoro phrase warm finished", "models", n)
	}
}

func kokoroSpeechConfig(m *Model) bool {
	if m == nil || m.ModelPath != "" || len(m.ProjectorPaths) > 0 {
		return false
	}
	return modality.ResolveSpeechBackend(m.Config) == model.BackendKokoro
}

func regularFile(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}
