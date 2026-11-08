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

// warmPiperPhrases synthesizes the short Piper acknowledgements for each
// installed Piper tag whose ONNX file is present. ZEROLLAMA_PIPER_WARM=0
// skips the pass. It does not load a GGUF runner.
func (s *Server) warmPiperPhrases(ctx context.Context) {
	if !envconfig.PiperPhraseWarm() {
		return
	}
	ms, err := manifest.Manifests(true)
	if err != nil {
		slog.Debug("piper phrase warm skipped", "error", err)
		return
	}
	n := 0
	for name := range ms {
		if ctx.Err() != nil {
			return
		}
		m, err := GetModel(name.String())
		if err != nil || m == nil || !piperSpeechConfig(m) {
			continue
		}
		path := modality.PathFor(m.Config, "piper_model")
		if path == "" {
			continue
		}
		st, err := os.Stat(path)
		if err != nil || st.IsDir() {
			continue
		}
		modality.WarmPiperPhrases(ctx, m.Config, "")
		n++
	}
	if n > 0 {
		slog.Info("piper phrase warm finished", "models", n)
	}
}

func piperSpeechConfig(m *Model) bool {
	if m == nil || m.ModelPath != "" || len(m.ProjectorPaths) > 0 {
		return false
	}
	if modality.BackendFor(m.Config, model.ModalitySpeech) == model.BackendPiper {
		return true
	}
	return modality.BackendFor(m.Config, model.ModalitySpeech) == "" && modality.PathFor(m.Config, "piper_model") != ""
}
