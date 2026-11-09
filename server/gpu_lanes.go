package server

import (
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

// seeGPUIndex is ZEROLLAMA_SEE_GPU. Vision ggml runners load only on that
// card so a voice model on the other card can keep decoding.
func seeGPUIndex() (int, bool) {
	raw := strings.TrimSpace(os.Getenv("ZEROLLAMA_SEE_GPU"))
	if raw == "" {
		return 0, false
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

func cudaOrdinal(g ml.DeviceInfo) (int, bool) {
	token := g.FilterID
	if token == "" {
		token = g.ID
	}
	token = strings.TrimPrefix(strings.ToUpper(token), "CUDA")
	n, err := strconv.Atoi(token)
	if err != nil {
		return 0, false
	}
	return n, true
}

// gpusForSeeLane limits a vision model to the see GPU. Text and embedding
// models keep the full device list. With the lane unset, nothing changes.
func gpusForSeeLane(m *Model, gpus []ml.DeviceInfo) []ml.DeviceInfo {
	if m == nil || len(gpus) < 2 || m.CheckCapabilities(model.CapabilityVision) != nil {
		return gpus
	}
	idx, ok := seeGPUIndex()
	if !ok {
		return gpus
	}
	for _, g := range gpus {
		if n, ok := cudaOrdinal(g); ok && n == idx {
			name := m.ShortName
			slog.Info("vision model pinned to see GPU", "model", name, "gpu", g.ID, "ordinal", idx)
			return []ml.DeviceInfo{g}
		}
	}
	if idx < len(gpus) {
		slog.Info("vision model pinned to see GPU", "model", m.ShortName, "gpu", gpus[idx].ID, "ordinal", idx)
		return []ml.DeviceInfo{gpus[idx]}
	}
	return gpus
}
