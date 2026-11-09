package server

import (
	"context"
	"log/slog"
	"os"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/server/vram"
)

// ggmlWeightsFit reports whether weightBytes plus a small KV/scratch slack
// fits in the free VRAM already reported by the devices. Runtime occupancy is
// included in FreeMemory, so a true result means the ggml load can proceed
// without stopping llama-server.
func ggmlWeightsFit(gpus []ml.DeviceInfo, weightBytes uint64) bool {
	if weightBytes == 0 || len(gpus) == 0 {
		return false
	}
	const slack = uint64(2 << 30)
	need := weightBytes + slack
	var sum uint64
	for _, g := range gpus {
		free := g.FreeMemory
		min := g.MinimumMemory()
		if min > 0 {
			if free <= min {
				continue
			}
			free -= min
		}
		sum += free
	}
	return sum >= need
}

// releaseRuntimeIfTight stops the Python llama-server only when the GGUF does
// not fit in current free VRAM. A missing file still releases, matching the
// old unconditional handoff.
func (s *Scheduler) releaseRuntimeIfTight(ctx context.Context, gpus []ml.DeviceInfo, modelPath string) {
	st, err := os.Stat(modelPath)
	if err != nil || st.Size() <= 0 || !ggmlWeightsFit(gpus, uint64(st.Size())) {
		slog.Info("releasing runtime VRAM before ggml load", "path", modelPath, "err", err)
		vram.PrepareForLegacyRunner(ctx)
		return
	}
	slog.Info("keeping runtime resident; ggml weights fit in free VRAM", "path", modelPath, "bytes", st.Size())
}
