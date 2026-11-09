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
	// Integrated FreeMemory is the same pool the runtime already uses.
	// A weight-sized fit would leave llama-server resident and then load ggml beside it.
	discrete := make([]ml.DeviceInfo, 0, len(gpus))
	for _, g := range gpus {
		if !g.Integrated {
			discrete = append(discrete, g)
		}
	}
	if len(discrete) == 0 {
		return false
	}
	gpus = discrete
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

// releaseRuntimeIfTight stops the Python llama-server before a ggml load.
// weightsFit is logged only. KeepRuntimeResident does not let a fit skip the handoff.
func (s *Scheduler) releaseRuntimeIfTight(ctx context.Context, gpus []ml.DeviceInfo, modelPath string) {
	var weightsFit bool
	if st, err := os.Stat(modelPath); err == nil && st.Size() > 0 {
		weightsFit = ggmlWeightsFit(gpus, uint64(st.Size()))
	}
	if KeepRuntimeResident(weightsFit) {
		slog.Info("keeping runtime resident; ggml weights fit in free VRAM", "path", modelPath)
		return
	}
	slog.Info("releasing runtime VRAM before ggml load", "path", modelPath, "weights_fit", weightsFit)
	vram.PrepareForLegacyRunner(ctx)
}
