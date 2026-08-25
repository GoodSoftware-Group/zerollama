// Flash-MoE pull-time sidecar auto-extract (M16 open item — see docs/flash-moe.md).
//
// Why opt-in and pull-only: extraction reads the entire routed-expert GGUF
// payload and can take minutes on 100GB+ MoE models. Operators who never
// asked for slot-bank streaming must not have `pull` silently balloon in
// time or disk use. ZEROLLAMA_FLASH_MOE_AUTO_EXTRACT=1 opts in; failures are
// logged only, never fail the pull itself (same pattern as
// EnrichManifestAfterPull).
package server

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// AutoExtractFlashMoESidecarAfterPull extracts a Flash-MoE sidecar for a
// freshly pulled/created MoE GGUF tag when ZEROLLAMA_FLASH_MOE_AUTO_EXTRACT=1
// and no sidecar is configured yet. On success the sidecar path is written
// into the manifest's `moe_sidecar` param so later loads need no env var —
// mirrors how EnrichManifestAfterPull persists GGUF-derived hints.
func AutoExtractFlashMoESidecarAfterPull(name model.Name, fn func(api.ProgressResponse)) {
	if !envconfig.FlashMoEAutoExtract() {
		return
	}
	entry, ok, err := discover.FlashMoEEntryForName(name)
	if err != nil || !ok {
		return
	}
	if entry.SidecarReady {
		slog.Debug("flash-moe sidecar already present", "model", name.DisplayShortest(), "sidecar", entry.Sidecar)
		return
	}
	if entry.GGUFPath == "" || entry.Sidecar == "" {
		return
	}

	if fn != nil {
		fn(api.ProgressResponse{Status: "extracting flash-moe sidecar"})
	}

	start := time.Now()
	if err := extractFlashMoESidecar(entry.GGUFPath, entry.Sidecar); err != nil {
		slog.Warn("flash-moe sidecar auto-extract failed", "model", name.DisplayShortest(), "error", err)
		return
	}
	slog.Info("flash-moe sidecar extracted", "model", name.DisplayShortest(), "sidecar", entry.Sidecar, "elapsed", time.Since(start))

	if err := writeFlashMoESidecarParam(name, entry.Sidecar); err != nil {
		slog.Warn("flash-moe sidecar param write failed", "model", name.DisplayShortest(), "error", err)
	}
}

// extractFlashMoESidecar shells out to anemll's flashmoe_sidecar.py extract —
// same tool docs/flash-moe.md and flash_moe_extract_sidecar.sh use. Why
// reimplement the wrapper here instead of exec'ing the .sh: pull runs inside
// the Go daemon with no shell/script dependency guarantees on non-dev hosts.
func extractFlashMoESidecar(ggufPath, outDir string) error {
	repo := envconfig.FlashMoERepo()
	script := filepath.Join(repo, "tools", "flashmoe-sidecar", "flashmoe_sidecar.py")
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("anemll-flash-llama.cpp sidecar tool missing at %s: %w", script, err)
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, "python3", script, "extract",
		"--model", ggufPath,
		"--out-dir", outDir,
		"--force",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("flashmoe_sidecar.py extract: %w: %s", err, truncateOutput(out))
	}
	return nil
}

func truncateOutput(b []byte) string {
	const max = 2000
	if len(b) > max {
		return string(b[len(b)-max:])
	}
	return string(b)
}

// writeFlashMoESidecarParam persists moe_sidecar into the manifest params
// layer, mirroring RepairModel's write path (readManifestParams ->
// setParameters -> createConfigLayer -> WriteManifest) without pulling in
// the full GGUF-guess repair diff.
func writeFlashMoESidecarParam(name model.Name, sidecar string) error {
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return err
	}
	cfg, err := readManifestConfig(mf)
	if err != nil {
		return err
	}
	params := readManifestParams(mf.Layers)
	if existing, ok := params["moe_sidecar"].(string); ok && existing == sidecar {
		return nil
	}
	params["moe_sidecar"] = sidecar

	layers := slices.Clone(mf.Layers)
	layers, err = setParameters(layers, params)
	if err != nil {
		return err
	}
	configLayer, err := createConfigLayer(cfg)
	if err != nil {
		return err
	}
	return manifest.WriteManifest(name, *configLayer, layers)
}
