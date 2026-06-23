package server

import (
	"context"
	"errors"
	"log/slog"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/lmstudio"
	"github.com/ollama/ollama/internal/modelhealth"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/sync/singleflight"
)

var lmStudioSyncGroup singleflight.Group

// SyncLMStudioModels imports discoverable LM Studio caches into OLLAMA_MODELS and
// re-imports local tags whose manifest blobs are missing. Why on list/serve: operators
// expect `zerollama list` to reflect LM Studio without a separate pull per model, and
// broken MLX/GGUF registrations (missing blobs) should self-heal from cache.
func SyncLMStudioModels(ctx context.Context) error {
	_, err, _ := lmStudioSyncGroup.Do("sync", func() (any, error) {
		return nil, syncLMStudioModelsOnce(ctx)
	})
	return err
}

// RepairLMStudioModelIfNeeded re-imports one tag from LM Studio when blobs are missing.
// Returns true when a new import completed successfully.
func RepairLMStudioModelIfNeeded(ctx context.Context, name string) (bool, error) {
	if !envconfig.LMStudioImport(true) {
		return false, nil
	}

	n := model.ParseName(name)
	if !n.IsValid() {
		return false, nil
	}

	key := "repair:" + n.DisplayShortest()
	v, err, _ := lmStudioSyncGroup.Do(key, func() (any, error) {
		return repairLMStudioModelOnce(ctx, n)
	})
	if err != nil {
		return false, err
	}
	repaired, _ := v.(bool)
	return repaired, nil
}

func syncLMStudioModelsOnce(ctx context.Context) error {
	if !envconfig.LMStudioImport(true) {
		return nil
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	entries := lmstudio.List()
	if len(entries) == 0 {
		return nil
	}

	var synced, orphaned int
	for _, e := range entries {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if !lmStudioEntrySyncable(e) {
			continue
		}

		n := model.ParseName(e.Name)
		if !n.IsValid() {
			continue
		}

		if !needsLMStudioSync(n) {
			continue
		}

		repaired, err := repairLMStudioModelOnce(ctx, n)
		if err != nil {
			slog.Warn("lm studio sync import failed", "model", e.Name, "error", err)
			continue
		}
		if repaired {
			synced++
			continue
		}
		if report, rerr := modelhealth.CheckName(n.String()); rerr == nil && report.Status == modelhealth.StatusOrphaned {
			orphaned++
			slog.Warn("lm studio model orphaned (missing blobs, no cache source)",
				"model", e.Name,
				"fix", report.FixHint)
		}
	}

	if synced > 0 {
		slog.Info("lm studio sync complete", "imported", synced)
	}
	if orphaned > 0 {
		slog.Info("lm studio sync found orphaned registrations", "count", orphaned,
			"hint", "zerollama doctor --models or zerollama rm <name>")
	}
	return nil
}

func repairLMStudioModelOnce(ctx context.Context, n model.Name) (bool, error) {
	if !needsLMStudioSync(n) {
		return false, nil
	}

	if _, _, ok := lmstudio.MatchSelection(n); !ok {
		return false, nil
	}

	deleteMap := buildPullDeleteMap(n)
	imported, err := tryImportFromLMStudio(ctx, n, deleteMap, lmStudioSyncProgress)
	if err != nil {
		return false, err
	}
	if imported {
		slog.Info("lm studio sync imported", "model", n.DisplayShortest())
	}
	return imported, nil
}

// lmStudioEntrySyncable mirrors catalog visibility: skip MLX imports when OLLAMA_MODELS
// lacks headroom unless OLLAMA_LMSTUDIO_LIST_ALL=1 (pull/sync still enforce on attempt).
func lmStudioEntrySyncable(e lmstudio.Entry) bool {
	if envconfig.LMStudioListAll() {
		return true
	}
	ok, _, need, err := lmstudio.HasDiskForImport(e)
	if need == 0 {
		return true
	}
	if err != nil {
		slog.Debug("lm studio sync skip: disk check failed", "model", e.Name, "error", err)
		return false
	}
	return ok
}

func needsLMStudioSync(n model.Name) bool {
	mf, err := manifest.ParseNamedManifest(n)
	if errors.Is(err, os.ErrNotExist) {
		return true
	}
	if err != nil {
		slog.Debug("lm studio sync: bad existing manifest", "model", n.DisplayShortest(), "error", err)
		return true
	}
	return modelhealth.HasMissingBlobs(mf)
}

func lmStudioSyncProgress(p api.ProgressResponse) {
	if p.Status == "" {
		return
	}
	slog.Info("lm studio sync", "status", p.Status)
}

func manifestHasMissingBlobs(mf *manifest.Manifest) bool {
	return modelhealth.HasMissingBlobs(mf)
}
