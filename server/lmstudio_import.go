package server

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/lmstudio"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	typesmodel "github.com/ollama/ollama/types/model"
)

// tryImportFromLMStudio registers the model from a matching LM Studio GGUF
// directory when present, avoiding a registry blob download. It returns true if
// the model was created locally.
func tryImportFromLMStudio(ctx context.Context, n typesmodel.Name, deleteMap map[string]struct{}, fn func(api.ProgressResponse)) (bool, error) {
	if !envconfig.LMStudioImport(true) {
		return false, nil
	}

	select {
	case <-ctx.Done():
		return false, ctx.Err()
	default:
	}

	dir, ok := lmstudio.MatchDir(n)
	if !ok {
		return false, nil
	}

	slog.Info("using LM Studio model files instead of registry download", "model", n.DisplayShortest(), "dir", dir)
	fn(api.ProgressResponse{Status: fmt.Sprintf("using LM Studio cache: %s", dir)})

	files, err := parser.FileDigestMap(dir)
	if err != nil {
		slog.Debug("lm studio import skipped", "dir", dir, "reason", err)
		return false, nil
	}

	if err := stageFilesToBlobs(files); err != nil {
		return false, err
	}

	if err := createFromLMStudioFiles(n, files, fn); err != nil {
		return false, err
	}

	if !envconfig.NoPrune() && len(deleteMap) > 0 {
		fn(api.ProgressResponse{Status: "removing unused layers"})
		if err := deleteUnusedLayers(deleteMap); err != nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("couldn't remove unused layers: %v", err)})
		}
	}

	fn(api.ProgressResponse{Status: "success"})
	return true, nil
}

func stageFilesToBlobs(files map[string]string) error {
	for src, digest := range files {
		dst, err := manifest.BlobsPath(digest)
		if err != nil {
			return err
		}
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		if err := createLink(src, dst); err != nil {
			return fmt.Errorf("stage blob for %s: %w", src, err)
		}
	}
	return nil
}

func createFromLMStudioFiles(name typesmodel.Name, files map[string]string, fn func(api.ProgressResponse)) error {
	config := &typesmodel.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: typesmodel.RootFS{
			Type: "layers",
		},
	}

	r := api.CreateRequest{}
	baseLayers, err := convertModelFromFiles(files, nil, false, fn)
	if err != nil {
		return err
	}

	return createModel(r, name, baseLayers, config, fn)
}
