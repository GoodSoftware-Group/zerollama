package server

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/lmstudio"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	typesmodel "github.com/ollama/ollama/types/model"
	xcreate "github.com/ollama/ollama/x/create"
	xcreateclient "github.com/ollama/ollama/x/create/client"
)

// tryImportFromLMStudio registers the model from a matching LM Studio cache
// directory (GGUF or safetensors) when present, avoiding a registry blob
// download. It returns true if the model was created locally.
func tryImportFromLMStudio(ctx context.Context, n typesmodel.Name, deleteMap map[string]struct{}, fn func(api.ProgressResponse)) (bool, error) {
	if !envconfig.LMStudioImport(true) {
		return false, nil
	}

	select {
	case <-ctx.Done():
		return false, ctx.Err()
	default:
	}

	dir, weightFile, ok := lmstudio.MatchSelection(n)
	if !ok {
		return false, nil
	}

	slog.Info("using LM Studio model files instead of registry download", "model", n.DisplayShortest(), "dir", dir)
	fn(api.ProgressResponse{Status: fmt.Sprintf("using LM Studio cache: %s", dir)})

	// MLX / HF safetensors trees: register native tensor blobs (no GGUF conversion).
	// GGUF trees (and legacy safetensors without config.json) use blob staging + convert.
	if lmStudioUseNativeSafetensorsImport(dir) {
		if err := xcreateclient.ImportSafetensorsFromDirectory(n.String(), dir, func(status string) {
			fn(api.ProgressResponse{Status: status})
		}); err != nil {
			return false, fmt.Errorf("lm studio safetensors import: %w", err)
		}
	} else {
		files, err := parser.FileDigestMap(dir)
		if err != nil {
			slog.Debug("lm studio import skipped", "dir", dir, "reason", err)
			return false, nil
		}
		if weightFile != "" {
			files = filterLMStudioImportFiles(files, weightFile)
		}

		if err := stageFilesToBlobs(files); err != nil {
			return false, err
		}

		// Safetensors conversion expects map keys relative to the model root, not
		// absolute paths from FileDigestMap.
		if err := createFromLMStudioFiles(n, dir, files, fn); err != nil {
			return false, err
		}
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

func createFromLMStudioFiles(name typesmodel.Name, dir string, files map[string]string, fn func(api.ProgressResponse)) error {
	files = relativePathsInDir(dir, files)
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

// lmStudioUseNativeSafetensorsImport reports whether LM Studio cache dir should use
// CreateSafetensorsModel (MLX/HF layout) instead of GGUF/safetensors→GGUF conversion.
func lmStudioUseNativeSafetensorsImport(dir string) bool {
	return xcreate.IsSafetensorsModelDir(dir)
}

func relativePathsInDir(dir string, files map[string]string) map[string]string {
	out := make(map[string]string, len(files))
	for path, digest := range files {
		rel, err := filepath.Rel(dir, path)
		if err != nil || strings.HasPrefix(rel, "..") {
			rel = filepath.Base(path)
		}
		out[filepath.ToSlash(rel)] = digest
	}
	return out
}

func filterLMStudioImportFiles(files map[string]string, weightFile string) map[string]string {
	out := make(map[string]string, len(files))
	for path, digest := range files {
		base := strings.ToLower(filepath.Base(path))
		switch {
		case base == strings.ToLower(weightFile):
			out[path] = digest
		case strings.HasPrefix(base, "mmproj"):
			out[path] = digest
		case strings.HasSuffix(base, ".json"), base == "tokenizer.model":
			out[path] = digest
		case strings.Contains(path, string(filepath.Separator)) && strings.HasSuffix(base, ".json"):
			out[path] = digest
		}
	}
	return out
}
