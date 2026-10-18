package server

import (
	"context"

	"github.com/ollama/ollama/api"
)

// PullModelOpts is the storage-serve entry point for POST /v1/pull.
// Why a thin exported wrapper: registryOptions is package-private; storaged
// injects this via cmd without importing inference internals.
func PullModelOpts(ctx context.Context, name string, insecure bool, hfSource string, fn func(api.ProgressResponse)) error {
	return PullModel(ctx, name, &registryOptions{Insecure: insecure, HFSource: hfSource}, fn)
}
