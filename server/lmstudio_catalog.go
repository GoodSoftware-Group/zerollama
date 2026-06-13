package server

import (
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/lmstudio"
)

// mergeLMStudioModels appends discoverable LM Studio caches to local listings.
// Registered local models win on duplicate names (case-insensitive).
func mergeLMStudioModels(local []api.ListModelResponse) []api.ListModelResponse {
	if !envconfig.LMStudioImport(true) {
		return local
	}

	entries := lmstudio.List()
	if len(entries) == 0 {
		return local
	}

	seen := make(map[string]struct{}, len(local)+len(entries))
	out := make([]api.ListModelResponse, 0, len(local)+len(entries))
	for _, m := range local {
		k := strings.ToLower(m.Model)
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, m)
	}

	now := time.Now()
	for _, e := range entries {
		k := strings.ToLower(e.Name)
		if _, ok := seen[k]; ok {
			continue
		}
		if !envconfig.LMStudioListAll() {
			if ok, free, need, err := lmstudio.HasDiskForImport(e); err != nil {
				slog.Debug("lm studio catalog skip: disk check failed", "model", e.Name, "error", err)
				if lmstudio.ImportCopyBytes(e) > 0 {
					continue
				}
			} else if !ok {
				slog.Debug("lm studio catalog skip: insufficient disk for import copy",
					"model", e.Name, "need_bytes", need, "free_bytes", free)
				continue
			}
		}
		seen[k] = struct{}{}
		modified := e.Modified
		if modified.IsZero() {
			modified = now
		}
		out = append(out, api.ListModelResponse{
			Name:        e.Name,
			Model:       e.Name,
			RemoteModel: e.Dir,
			RemoteHost:  lmstudio.RemoteHost(),
			Size:        e.Size,
			ModifiedAt:  modified,
			Details: api.ModelDetails{
				Format: e.Format,
				Family: "lmstudio",
			},
		})
	}
	return out
}
