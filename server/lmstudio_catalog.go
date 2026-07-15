package server

import (
	"encoding/json"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/lmstudio"
)

// mergeLMStudioModels appends discoverable LM Studio caches to local listings.
// Registered local models win on duplicate names (case-insensitive).
//
// Why disk checks: MLX safetensors import repacks ~full model size into OLLAMA_MODELS;
// GGUF/legacy safetensors symlink in place. Listing unimportable MLX models wastes
// operator time — hide unless OLLAMA_LMSTUDIO_LIST_ALL=1 (pull still enforces space).
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
				if lmstudio.ImportCopyBytes(e) > 0 {
					slog.Warn("lm studio catalog skip: disk check failed for MLX import",
						"model", e.Name, "error", err)
					continue
				}
				slog.Debug("lm studio catalog skip: disk check failed", "model", e.Name, "error", err)
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
		details := api.ModelDetails{
			Format: e.Format,
			Family: "lmstudio",
		}
		enrichModelDetailsFromConfigPath(&details, e.ConfigPath())
		out = append(out, api.ListModelResponse{
			Name:        e.Name,
			Model:       e.Name,
			RemoteModel: e.Dir,
			RemoteHost:  lmstudio.RemoteHost(),
			Size:        e.Size,
			// Stock ollama clients panic on empty digests (digest[:12]).
			Digest:     listCatalogDigest("lmstudio:" + e.Name),
			ModifiedAt: modified,
			Details:    details,
		})
	}
	return out
}

func enrichModelDetailsFromConfigPath(details *api.ModelDetails, configPath string) {
	if configPath == "" {
		return
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		return
	}

	if details.Family == "" || details.Family == "lmstudio" {
		if architectures, ok := cfg["architectures"].([]any); ok && len(architectures) > 0 {
			if architecture, ok := architectures[0].(string); ok && architecture != "" {
				details.Family = architecture
			}
		}
		if details.Family == "" || details.Family == "lmstudio" {
			if modelType, ok := cfg["model_type"].(string); ok && modelType != "" {
				details.Family = modelType
			}
		}
	}

	expertCount := configUint32(cfg, "num_experts", "num_local_experts", "n_routed_experts")
	expertUsedCount := configUint32(cfg, "experts_per_token", "num_experts_per_tok", "top_k_experts")
	if expertCount == 0 {
		return
	}

	details.ArchitectureType = "moe"
	details.ExpertCount = expertCount
	details.ExpertUsedCount = expertUsedCount
}

func configUint32(cfg map[string]any, keys ...string) uint32 {
	for _, key := range keys {
		switch v := cfg[key].(type) {
		case float64:
			if v > 0 {
				return uint32(v)
			}
		case int:
			if v > 0 {
				return uint32(v)
			}
		}
	}
	return 0
}
