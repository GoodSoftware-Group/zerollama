// list_host_context — VRAM/RAM-aware max context for /api/tags (zerollama ls CTX).
//
// WHY: operators need “what can this host do right now” next to PARAMS/PERF, not only
// n_ctx_train. When free memory can’t hold train max, CLI shows a range host–train.
package server

import (
	"context"
	"log/slog"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// enrichListHostContexts fills HostMaxContext using current free VRAM (plus a credit
// when the model is already loaded). Train max stays in Details.ContextLength.
//
// Fast path: uses WeightSizeBytes / Size already filled by list enrichment and a
// size heuristic — no second GetModel + GraphSize binary-search per tag (that made
// /api/tags ~10s with hundreds of local GGUFs). Set ZEROLLAMA_TAGS_GRAPHSIZE=1 for
// the slower GraphSize estimate.
func (s *Server) enrichListHostContexts(ctx context.Context, models []api.ListModelResponse) {
	if len(models) == 0 {
		return
	}

	free := s.effectiveGgmlFreeVRAMForSuggest(ctx, false)
	credits := s.loadedVRAMByShortName()
	// Ghost FreeMemory=0 with nothing loaded: skip entirely (was GetModel×N for zeros).
	if free == 0 && len(credits) == 0 {
		return
	}

	accurate := tagsGraphSizeEnabled()

	for i := range models {
		m := &models[i]
		if m.RemoteModel != "" && m.RemoteHost != "lmstudio" {
			continue // cloud catalog stubs — no local VRAM math
		}

		train := m.Details.ContextLength
		name := m.Name
		if name == "" {
			name = m.Model
		}

		budget := free
		if credit := credits[name]; credit > 0 {
			budget += credit
		}
		if budget == 0 {
			continue
		}

		sizeBytes := uint64(0)
		if m.Details.WeightSizeBytes > 0 {
			sizeBytes = m.Details.WeightSizeBytes
		} else if m.Size > 0 {
			sizeBytes = uint64(m.Size)
		}

		host := 0
		if accurate {
			mdl, err := GetModel(name)
			if err == nil && mdl != nil {
				if train <= 0 {
					if t := modelMaxNumCtx(mdl); t > 0 {
						train = t
						m.Details.ContextLength = t
					}
				}
				host = s.hostMaxContextForModel(mdl, budget, train, sizeBytes)
			} else if train > 0 && sizeBytes > 0 {
				host = suggestHostCtxFromSize(sizeBytes, budget, train)
			}
		} else if train > 0 && sizeBytes > 0 {
			host = suggestHostCtxFromSize(sizeBytes, budget, train)
		} else {
			// Missing train/size on the list row — one GetModel to fill holes.
			mdl, err := GetModel(name)
			if err == nil && mdl != nil {
				if train <= 0 {
					if t := modelMaxNumCtx(mdl); t > 0 {
						train = t
						m.Details.ContextLength = t
					}
				}
				host = s.hostMaxContextForModel(mdl, budget, train, sizeBytes)
			}
		}

		if host > 0 {
			m.HostMaxContext = host
		}
	}
}

func tagsGraphSizeEnabled() bool {
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_TAGS_GRAPHSIZE"))
	return s == "1" || strings.EqualFold(s, "true") || strings.EqualFold(s, "on")
}

func (s *Server) hostMaxContextForModel(mdl *Model, budget uint64, train int, sizeBytes uint64) int {
	if mdl == nil || budget == 0 {
		return 0
	}

	if mdl.IsMLX() {
		return suggestHostCtxFromSize(sizeBytes, budget, train)
	}

	if mdl.ModelPath == "" {
		return 0
	}

	f, err := llm.LoadModelMetadata(mdl.ModelPath)
	if err != nil {
		slog.Debug("list host ctx: metadata skipped", "model", mdl.ShortName, "error", err)
		return suggestHostCtxFromSize(sizeBytes, budget, train)
	}

	opts := api.DefaultOptions()
	if train > 0 {
		opts.NumCtx = train
	}
	profile := ggmlLoadProfileFor(mdl, opts)
	suggested := suggestMaxGgmlNumCtx(f, mdl.ModelPath, budget, profile)
	if suggested <= 0 {
		return 0
	}
	if train > 0 && suggested > train {
		return train
	}
	return suggested
}

// suggestHostCtxFromSize is a coarse UMA/MLX fallback when GGUF GraphSize is unavailable.
// Assumes KV for full train ≈ half of weight bytes (conservative); scales leftover linearly.
func suggestHostCtxFromSize(weightBytes, budget uint64, train int) int {
	if budget == 0 || weightBytes == 0 {
		return 0
	}
	margin := uint64(float64(weightBytes) * 1.10)
	if margin >= budget {
		return 0
	}
	leftover := budget - margin
	if train <= 0 {
		train = 8192
	}
	// Full-train KV proxy: half of weights (typical large-ctx pressure vs weights).
	kvFull := weightBytes / 2
	if kvFull == 0 {
		return 0
	}
	if leftover >= kvFull {
		return train
	}
	host := int(float64(train) * float64(leftover) / float64(kvFull))
	if host < 512 {
		if leftover > 0 {
			return 512
		}
		return 0
	}
	if host > train {
		return train
	}
	return host
}

// loadedVRAMByShortName maps DisplayShortest / ShortName → runner vramSize.
func (s *Server) loadedVRAMByShortName() map[string]uint64 {
	out := map[string]uint64{}
	if s == nil || s.sched == nil {
		return out
	}
	s.sched.loadedMu.Lock()
	defer s.sched.loadedMu.Unlock()
	for _, r := range s.sched.loaded {
		if r == nil || r.model == nil || r.vramSize == 0 {
			continue
		}
		name := r.model.ShortName
		if name == "" {
			continue
		}
		if prev, ok := out[name]; !ok || r.vramSize > prev {
			out[name] = r.vramSize
		}
	}
	return out
}
