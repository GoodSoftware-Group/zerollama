package server

import (
	"fmt"
	"maps"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/server/autoroute"
)

// routeInventory memoises per-model capabilities for the auto-router (docs/auto-router.md §5).
var routeInventory = autoroute.NewInventory(loadRouteCandidate)

// loadRouteCandidate is the expensive per-digest step: GetModel plus a GGUF open for the
// serving capability set and trained context length.
//
// WHY Capabilities, not publicCapabilities: routing must match what CheckCapabilities
// enforces at serve time; the public subset hides decision-model internals from /api/tags.
func loadRouteCandidate(ref autoroute.Ref) (autoroute.Candidate, error) {
	m, err := GetModel(ref.ID)
	if err != nil {
		return autoroute.Candidate{}, fmt.Errorf("load %s: %w", ref.Name, err)
	}
	// Cloud catalog stubs and LAN passthrough tags never run locally; can-load and
	// decisions are local-only, so they are not routing candidates (v1).
	if m.Config.RemoteModel != "" {
		return autoroute.Candidate{}, autoroute.ErrSkip
	}
	return autoroute.Candidate{
		Capabilities:     m.Capabilities(),
		Backends:         maps.Clone(m.Config.ModalityBackends),
		Format:           m.Config.ModelFormat,
		Family:           m.Config.ModelFamily,
		ContextLength:    modelMaxNumCtx(m),
		SchedKey:         schedulerModelKey(m),
		RuntimeInference: modelUsesRuntimeInference(m),
	}, nil
}

// routeCandidates walks local manifests and returns the memoised candidate list.
// Steady state is one manifest walk; GGUF opens happen only for new or changed digests.
func routeCandidates() ([]autoroute.Candidate, map[string]error, error) {
	ms, err := manifest.Manifests(true)
	if err != nil {
		return nil, nil, err
	}
	refs := make([]autoroute.Ref, 0, len(ms))
	for n, m := range ms {
		refs = append(refs, autoroute.Ref{
			Name:      n.DisplayShortest(),
			ID:        n.String(),
			Digest:    m.Digest(),
			SizeBytes: m.Size(),
		})
	}
	cands, failed := routeInventory.Refresh(refs)
	return cands, failed, nil
}
