package fleet

import (
	"errors"
	"strings"
	"time"
)

var (
	ErrNoPeers       = errors.New("no fleet peers configured")
	ErrNoNodes       = errors.New("no available fleet nodes")
	ErrNoWarmNode    = errors.New("no node has the requested model loaded")
	ErrModelRequired = errors.New("model is required")
)

// Assign picks a node for the requested model using filter-then-score routing.
// Optional cache biases toward nodes that recently served the same session_key (L3 fleet affinity).
// Pure function over a snapshot so unit tests and /internal/score stay deterministic.
func Assign(nodes []NodeSnapshot, req AssignRequest, cache *PrefixCache) (AssignResponse, error) {
	model := strings.TrimSpace(req.Model)
	if model == "" {
		return AssignResponse{}, ErrModelRequired
	}
	if len(nodes) == 0 {
		return AssignResponse{}, ErrNoNodes
	}

	scoreReq := ScoreRequest{
		Model:          model,
		PreferWarm:     req.PreferWarm,
		WarmOnly:       req.WarmOnly,
		Exclude:        req.Exclude,
		SessionKey:     req.SessionKey,
		PromptCacheKey: req.PromptCacheKey,
	}
	result := ScoreCandidates(nodes, scoreReq, cache)
	if result.Best == nil {
		if req.WarmOnly {
			return AssignResponse{}, ErrNoWarmNode
		}
		return AssignResponse{}, ErrNoNodes
	}

	best := result.Best
	now := time.Now().UTC()
	sessionKey := strings.TrimSpace(req.SessionKey)
	if sessionKey == "" {
		sessionKey = strings.TrimSpace(req.PromptCacheKey)
	}
	if cache != nil && sessionKey != "" {
		cache.Remember(model, sessionKey, best.ID)
	}

	resp := assignFromScored(*best, now)
	return resp, nil
}

func assignFromScored(n ScoredNode, now time.Time) AssignResponse {
	return AssignResponse{
		URL:         n.URL,
		NodeID:      n.ID,
		Warm:        n.Warm,
		QueueDepth:  n.QueueDepth,
		Loading:     n.Loading,
		Score:       n.Score,
		GeneratedAt: now,
	}
}

func excludedSet(exclude []string) map[string]struct{} {
	out := make(map[string]struct{}, len(exclude))
	for _, id := range exclude {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		out[strings.ToLower(id)] = struct{}{}
	}
	return out
}

func filterCandidates(nodes []NodeSnapshot, excluded map[string]struct{}) []NodeSnapshot {
	out := make([]NodeSnapshot, 0, len(nodes))
	for _, n := range nodes {
		if !n.Available {
			continue
		}
		if _, ok := excluded[strings.ToLower(n.ID)]; ok {
			continue
		}
		out = append(out, n)
	}
	return out
}

func nodeHasModel(n NodeSnapshot, model string) bool {
	for _, loaded := range n.LoadedModels {
		if modelMatches(loaded, model) {
			return true
		}
	}
	return false
}

func modelMatches(loaded, requested string) bool {
	loaded = strings.ToLower(strings.TrimSpace(loaded))
	requested = strings.ToLower(strings.TrimSpace(requested))
	if loaded == requested {
		return true
	}
	// Why base-name match: agents often omit :tag; loaded_models from F2 use short names.
	loadedBase, _, _ := strings.Cut(loaded, ":")
	reqBase, _, _ := strings.Cut(requested, ":")
	return loadedBase != "" && loadedBase == reqBase
}

// BuildWarmMap groups loaded models across available nodes.
// Why a derived map: agents and dashboards want model→[nodes] without scanning every peer snapshot.
func BuildWarmMap(nodes []NodeSnapshot) []WarmModelEntry {
	type modelGroup struct {
		display string
		nodes   []WarmNodeEntry
	}
	byModel := make(map[string]*modelGroup)
	for _, n := range nodes {
		if !n.Available {
			continue
		}
		seen := make(map[string]struct{})
		for _, m := range n.LoadedModels {
			m = strings.TrimSpace(m)
			if m == "" {
				continue
			}
			k := strings.ToLower(m)
			if _, ok := seen[k]; ok {
				continue
			}
			seen[k] = struct{}{}
			g, ok := byModel[k]
			if !ok {
				g = &modelGroup{display: m}
				byModel[k] = g
			}
			g.nodes = append(g.nodes, WarmNodeEntry{
				NodeID:     n.ID,
				URL:        n.URL,
				QueueDepth: n.QueueDepth,
				Loading:    n.Loading,
			})
		}
	}

	out := make([]WarmModelEntry, 0, len(byModel))
	for _, g := range byModel {
		sortWarmNodes(g.nodes)
		out = append(out, WarmModelEntry{
			Model: g.display,
			Nodes: g.nodes,
		})
	}
	sortWarmModels(out)
	return out
}

func sortWarmNodes(nodes []WarmNodeEntry) {
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			if warmNodeLess(nodes[j], nodes[i]) {
				nodes[i], nodes[j] = nodes[j], nodes[i]
			}
		}
	}
}

func warmNodeLess(a, b WarmNodeEntry) bool {
	if a.QueueDepth != b.QueueDepth {
		return a.QueueDepth < b.QueueDepth
	}
	if a.Loading != b.Loading {
		return !a.Loading
	}
	return a.NodeID < b.NodeID
}

func sortWarmModels(models []WarmModelEntry) {
	for i := 0; i < len(models); i++ {
		for j := i + 1; j < len(models); j++ {
			if strings.ToLower(models[j].Model) < strings.ToLower(models[i].Model) {
				models[i], models[j] = models[j], models[i]
			}
		}
	}
}
