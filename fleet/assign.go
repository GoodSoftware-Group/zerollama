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

// Assign picks a node for the requested model using warm-first, lowest-queue routing.
// Pure function over a snapshot so unit tests and future F5 token validation stay deterministic.
func Assign(nodes []NodeSnapshot, req AssignRequest) (AssignResponse, error) {
	model := strings.TrimSpace(req.Model)
	if model == "" {
		return AssignResponse{}, ErrModelRequired
	}
	if len(nodes) == 0 {
		return AssignResponse{}, ErrNoNodes
	}

	preferWarm := true
	if req.PreferWarm != nil {
		preferWarm = *req.PreferWarm
	}

	excluded := excludedSet(req.Exclude)
	candidates := filterCandidates(nodes, excluded)
	if len(candidates) == 0 {
		return AssignResponse{}, ErrNoNodes
	}

	now := time.Now().UTC()
	if preferWarm {
		if warm := pickBestNode(candidates, model, true); warm != nil {
			return assignFromNode(*warm, true, now), nil
		}
		if req.WarmOnly {
			return AssignResponse{}, ErrNoWarmNode
		}
	}

	// Cold route: lowest queue; Warm reflects whether the chosen node already has the model.
	if cold := pickBestNode(candidates, model, false); cold != nil {
		return assignFromNode(*cold, nodeHasModel(*cold, model), now), nil
	}
	return AssignResponse{}, ErrNoNodes
}

func assignFromNode(n NodeSnapshot, warm bool, now time.Time) AssignResponse {
	return AssignResponse{
		URL:         n.URL,
		NodeID:      n.ID,
		Warm:        warm,
		QueueDepth:  n.QueueDepth,
		Loading:     n.Loading,
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

func pickBestNode(nodes []NodeSnapshot, model string, warmOnly bool) *NodeSnapshot {
	var best *NodeSnapshot
	for i := range nodes {
		n := &nodes[i]
		if warmOnly && !nodeHasModel(*n, model) {
			continue
		}
		if best == nil || nodeLess(*n, *best) {
			best = n
		}
	}
	return best
}

func nodeLess(a, b NodeSnapshot) bool {
	if a.QueueDepth != b.QueueDepth {
		return a.QueueDepth < b.QueueDepth
	}
	if a.Loading != b.Loading {
		return !a.Loading
	}
	return a.ID < b.ID
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
