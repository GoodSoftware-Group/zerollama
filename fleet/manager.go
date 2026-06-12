package fleet

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

// Config configures the fleet manager poll loop.
type Config struct {
	Peers        []string      // Zerollama base URLs; static list until F4 mDNS.
	PollInterval time.Duration // How often to refresh warm-model map from F2 /api/status.
	ProbeTimeout time.Duration // Per-peer HTTP timeout; kept separate from PollInterval so slow nodes don't block the tick.
}

// Manager polls zerollama peers and maintains a warm-model map.
// Why poll not push: nodes already expose F2 status; push registration adds wire complexity
// before we need it. Management stays read-only relative to node schedulers.
type Manager struct {
	cfg    Config
	client *http.Client

	mu    sync.RWMutex
	nodes map[string]NodeSnapshot
}

func NewManager(cfg Config) (*Manager, error) {
	if len(cfg.Peers) == 0 {
		return nil, ErrNoPeers
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 3 * time.Second
	}
	if cfg.ProbeTimeout <= 0 {
		cfg.ProbeTimeout = 2 * time.Second
	}

	nodes := make(map[string]NodeSnapshot, len(cfg.Peers))
	for _, peer := range cfg.Peers {
		id := NodeIDFromURL(peer)
		nodes[id] = NodeSnapshot{ID: id, URL: peer}
	}

	return &Manager{
		cfg: cfg,
		client: &http.Client{},
		nodes: nodes,
	}, nil
}

// Run polls peers until ctx is cancelled.
func (m *Manager) Run(ctx context.Context) {
	m.pollAll(ctx)

	ticker := time.NewTicker(m.cfg.PollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.pollAll(ctx)
		}
	}
}

func (m *Manager) pollAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, peer := range m.cfg.Peers {
		wg.Add(1)
		go func(peer string) {
			defer wg.Done()
			m.pollPeer(ctx, peer)
		}(peer)
	}
	wg.Wait()
}

func (m *Manager) pollPeer(ctx context.Context, peer string) {
	id := NodeIDFromURL(peer)
	snap := NodeSnapshot{
		ID:        id,
		URL:       peer,
		LastPoll:  time.Now().UTC(),
		Available: false,
	}

	probeCtx, cancel := context.WithTimeout(ctx, m.cfg.ProbeTimeout)
	defer cancel()

	status, err := fetchStatus(probeCtx, m.client, peer)
	if err != nil {
		snap.LastError = err.Error()
		m.setNode(snap)
		return
	}

	snap.Available = true
	snap.Inference = status.Inference
	snap.LoadedModels = append([]string(nil), status.Inference.Ggml.LoadedModels...)
	snap.QueueDepth = queueDepth(status.Inference)
	snap.Loading = status.Inference.Ggml.Loading
	m.setNode(snap)
}

func fetchStatus(ctx context.Context, client *http.Client, base string) (*api.StatusResponse, error) {
	url := strings.TrimSuffix(base, "/") + "/api/status"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var status api.StatusResponse
	if err := json.Unmarshal(body, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func queueDepth(in api.InferenceStatus) int {
	// Why sum ggml + runtime: a node may route text through Python runtime while ggml
	// still holds other runners; agents care about total inference backlog for routing.
	depth := in.Ggml.Pending + in.Ggml.Active
	if in.Runtime.Enabled && in.Runtime.Available {
		if in.Runtime.Waiting != nil {
			depth += *in.Runtime.Waiting
		}
		if in.Runtime.Running != nil {
			depth += *in.Runtime.Running
		}
	}
	return depth
}

func (m *Manager) setNode(snap NodeSnapshot) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nodes[snap.ID] = snap
}

// Snapshot returns the current fleet view.
func (m *Manager) Snapshot() FleetStatusResponse {
	m.mu.RLock()
	defer m.mu.RUnlock()

	nodes := make([]NodeSnapshot, 0, len(m.nodes))
	for _, n := range m.nodes {
		nodes = append(nodes, n)
	}
	sortNodes(nodes)

	now := time.Now().UTC()
	return FleetStatusResponse{
		Nodes:       nodes,
		WarmModels:  BuildWarmMap(nodes),
		GeneratedAt: now,
	}
}

func sortNodes(nodes []NodeSnapshot) {
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			if nodes[j].ID < nodes[i].ID {
				nodes[i], nodes[j] = nodes[j], nodes[i]
			}
		}
	}
}

// Assign routes a model request to the best node.
func (m *Manager) Assign(req AssignRequest) (AssignResponse, error) {
	snap := m.Snapshot()
	return Assign(snap.Nodes, req)
}
