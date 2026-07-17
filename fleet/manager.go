package fleet

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fleet/mdns"
)

// discoveredPeerStale is how long a mDNS-only peer may stay unavailable before pruning.
// Why not immediate: brief restarts and poll blips should not drop a host from the fleet map.
const discoveredPeerStale = 5 * time.Minute

// DiscoveryConfig enables mDNS peer browse (F4). Static peers from Config.Peers are always kept.
type DiscoveryConfig struct {
	Enabled bool
}

// Config configures the fleet manager poll loop.
type Config struct {
	Peers        []string // Zerollama base URLs; merged with mDNS discovery when enabled.
	PollInterval time.Duration
	ProbeTimeout time.Duration
	Discovery    DiscoveryConfig
}

// Manager polls zerollama peers and maintains a warm-model map.
type Manager struct {
	cfg    Config
	client *http.Client

	mu            sync.RWMutex
	nodes         map[string]NodeSnapshot
	staticPeers   map[string]struct{}
	discoveredSet map[string]struct{}
	discoveredAt  map[string]time.Time
	lastAvailable map[string]time.Time
	peerURLs      []string
	prefixCache   *PrefixCache
	probeCache    *ProbeCache
}

func NewManager(cfg Config) (*Manager, error) {
	static := make(map[string]struct{}, len(cfg.Peers))
	peerURLs := make([]string, 0, len(cfg.Peers))
	for _, peer := range cfg.Peers {
		u, err := normalizePeerURL(peer)
		if err != nil {
			return nil, err
		}
		if _, ok := static[u]; ok {
			continue
		}
		static[u] = struct{}{}
		peerURLs = append(peerURLs, u)
	}

	if len(peerURLs) == 0 && !cfg.Discovery.Enabled {
		return nil, ErrNoPeers
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 3 * time.Second
	}
	if cfg.ProbeTimeout <= 0 {
		cfg.ProbeTimeout = 2 * time.Second
	}

	nodes := make(map[string]NodeSnapshot, len(peerURLs))
	for _, peer := range peerURLs {
		id := NodeIDFromURL(peer)
		nodes[id] = NodeSnapshot{ID: id, URL: peer}
	}

	return &Manager{
		cfg:           cfg,
		client:        &http.Client{},
		nodes:         nodes,
		staticPeers:   static,
		discoveredSet: make(map[string]struct{}),
		discoveredAt:  make(map[string]time.Time),
		lastAvailable: make(map[string]time.Time),
		peerURLs:      peerURLs,
		prefixCache:   newPrefixCacheFromEnv(),
		probeCache:    NewProbeCache(probeCacheTTLFromEnv()),
	}, nil
}

// AddDiscoveredPeer registers a peer found via mDNS browse. Static peers are unchanged.
func (m *Manager) AddDiscoveredPeer(rawURL string) error {
	u, err := normalizePeerURL(rawURL)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.staticPeers[u]; ok {
		return nil
	}
	if _, ok := m.discoveredSet[u]; ok {
		return nil
	}
	now := time.Now().UTC()
	m.discoveredSet[u] = struct{}{}
	m.discoveredAt[u] = now
	m.peerURLs = append(m.peerURLs, u)
	id := NodeIDFromURL(u)
	if _, ok := m.nodes[id]; !ok {
		m.nodes[id] = NodeSnapshot{ID: id, URL: u}
	}
	slog.Info("fleet discovered peer", "url", u)
	return nil
}

// PeerCount returns static + discovered peers.
func (m *Manager) PeerCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.peerURLs)
}

func (m *Manager) peerList() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]string, len(m.peerURLs))
	copy(out, m.peerURLs)
	return out
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
	peers := m.peerList()
	if len(peers) == 0 {
		return
	}
	var wg sync.WaitGroup
	for _, peer := range peers {
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

	status, err := m.probeCache.Fetch(probeCtx, peer, func(ctx context.Context) (*api.StatusResponse, error) {
		return fetchStatus(ctx, m.client, peer)
	})
	if err != nil {
		snap.LastError = err.Error()
		if m.prefixCache != nil {
			m.prefixCache.DropNode(id)
		}
		m.setNode(snap)
		m.maybePruneDiscovered(peer)
		return
	}

	snap.Available = true
	snap.Inference = status.Inference
	snap.LoadedModels = append([]string(nil), status.Inference.Ggml.LoadedModels...)
	snap.QueueDepth = queueDepth(status.Inference)
	snap.Loading = status.Inference.Ggml.Loading
	m.markAvailable(peer)
	m.setNode(snap)
}

func (m *Manager) markAvailable(url string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastAvailable[url] = time.Now().UTC()
}

func (m *Manager) maybePruneDiscovered(url string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.staticPeers[url]; ok {
		return
	}
	if _, ok := m.discoveredSet[url]; !ok {
		return
	}

	staleSince := m.discoveredAt[url]
	if lastOK, ok := m.lastAvailable[url]; ok {
		staleSince = lastOK
	}
	if time.Since(staleSince) < discoveredPeerStale {
		return
	}

	m.removeDiscoveredPeerLocked(url)
	slog.Info("fleet pruned stale discovered peer", "url", url)
}

func (m *Manager) removeDiscoveredPeerLocked(url string) {
	delete(m.discoveredSet, url)
	delete(m.discoveredAt, url)
	delete(m.lastAvailable, url)

	filtered := m.peerURLs[:0]
	for _, u := range m.peerURLs {
		if u != url {
			filtered = append(filtered, u)
		}
	}
	m.peerURLs = filtered

	id := NodeIDFromURL(url)
	delete(m.nodes, id)
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
	// Pending already includes AssignHolds on nodes that fold soft holds into pending (F5).
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

// Assign routes a model request to the best node and optionally mints an F5 hold token.
func (m *Manager) Assign(req AssignRequest) (AssignResponse, error) {
	snap := m.Snapshot()
	resp, err := Assign(snap.Nodes, req, m.prefixCache)
	if err != nil {
		return resp, err
	}
	if AssignPushHoldEnabled() && resp.AssignmentToken != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		m.pushAssignHold(ctx, resp)
	}
	return resp, nil
}

// Score ranks nodes for a model without committing an assign (or updating affinity).
func (m *Manager) Score(req ScoreRequest) ScoreResponse {
	snap := m.Snapshot()
	return ScoreCandidates(snap.Nodes, req, m.prefixCache)
}

// RunMDNSDiscovery browses _zerollama._tcp until ctx is cancelled.
func RunMDNSDiscovery(ctx context.Context, m *Manager) {
	_ = mdns.Browse(ctx, mdns.BrowseOpts{
		Service: mdns.ServiceNode,
		OnPeer: func(ev mdns.PeerEvent) {
			if err := m.AddDiscoveredPeer(ev.URL); err != nil {
				slog.Debug("fleet discovery skip peer", "url", ev.URL, "error", err)
			}
		},
	})
}
