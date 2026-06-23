package fleet

import (
	"context"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

// ProbeCache coalesces short-lived GET /api/status results per peer URL.
// Why: assign retries and overlapping poll ticks should not hammer the same node.
type ProbeCache struct {
	mu       sync.Mutex
	ttl      time.Duration
	entries  map[string]probeCacheEntry
	inflight map[string]*probeInflight
}

type probeCacheEntry struct {
	status    *api.StatusResponse
	err       error
	fetchedAt time.Time
}

type probeInflight struct {
	done chan struct{}
	res  probeCacheEntry
}

func NewProbeCache(ttl time.Duration) *ProbeCache {
	return &ProbeCache{
		ttl:      ttl,
		entries:  make(map[string]probeCacheEntry),
		inflight: make(map[string]*probeInflight),
	}
}

func probeCacheTTLFromEnv() time.Duration {
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_PROBE_CACHE_TTL"))
	if s == "" {
		return time.Second
	}
	if strings.EqualFold(s, "0") || strings.EqualFold(s, "off") || strings.EqualFold(s, "false") {
		return 0
	}
	if d, err := time.ParseDuration(s); err == nil && d > 0 {
		return d
	}
	if n, err := strconv.ParseInt(s, 10, 64); err == nil && n > 0 {
		return time.Duration(n) * time.Second
	}
	return time.Second
}

func (c *ProbeCache) Fetch(ctx context.Context, peer string, fetch func(context.Context) (*api.StatusResponse, error)) (*api.StatusResponse, error) {
	if c == nil || c.ttl <= 0 {
		return fetch(ctx)
	}

	now := time.Now()

	c.mu.Lock()
	if ent, ok := c.entries[peer]; ok && now.Sub(ent.fetchedAt) < c.ttl {
		status, err := ent.status, ent.err
		c.mu.Unlock()
		return status, err
	}
	if flight, ok := c.inflight[peer]; ok {
		c.mu.Unlock()
		select {
		case <-flight.done:
			return flight.res.status, flight.res.err
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	flight := &probeInflight{done: make(chan struct{})}
	c.inflight[peer] = flight
	c.mu.Unlock()

	status, err := fetch(ctx)
	ent := probeCacheEntry{status: status, err: err, fetchedAt: time.Now()}

	c.mu.Lock()
	c.entries[peer] = ent
	delete(c.inflight, peer)
	flight.res = ent
	close(flight.done)
	c.mu.Unlock()

	return status, err
}
