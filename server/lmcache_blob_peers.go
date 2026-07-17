package server

import (
	"os"
	"strings"
)

// lmcacheBlobPeersForCoordination returns peer base URLs for L3-R10/R11 HTTP
// blob pull, pushed to Python via /internal/go-coordination.
// Explicit ZEROLLAMA_LMCACHE_BLOB_PEERS wins; else ZEROLLAMA_FLEET_PEERS.
func lmcacheBlobPeersForCoordination() []string {
	raw := strings.TrimSpace(os.Getenv("ZEROLLAMA_LMCACHE_BLOB_PEERS"))
	if raw == "" {
		raw = strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_PEERS"))
	}
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	seen := make(map[string]struct{}, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(strings.TrimRight(p, "/"))
		if p == "" {
			continue
		}
		key := strings.ToLower(p)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, p)
	}
	return out
}
