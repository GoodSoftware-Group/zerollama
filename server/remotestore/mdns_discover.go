package remotestore

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fleet/mdns"
)

// BrowseStorage watches for _zerollama-storage._tcp until wait elapses or ctx cancels.
// Dedupes by URL. Safe for CLI discover and serve bootstrap.
func BrowseStorage(ctx context.Context, wait time.Duration) []string {
	if wait <= 0 {
		wait = 2 * time.Second
	}
	browseCtx, cancel := context.WithTimeout(ctx, wait)
	defer cancel()

	seen := make(map[string]struct{})
	var mu sync.Mutex
	var out []string

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = mdns.Browse(browseCtx, mdns.BrowseOpts{
			Service: mdns.ServiceStorage,
			OnPeer: func(ev mdns.PeerEvent) {
				url := strings.TrimRight(strings.TrimSpace(ev.URL), "/")
				if url == "" {
					return
				}
				mu.Lock()
				defer mu.Unlock()
				if _, ok := seen[url]; ok {
					return
				}
				seen[url] = struct{}{}
				out = append(out, url)
				slog.Info("storage mdns peer", "url", url, "instance", ev.Instance)
			},
		})
	}()
	<-done

	mu.Lock()
	defer mu.Unlock()
	return append([]string(nil), out...)
}

// BootstrapMDNSIfNeeded fills the process-wide resolver from LAN discovery when
// ZEROLLAMA_STORAGE_SERVERS is empty and storage mDNS browse is enabled.
// Requires ZEROLLAMA_STORAGE_SECRET (same as static servers). Never starts storage serve.
func BootstrapMDNSIfNeeded(ctx context.Context) {
	if envconfig.StorageServers() != "" {
		return
	}
	if !envconfig.StorageMDNSBrowse() {
		return
	}
	secret := envconfig.StorageSecret()
	if secret == "" {
		slog.Info("storage mdns browse skipped", "reason", "ZEROLLAMA_STORAGE_SECRET unset")
		return
	}
	wait := envconfig.StorageMDNSBrowseWait()
	found := BrowseStorage(ctx, wait)
	if len(found) == 0 {
		slog.Info("storage mdns browse: no peers", "wait", wait.String(), "hint", "ZEROLLAMA_MDNS=1 zerollama storage serve on the model host")
		return
	}
	auth, err := NewAuth(secret)
	if err != nil {
		slog.Warn("storage mdns browse: auth", "error", err)
		return
	}
	r := ConfigFromEnv(auth)
	r.Servers = found
	if r.CacheDir == "" {
		r.CacheDir = envconfig.Models()
	}
	if r.MaxBytes == 0 {
		r.MaxBytes = envconfig.RemoteCacheMaxBytes()
	}
	SetDefault(r)
	slog.Info("storage mdns browse: configured", "servers", strings.Join(found, ","))
}
