package remotestore

import (
	"sync"

	"github.com/ollama/ollama/envconfig"
)

var (
	defaultMu sync.Mutex
	defaultR  *Resolver
)

// Default returns a process-wide Resolver configured from env, or nil if disabled.
// Why a singleton: GetModel / ensureBlob / scheduler pin paths all need the same
// cache + pin set; constructing per-call would lose pin state and LRU accounting.
func Default() *Resolver {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	if defaultR != nil {
		return defaultR
	}
	secret := envconfig.StorageSecret()
	if secret == "" || envconfig.StorageServers() == "" {
		return nil
	}
	auth, err := NewAuth(secret)
	if err != nil {
		return nil
	}
	r := ConfigFromEnv(auth)
	if r.CacheDir == "" {
		r.CacheDir = envconfig.Models()
	}
	if r.MaxBytes == 0 {
		r.MaxBytes = envconfig.RemoteCacheMaxBytes()
	}
	if len(r.Servers) == 0 {
		return nil
	}
	defaultR = r
	return defaultR
}

// SetDefault overrides the process-wide resolver (tests).
func SetDefault(r *Resolver) {
	defaultMu.Lock()
	defaultR = r
	defaultMu.Unlock()
}

// ResetDefault clears the singleton (tests).
func ResetDefault() {
	SetDefault(nil)
}
