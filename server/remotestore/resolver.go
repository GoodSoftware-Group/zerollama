package remotestore

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"

	"github.com/ollama/ollama/envconfig"
)

// CacheMode selects how fetched blobs are retained.
type CacheMode string

const (
	// CachePersist keeps blobs under the local cache root and may LRU-evict.
	// Why default: warm restarts without re-fetching multi-GB weights.
	CachePersist CacheMode = "persist"
	// CacheEphemeral writes to scratch and deletes on runner unload.
	// Why: operators who want "stream without a lasting footprint" without
	// waiting for llama.cpp tensor streaming patches.
	CacheEphemeral CacheMode = "ephemeral"
)

// Resolver fetches content-addressed blobs from remote storage servers into a local cache.
// Why this sits under GetModel: every load path already resolves manifests + layer
// digests; miss-fetch here keeps run/chat APIs unchanged.
type Resolver struct {
	Servers    []string
	Auth       *Auth
	CacheDir   string // models root for blobs (…/blobs); manifests always use OLLAMA_MODELS
	MaxBytes   int64  // 0 = unlimited
	Mode       CacheMode
	ScratchDir string // ephemeral only
	Chain      *TransportChain
	HTTP       *http.Client

	mu        sync.Mutex
	caps      map[string]Capability
	pinned    map[string]int     // digest → refcount of loaded models holding this blob
	ephemeral map[string]string  // digest → scratch path
	dlGroup   singleflight.Group // deduplicates concurrent downloads of the same digest
}

// Config builds a Resolver from environment defaults.
func ConfigFromEnv(auth *Auth) *Resolver {
	servers := StorageServers()
	mode := CacheMode(strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_REMOTE_CACHE_MODE"))))
	if mode != CacheEphemeral {
		mode = CachePersist
	}
	cache := os.Getenv("ZEROLLAMA_REMOTE_CACHE_DIR")
	if cache == "" {
		cache = "" // filled by caller with Models()
	}
	var max int64
	if s := strings.TrimSpace(os.Getenv("ZEROLLAMA_REMOTE_CACHE_MAX_BYTES")); s != "" {
		fmt.Sscanf(s, "%d", &max)
	}
	scratch := strings.TrimSpace(os.Getenv("ZEROLLAMA_REMOTE_SCRATCH_DIR"))
	r := &Resolver{
		Servers:    servers,
		Auth:       auth,
		CacheDir:   cache,
		MaxBytes:   max,
		Mode:       mode,
		ScratchDir: scratch,
		Chain:      PreferRDMAThenTCP(auth),
		HTTP:       &http.Client{Timeout: 30 * time.Minute},
		caps:       make(map[string]Capability),
		pinned:     make(map[string]int),
		ephemeral:  make(map[string]string),
	}
	return r
}

// StorageServers parses ZEROLLAMA_STORAGE_SERVERS (comma-separated base URLs).
func StorageServers() []string {
	raw := strings.TrimSpace(os.Getenv("ZEROLLAMA_STORAGE_SERVERS"))
	if raw == "" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(raw, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, strings.TrimRight(p, "/"))
		}
	}
	return out
}

// Enabled is true when at least one storage server is configured and auth works.
func (r *Resolver) Enabled() bool {
	return r != nil && len(r.Servers) > 0 && r.Auth != nil
}

// Pin marks digests as non-evictable while a model is loaded.
// Why refcounts: shared layers across concurrent loaded models must stay pinned
// until the last holder unpins — a boolean set would let unload of A delete B's mmap.
func (r *Resolver) Pin(digests ...string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.pinned == nil {
		r.pinned = make(map[string]int)
	}
	for _, d := range digests {
		if d == "" {
			continue
		}
		r.pinned[normalizeDigest(d)]++
	}
}

// Unpin releases one pin reference per digest. Digests drop out of the
// non-evictable set when their refcount reaches zero.
func (r *Resolver) Unpin(digests ...string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, d := range digests {
		if d == "" {
			continue
		}
		key := normalizeDigest(d)
		n := r.pinned[key]
		if n <= 1 {
			delete(r.pinned, key)
		} else {
			r.pinned[key] = n - 1
		}
	}
}

// ReleaseModelBlobs unpins digests and, in ephemeral mode, deletes scratch files.
// Why call from the scheduler on unload: LRU must not race a live runner, and
// ephemeral mode must not leak scratch across sessions.
func (r *Resolver) ReleaseModelBlobs(digests ...string) {
	if r == nil {
		return
	}
	r.Unpin(digests...)
	if r.Mode == CacheEphemeral {
		for _, d := range digests {
			r.ReleaseEphemeral(d)
		}
	}
}

// LocalBlobPath returns the on-disk path for a digest in the persist cache.
func (r *Resolver) LocalBlobPath(digest string) (string, error) {
	digest = normalizeDigest(digest)
	dir := filepath.Join(r.CacheDir, "blobs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return filepath.Join(dir, digest), nil
}

// Fetch ensures digest exists locally and returns its path.
func (r *Resolver) Fetch(ctx context.Context, digest string) (path string, err error) {
	if r == nil || !r.Enabled() {
		return "", errors.New("remotestore: not configured")
	}
	digest = normalizeDigest(digest)

	if r.Mode == CacheEphemeral {
		return r.fetchEphemeral(ctx, digest)
	}
	return r.fetchPersist(ctx, digest)
}

func (r *Resolver) fetchPersist(ctx context.Context, digest string) (string, error) {
	path, err := r.LocalBlobPath(digest)
	if err != nil {
		return "", err
	}
	if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
		_ = os.Chtimes(path, time.Now(), time.Now())
		return path, nil
	}
	// Coalesce concurrent fetches of the same digest to a single download.
	_, dlErr, _ := r.dlGroup.Do(digest, func() (any, error) {
		// Re-check inside the group in case a concurrent call already completed.
		if fi, err2 := os.Stat(path); err2 == nil && fi.Size() > 0 {
			_ = os.Chtimes(path, time.Now(), time.Now())
			return nil, nil
		}
		if err2 := r.download(ctx, digest, path); err2 != nil {
			return nil, err2
		}
		if err2 := r.evictIfNeeded(); err2 != nil {
			slog.Warn("remotestore eviction", "error", err2)
		}
		return nil, nil
	})
	if dlErr != nil {
		return "", dlErr
	}
	return path, nil
}

func (r *Resolver) fetchEphemeral(ctx context.Context, digest string) (string, error) {
	r.mu.Lock()
	if r.ephemeral == nil {
		r.ephemeral = make(map[string]string)
	}
	if p, ok := r.ephemeral[digest]; ok {
		if _, err := os.Stat(p); err == nil {
			r.mu.Unlock()
			return p, nil
		}
		delete(r.ephemeral, digest)
	}
	r.mu.Unlock()

	scratch := r.ScratchDir
	if scratch == "" {
		scratch = filepath.Join(os.TempDir(), "zerollama-remote-scratch")
	}
	if err := os.MkdirAll(scratch, 0o700); err != nil {
		return "", err
	}
	path := filepath.Join(scratch, digest)
	if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
		r.mu.Lock()
		r.ephemeral[digest] = path
		r.mu.Unlock()
		return path, nil
	}
	// Coalesce concurrent downloads.
	_, dlErr, _ := r.dlGroup.Do("eph:"+digest, func() (any, error) {
		if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
			return nil, nil
		}
		return nil, r.download(ctx, digest, path)
	})
	if dlErr != nil {
		return "", dlErr
	}
	r.mu.Lock()
	r.ephemeral[digest] = path
	r.mu.Unlock()
	return path, nil
}

// ReleaseEphemeral deletes a scratch blob after the consumer is done.
func (r *Resolver) ReleaseEphemeral(digest string) {
	if r == nil || r.Mode != CacheEphemeral {
		return
	}
	digest = normalizeDigest(digest)
	r.mu.Lock()
	p := r.ephemeral[digest]
	delete(r.ephemeral, digest)
	r.mu.Unlock()
	if p != "" {
		_ = os.Remove(p)
	}
}

func (r *Resolver) download(ctx context.Context, digest, dest string) error {
	partial := dest + ".partial"
	_ = os.Remove(partial)

	var last error
	for _, base := range r.Servers {
		cap := r.capability(ctx, base)
		tcp := NewTCPTransport(r.Auth)
		tcp.Client = r.HTTP
		size, ok, err := tcp.HeadBlob(ctx, base, digest)
		if err != nil {
			last = err
			continue
		}
		if !ok {
			last = fmt.Errorf("blob not found on %s", base)
			continue
		}
		_ = size

		rc, n, via, err := r.Chain.FetchChunk(ctx, base, digest, 0, 0, cap)
		if err != nil {
			last = err
			continue
		}
		err = writeAtomic(partial, dest, rc, n, digest)
		rc.Close()
		if err != nil {
			last = err
			continue
		}
		slog.Info("remotestore fetched blob", "digest", digest, "via", via, "server", base)
		return nil
	}
	if last == nil {
		last = fmt.Errorf("blob %s not found on any storage server", digest)
	}
	return last
}

func (r *Resolver) capability(ctx context.Context, base string) Capability {
	r.mu.Lock()
	if r.caps == nil {
		r.caps = make(map[string]Capability)
	}
	if c, ok := r.caps[base]; ok {
		r.mu.Unlock()
		return c
	}
	r.mu.Unlock()
	c, err := GetCapability(ctx, r.Auth, r.HTTP, base)
	if err != nil {
		// Do not cache auth/config failures as "tcp OK" — retry next time.
		slog.Debug("remotestore capability probe failed", "server", base, "error", err)
		return Capability{Transports: []string{"tcp"}}
	}
	r.mu.Lock()
	if r.caps == nil {
		r.caps = make(map[string]Capability)
	}
	r.caps[base] = c
	r.mu.Unlock()
	return c
}

// writeAtomic streams rc to partial, verifies the SHA-256 against digest (must
// be normalised "sha256-<hex>"), then renames to final only on success.
//
// Why verify before rename: a crash after rename-before-hash left corrupt bytes
// at the final path; the next cache hit treated Size()>0 as trusted and loaded
// bad weights. Hash-while-write avoids a second full read when possible.
func writeAtomic(partial, final string, rc io.Reader, expect int64, digest string) error {
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	h := sha256.New()
	written, err := io.Copy(io.MultiWriter(f, h), rc)
	cerr := f.Close()
	if err != nil {
		_ = os.Remove(partial)
		return err
	}
	if cerr != nil {
		_ = os.Remove(partial)
		return cerr
	}
	if expect > 0 && written != expect {
		_ = os.Remove(partial)
		return fmt.Errorf("short write: got %d want %d", written, expect)
	}
	// Verify before rename so corrupt bytes never reach the final cache path.
	if digest != "" {
		want := strings.TrimPrefix(normalizeDigest(digest), "sha256-")
		got := hex.EncodeToString(h.Sum(nil))
		if !strings.EqualFold(got, want) {
			_ = os.Remove(partial)
			return fmt.Errorf("digest mismatch: got %s want %s", got, want)
		}
	}
	return os.Rename(partial, final)
}


type blobStat struct {
	path string
	size int64
	atime time.Time
}

func (r *Resolver) evictIfNeeded() error {
	if r.MaxBytes <= 0 {
		return nil
	}
	dir := filepath.Join(r.CacheDir, "blobs")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	var total int64
	var stats []blobStat
	r.mu.Lock()
	pinned := make(map[string]int, len(r.pinned))
	for k, n := range r.pinned {
		if n > 0 {
			pinned[k] = n
		}
	}
	r.mu.Unlock()

	for _, e := range entries {
		if e.IsDir() || strings.HasSuffix(e.Name(), ".partial") {
			continue
		}
		p := filepath.Join(dir, e.Name())
		fi, err := os.Stat(p)
		if err != nil {
			continue
		}
		total += fi.Size()
		stats = append(stats, blobStat{path: p, size: fi.Size(), atime: fi.ModTime()})
	}
	if total <= r.MaxBytes {
		return nil
	}
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].atime.Before(stats[j].atime)
	})
	for _, s := range stats {
		if total <= r.MaxBytes {
			break
		}
		name := filepath.Base(s.path)
		if pinned[name] > 0 {
			continue
		}
		if err := os.Remove(s.path); err == nil {
			total -= s.size
			slog.Info("remotestore evicted blob", "path", s.path, "size", s.size)
		}
	}
	return nil
}

// FetchManifest downloads a remote manifest JSON into the local manifests tree.
// Why always write under envconfig.Models(): ParseNamedManifest only looks there.
// Blob CacheDir may differ (separate SSD); following CacheDir for manifests made
// fetch "succeed" while GetModel still failed to open the manifest.
func (r *Resolver) FetchManifest(ctx context.Context, host, ns, model, tag string) ([]byte, error) {
	if !r.Enabled() {
		return nil, errors.New("remotestore: not configured")
	}
	rel := filepath.Join(host, ns, model, tag)
	var last error
	for _, base := range r.Servers {
		url := base + "/v1/manifest/" + strings.ReplaceAll(rel, string(filepath.Separator), "/")
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			last = err
			continue
		}
		if err := r.Auth.SignRequest(req, nil); err != nil {
			return nil, err
		}
		resp, err := r.HTTP.Do(req)
		if err != nil {
			last = err
			continue
		}
		b, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			last = err
			continue
		}
		if resp.StatusCode != http.StatusOK {
			last = fmt.Errorf("%s: %s", url, resp.Status)
			continue
		}
		// Manifests must land under OLLAMA_MODELS so ParseNamedManifest can find them,
		// even when ZEROLLAMA_REMOTE_CACHE_DIR points blobs elsewhere.
		dest := filepath.Join(envconfig.Models(), "manifests", host, ns, model, tag)
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return b, nil // still return bytes
		}
		_ = os.WriteFile(dest, b, 0o644)
		return b, nil
	}
	return nil, last
}

// PushBlob uploads a local file as digest to a storage server.
// Why sign empty body: Content-Length can be multi-GB; HMAC of the full body
// would force a full read. Server hashes the stream and rejects digest mismatch.
func PushBlob(ctx context.Context, auth *Auth, client *http.Client, baseURL, digest, path string) error {
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Minute}
	}
	digest = normalizeDigest(digest)
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	url := strings.TrimRight(baseURL, "/") + "/v1/blob/" + digest
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, f)
	if err != nil {
		return err
	}
	if fi, err := f.Stat(); err == nil {
		req.ContentLength = fi.Size()
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	// Sign empty body: integrity is enforced by digest path + server-side hash.
	if err := auth.SignRequest(req, nil); err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("put blob: %s: %s", resp.Status, b)
	}
	return nil
}

// PushManifest uploads manifest JSON.
func PushManifest(ctx context.Context, auth *Auth, client *http.Client, baseURL, host, ns, model, tag string, data []byte) error {
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute}
	}
	rel := strings.Join([]string{host, ns, model, tag}, "/")
	url := strings.TrimRight(baseURL, "/") + "/v1/manifest/" + rel
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := auth.SignRequest(req, data); err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("put manifest: %s: %s", resp.Status, b)
	}
	return nil
}
