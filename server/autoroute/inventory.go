// Package autoroute holds the model-agnostic parts of the auto-router (docs/auto-router.md):
// the candidate inventory, request signals, filters and scoring. It must not import
// server — the server package supplies a Loader that knows how to open manifests.
//
// WHY a separate package: the policy is pure data in / data out, so it is unit-testable
// without the CGO runner tree that the server package links.
package autoroute

import (
	"errors"
	"slices"
	"sort"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/internal/benchcache"
	"github.com/ollama/ollama/types/model"
)

// Candidate is one local model as the router sees it.
type Candidate struct {
	Name          string             `json:"name"`               // display name clients send (DisplayShortest)
	Digest        string             `json:"digest"`             // manifest digest — invalidation key and bench.json key
	Capabilities  []model.Capability `json:"capabilities"`       // serving set (what CheckCapabilities enforces), not the public subset
	Backends      map[string]string  `json:"backends,omitempty"` // ConfigV2.ModalityBackends
	Format        string             `json:"format,omitempty"`   // gguf | safetensors | …
	Family        string             `json:"family,omitempty"`
	ContextLength int                `json:"context_length,omitempty"` // trained / manifest limit; 0 = unknown (never filters)
	SizeBytes     int64              `json:"size"`

	// Server-supplied facts the availability probe needs without re-opening the model.
	ID               string `json:"id"`                          // fully qualified name (GetModel key)
	SchedKey         string `json:"-"`                           // scheduler loaded-map / cooldown key (often a blob path)
	RuntimeInference bool   `json:"runtime_inference,omitempty"` // model is configured for the Python runtime sidecar
}

// Has reports whether c has every capability in want.
func (c Candidate) Has(want ...model.Capability) bool {
	for _, w := range want {
		if !slices.Contains(c.Capabilities, w) {
			return false
		}
	}
	return true
}

// Backend returns the configured driver for modality ("" when unset).
func (c Candidate) Backend(modality string) string {
	return c.Backends[modality]
}

// Perf returns the bench.json entry for c's digest, when `zerollama bench` has run.
// WHY not memoised on the row: bench can rewrite the file while serve runs; Snapshot
// already caches by mtime.
func (c Candidate) Perf() (benchcache.Entry, bool) {
	e, ok := benchcache.Snapshot()[c.Digest]
	if !ok || !e.Cached() {
		return benchcache.Entry{}, false
	}
	return e, true
}

// Ref identifies one manifest for Refresh.
type Ref struct {
	Name      string // display name (DisplayShortest)
	ID        string // fully qualified name passed to the loader
	Digest    string
	SizeBytes int64
}

// ErrSkip tells Refresh the model is intentionally not a candidate (e.g. a cloud stub).
// It is cached like a success so the loader is not re-run until the digest changes.
var ErrSkip = errors.New("autoroute: not a local candidate")

// Loader builds a Candidate for ref. It is the expensive step (manifest parse, GGUF open
// for capabilities and context length), which is why Inventory memoises by digest.
type Loader func(Ref) (Candidate, error)

type row struct {
	digest string
	cand   Candidate
	err    error // nil, ErrSkip, or a load failure — all cached until the digest changes
}

// Inventory memoises Loader results per model name, keyed by manifest digest.
//
// WHY digest invalidation: Model.Capabilities reopens the GGUF on every call. Building a
// candidate list per routed request would open every GGUF on disk; recomputing only rows
// whose digest changed keeps steady-state cost at one manifest walk.
type Inventory struct {
	mu          sync.Mutex
	rows        map[string]row
	load        Loader
	parallelism int
}

// NewInventory returns an empty inventory backed by load.
func NewInventory(load Loader) *Inventory {
	return &Inventory{rows: map[string]row{}, load: load, parallelism: 8}
}

// Refresh reconciles the inventory with refs (the current manifest list) and returns the
// usable candidates sorted by name, plus load failures by name. Rows with an unchanged
// digest are reused; new or changed rows are loaded (bounded parallelism); rows missing
// from refs are dropped.
//
// WHY failures are cached too: a corrupt manifest would otherwise be re-opened on every
// routed request. Re-pulling or repairing it changes the digest and clears the entry.
func (inv *Inventory) Refresh(refs []Ref) ([]Candidate, map[string]error) {
	inv.mu.Lock()
	defer inv.mu.Unlock()

	seen := make(map[string]struct{}, len(refs))
	var stale []Ref
	for _, ref := range refs {
		seen[ref.Name] = struct{}{}
		if r, ok := inv.rows[ref.Name]; ok && r.digest == ref.Digest {
			continue
		}
		stale = append(stale, ref)
	}
	for name := range inv.rows {
		if _, ok := seen[name]; !ok {
			delete(inv.rows, name)
		}
	}

	if len(stale) > 0 {
		loaded := make([]row, len(stale))
		var g errgroup.Group
		g.SetLimit(max(1, inv.parallelism))
		for i, ref := range stale {
			g.Go(func() error {
				c, err := inv.load(ref)
				if err == nil {
					c.Name, c.Digest, c.ID = ref.Name, ref.Digest, ref.ID
					if c.SizeBytes == 0 {
						c.SizeBytes = ref.SizeBytes
					}
				}
				loaded[i] = row{digest: ref.Digest, cand: c, err: err}
				return nil
			})
		}
		_ = g.Wait()
		for i, ref := range stale {
			inv.rows[ref.Name] = loaded[i]
		}
	}

	out := make([]Candidate, 0, len(inv.rows))
	var failed map[string]error
	for name, r := range inv.rows {
		switch {
		case r.err == nil:
			out = append(out, r.cand)
		case errors.Is(r.err, ErrSkip):
		default:
			if failed == nil {
				failed = map[string]error{}
			}
			failed[name] = r.err
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, failed
}
