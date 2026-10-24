package autoroute

import (
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
)

type countingLoader struct {
	calls atomic.Int32
	fn    func(Ref) (Candidate, error)
}

func (l *countingLoader) load(ref Ref) (Candidate, error) {
	l.calls.Add(1)
	return l.fn(ref)
}

func names(cs []Candidate) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.Name
	}
	return out
}

func TestInventoryMemoisesByDigest(t *testing.T) {
	l := &countingLoader{fn: func(ref Ref) (Candidate, error) {
		return Candidate{Capabilities: []model.Capability{model.CapabilityCompletion}}, nil
	}}
	inv := NewInventory(l.load)

	refs := []Ref{{Name: "b:latest", Digest: "d1"}, {Name: "a:latest", Digest: "d2", SizeBytes: 42}}
	got, failed := inv.Refresh(refs)
	if len(failed) != 0 {
		t.Fatalf("failed = %v", failed)
	}
	if want := []string{"a:latest", "b:latest"}; !equal(names(got), want) {
		t.Fatalf("names = %v, want %v (sorted)", names(got), want)
	}
	if got[0].Digest != "d2" || got[0].SizeBytes != 42 {
		t.Fatalf("ref fields not copied: %+v", got[0])
	}
	if n := l.calls.Load(); n != 2 {
		t.Fatalf("first refresh loads = %d, want 2", n)
	}

	inv.Refresh(refs)
	if n := l.calls.Load(); n != 2 {
		t.Fatalf("unchanged digests reloaded: loads = %d, want 2", n)
	}

	refs[0].Digest = "d1b"
	inv.Refresh(refs)
	if n := l.calls.Load(); n != 3 {
		t.Fatalf("changed digest: loads = %d, want 3", n)
	}
}

func TestInventoryReloadsWhenLinkedFileChanges(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, []byte("v1"), 0o644); err != nil {
		t.Fatal(err)
	}
	l := &countingLoader{fn: func(ref Ref) (Candidate, error) {
		cur, err := os.Stat(path)
		if err != nil {
			return Candidate{}, err
		}
		return Candidate{
			Capabilities: []model.Capability{model.CapabilityCompletion},
			ModelFile:    path,
			FileSize:     cur.Size(),
			FileMtime:    cur.ModTime().UnixNano(),
		}, nil
	}}
	inv := NewInventory(l.load)
	refs := []Ref{{Name: "linked", Digest: "same"}}
	if _, failed := inv.Refresh(refs); len(failed) != 0 {
		t.Fatalf("failed = %v", failed)
	}
	if n := l.calls.Load(); n != 1 {
		t.Fatalf("loads = %d", n)
	}
	inv.Refresh(refs)
	if n := l.calls.Load(); n != 1 {
		t.Fatalf("unchanged file reloaded: %d", n)
	}
	// Rewrite in place (create --link target); digest stays the same.
	time.Sleep(10 * time.Millisecond) // ensure mtime advances on coarse filesystems
	if err := os.WriteFile(path, []byte("v2-longer"), 0o644); err != nil {
		t.Fatal(err)
	}
	inv.Refresh(refs)
	if n := l.calls.Load(); n != 2 {
		t.Fatalf("linked file change: loads = %d, want 2", n)
	}
}

func TestInventoryDropsRemovedModels(t *testing.T) {
	inv := NewInventory(func(Ref) (Candidate, error) { return Candidate{}, nil })
	inv.Refresh([]Ref{{Name: "a", Digest: "1"}, {Name: "b", Digest: "2"}})
	got, _ := inv.Refresh([]Ref{{Name: "b", Digest: "2"}})
	if want := []string{"b"}; !equal(names(got), want) {
		t.Fatalf("names = %v, want %v", names(got), want)
	}
}

func TestInventorySkipAndFailureAreCached(t *testing.T) {
	boom := errors.New("bad manifest")
	l := &countingLoader{fn: func(ref Ref) (Candidate, error) {
		switch ref.Name {
		case "cloud":
			return Candidate{}, ErrSkip
		case "broken":
			return Candidate{}, boom
		}
		return Candidate{}, nil
	}}
	inv := NewInventory(l.load)
	refs := []Ref{{Name: "cloud", Digest: "c"}, {Name: "broken", Digest: "x"}, {Name: "ok", Digest: "o"}}

	got, failed := inv.Refresh(refs)
	if want := []string{"ok"}; !equal(names(got), want) {
		t.Fatalf("names = %v, want %v", names(got), want)
	}
	if len(failed) != 1 || !errors.Is(failed["broken"], boom) {
		t.Fatalf("failed = %v, want only broken", failed)
	}
	inv.Refresh(refs)
	if n := l.calls.Load(); n != 3 {
		t.Fatalf("skip/failure retried: loads = %d, want 3", n)
	}
}

func TestCandidateHasAndBackend(t *testing.T) {
	c := Candidate{
		Capabilities: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		Backends:     map[string]string{model.ModalitySpeech: model.BackendPiper},
	}
	if !c.Has(model.CapabilityCompletion, model.CapabilityVision) {
		t.Fatal("expected completion+vision")
	}
	if c.Has(model.CapabilityVision, model.CapabilityTools) {
		t.Fatal("tools should be missing")
	}
	if c.Backend(model.ModalitySpeech) != model.BackendPiper || c.Backend(model.ModalityImage) != "" {
		t.Fatalf("backend lookup wrong: %v", c.Backends)
	}
}

func TestCandidatePerf(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	path := filepath.Join(home, ".ollama", "bench.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"sha256-fast":{"model":"m","kind":"completion","tok_per_sec":88},"sha256-empty":{"model":"e","kind":"completion"}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	if e, ok := (Candidate{Digest: "sha256-fast"}).Perf(); !ok || e.TokPerSec != 88 {
		t.Fatalf("fast perf = %+v %v", e, ok)
	}
	if _, ok := (Candidate{Digest: "sha256-empty"}).Perf(); ok {
		t.Fatal("entry without a usable metric should not count")
	}
	if _, ok := (Candidate{Digest: "sha256-none"}).Perf(); ok {
		t.Fatal("missing digest should not count")
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
