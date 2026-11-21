package weightedit

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
)

// ErrIncompatible means the Store is not the checkpoint this Edit claims.
var ErrIncompatible = errors.New("incompatible checkpoint")

// Bind checks edit against store for the given side ("source" or "result").
//
// Layout, config, and tokenizer claims always come from edit.Source, matching
// DAWE: those sidecars are not part of the weight fingerprint, so restore must
// still refuse a patched tree whose tokenizer was rewritten.
// Per-tensor hashes and the full fingerprint use the requested side.
func Bind(store Store, edit *Edit, side string) (*BoundEdit, error) {
	if edit == nil {
		return nil, fmt.Errorf("nil edit")
	}
	if side != "source" && side != "result" {
		return nil, fmt.Errorf("side must be source or result, got %q", side)
	}
	problems, err := checkClaims(store, edit, side)
	if err != nil {
		return nil, err
	}
	if len(problems) > 0 {
		return nil, fmt.Errorf("%w: %s", ErrIncompatible, formatProblems(edit, side, problems))
	}
	return &BoundEdit{Edit: edit, Store: store, Side: side}, nil
}

func expectedIdentity(edit *Edit, side string) (Identity, error) {
	switch side {
	case "source":
		return edit.Source, nil
	case "result":
		return edit.Result, nil
	default:
		return Identity{}, fmt.Errorf("unknown side %q", side)
	}
}

func checkClaims(store Store, edit *Edit, side string) ([]string, error) {
	var problems []string
	src := edit.Source
	wantSide, err := expectedIdentity(edit, side)
	if err != nil {
		return nil, err
	}

	// Sidecar and layout claims are always the source's — DAWE check_checkpoint
	// reads patch.manifest["source"] for these on both apply and restore.
	if src.LayoutSHA256 != "" {
		got, err := store.LayoutDigest()
		if err != nil {
			return nil, err
		}
		if got != src.LayoutSHA256 {
			names, _ := store.TensorNames()
			problems = append(problems, fmt.Sprintf(
				"tensor names/dtypes/shapes differ from the patch's checkpoint (%d tensors here, %d expected)",
				len(names), src.NTensors,
			))
		}
	}

	if len(src.Config) > 0 || len(src.Tokenizer) > 0 {
		cfg, tok, err := store.SideFiles()
		if err != nil {
			return nil, err
		}
		if len(src.Config) > 0 {
			if diff := mapDiff(src.Config, cfg); len(diff) > 0 {
				problems = append(problems, "config files differ: "+joinSorted(diff))
			}
		}
		if len(src.Tokenizer) > 0 {
			if diff := mapDiff(src.Tokenizer, tok); len(diff) > 0 {
				problems = append(problems, "tokenizer files differ: "+joinSorted(diff))
			}
		}
	}

	for i := range edit.Changes {
		c := &edit.Changes[i]
		if err := ValidateChange(*c); err != nil {
			problems = append(problems, err.Error())
			continue
		}
		info, err := store.Info(c.Name)
		if err != nil {
			problems = append(problems, fmt.Sprintf("missing tensor %s", c.Name))
			continue
		}
		if info.Dtype != c.Dtype || !shapesEqual(info.Shape, c.Shape) {
			problems = append(problems, fmt.Sprintf("%s: is %s%v, patch expects %s%v", c.Name, info.Dtype, info.Shape, c.Dtype, c.Shape))
			continue
		}
		wantHash := c.SourceSHA256
		if side == "result" {
			wantHash = c.ResultSHA256
		}
		if wantHash != "" {
			got, err := store.TensorHash(c.Name)
			if err != nil {
				return nil, err
			}
			if got != wantHash {
				problems = append(problems, fmt.Sprintf("%s: tensor content differs from the %s checkpoint", c.Name, side))
				continue
			}
		}
		expect := c.Old
		if side == "result" {
			expect = c.New
		}
		if err := checkIndexBytes(store, *c, expect); err != nil {
			problems = append(problems, err.Error())
		}
	}

	if wantSide.Fingerprint != "" && len(problems) == 0 {
		fp, _, err := fingerprintStore(store)
		if err != nil {
			return nil, err
		}
		if fp != wantSide.Fingerprint {
			problems = append(problems, "full-checkpoint fingerprint differs: some tensor has different values than the patch's checkpoint")
		}
	}
	return problems, nil
}

func fingerprintStore(store Store) (string, map[string]string, error) {
	if ds, ok := store.(*DirectoryStore); ok {
		return ds.Fingerprint()
	}
	names, err := store.TensorNames()
	if err != nil {
		return "", nil, err
	}
	hashes := make(map[string]string, len(names))
	infos := make([]TensorInfo, 0, len(names))
	for _, name := range names {
		h, err := store.TensorHash(name)
		if err != nil {
			return "", nil, err
		}
		hashes[name] = h
		info, err := store.Info(name)
		if err != nil {
			return "", nil, err
		}
		infos = append(infos, info)
	}
	return fingerprintLines(infos, hashes), hashes, nil
}

func checkIndexBytes(store Store, c Change, expect []byte) error {
	size, err := c.ItemSize()
	if err != nil {
		return err
	}
	span, err := store.Span(c.Name)
	if err != nil {
		return err
	}
	f, err := openReadAt(span.Path)
	if err != nil {
		return err
	}
	defer f.Close()

	indices := c.Indices
	if len(indices) == 0 {
		got := make([]byte, c.Numel()*int64(size))
		if _, err := f.ReadAt(got, span.Start); err != nil {
			return err
		}
		if string(got) != string(expect) {
			return fmt.Errorf("%s: stored bytes differ from the edit's claimed values", c.Name)
		}
		return nil
	}
	for i, idx := range indices {
		got := make([]byte, size)
		off := span.Start + int64(idx)*int64(size)
		if _, err := f.ReadAt(got, off); err != nil {
			return err
		}
		want := expect[i*size : (i+1)*size]
		if string(got) != string(want) {
			return fmt.Errorf("%s: index %d differs from the edit's claimed value", c.Name, idx)
		}
	}
	return nil
}

type readAtCloser interface {
	io.ReaderAt
	io.Closer
}

func openReadAt(path string) (readAtCloser, error) {
	return os.Open(path)
}

func mapDiff(want, got map[string]string) []string {
	keys := map[string]struct{}{}
	for k := range want {
		keys[k] = struct{}{}
	}
	for k := range got {
		keys[k] = struct{}{}
	}
	var diff []string
	for k := range keys {
		if want[k] != got[k] {
			diff = append(diff, k)
		}
	}
	return diff
}

func joinSorted(ss []string) string {
	sort.Strings(ss)
	out := ""
	for i, s := range ss {
		if i > 0 {
			out += ", "
		}
		out += s
	}
	return out
}

func shapesEqual(a, b []int64) bool {
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

func formatProblems(edit *Edit, side string, problems []string) string {
	short := edit.ID
	if len(short) > 12 {
		short = short[:12]
	}
	msg := fmt.Sprintf("not the patch's %s checkpoint (%s, id %s)", side, edit.Name, short)
	for _, p := range problems {
		msg += "\n - " + p
	}
	return msg
}
