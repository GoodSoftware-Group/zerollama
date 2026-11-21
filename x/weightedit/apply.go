package weightedit

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const provenanceFile = "geodesic_applied.json"

// ApplyResult summarizes a successful write.
type ApplyResult struct {
	Operation         string   `json:"operation"`
	Out               string   `json:"out"`
	ChangedWeights    int      `json:"changed_weights"`
	Tensors           []string `json:"tensors"`
	OutputFingerprint string   `json:"output_fingerprint,omitempty"`
	FingerprintOK     bool     `json:"fingerprint_verified"`
}

// Apply writes out = store + bound edit. Restoring uses Side == "result" and
// swaps Old/New. The source inode is never opened for write.
func Apply(bound *BoundEdit, out string) (*ApplyResult, error) {
	if bound == nil || bound.Edit == nil || bound.Store == nil {
		return nil, fmt.Errorf("nil bound edit")
	}
	direction := "apply"
	afterSide := "result"
	if bound.Side == "result" {
		direction = "restore"
		afterSide = "source"
	}

	srcRoot := bound.Store.Root()
	outAbs, err := filepath.Abs(out)
	if err != nil {
		return nil, err
	}
	srcAbs, err := filepath.Abs(srcRoot)
	if err != nil {
		return nil, err
	}
	if err := checkOutPath(srcAbs, outAbs); err != nil {
		return nil, err
	}

	touched := map[string]Change{}
	for _, c := range bound.Edit.Changes {
		touched[c.Name] = c
	}
	touchedShards := map[string]struct{}{}
	for name := range touched {
		span, err := bound.Store.Span(name)
		if err != nil {
			return nil, err
		}
		touchedShards[span.Path] = struct{}{}
	}

	tmp, err := prepareOut(srcAbs, outAbs)
	if err != nil {
		return nil, err
	}
	defer func() {
		if tmp != "" {
			_ = os.RemoveAll(tmp)
		}
	}()

	if err := filepath.Walk(srcAbs, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(srcAbs, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		dst := filepath.Join(tmp, rel)
		if info.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		if info.Name() == provenanceFile {
			return nil
		}
		linkMode := "hardlink"
		if _, ok := touchedShards[path]; ok {
			linkMode = "copy"
		}
		return placeFile(path, dst, linkMode)
	}); err != nil {
		return nil, err
	}

	for _, c := range bound.Edit.Changes {
		values := c.New
		wantAfter := c.ResultSHA256
		if direction == "restore" {
			values = c.Old
			wantAfter = c.SourceSHA256
		}
		if err := spliceChange(srcAbs, tmp, bound.Store, c, values, wantAfter); err != nil {
			return nil, err
		}
	}

	var outFP string
	fpVerified := false
	wantAfter, _ := expectedIdentity(bound.Edit, afterSide)
	if wantAfter.Fingerprint != "" {
		outStore, err := OpenDirectory(tmp)
		if err != nil {
			return nil, fmt.Errorf("fingerprint written checkpoint: %w", err)
		}
		gotFP, _, err := outStore.Fingerprint()
		if err != nil {
			return nil, fmt.Errorf("fingerprint written checkpoint: %w", err)
		}
		if gotFP != wantAfter.Fingerprint {
			return nil, fmt.Errorf("written checkpoint fingerprint does not match the edit's %s fingerprint", afterSide)
		}
		outFP = gotFP
		fpVerified = true
	}

	inputFP := bound.Edit.Source.Fingerprint
	if direction == "restore" {
		inputFP = bound.Edit.Result.Fingerprint
	}
	prov := map[string]any{
		"tool":                 bound.Edit.Format,
		"operation":            direction,
		"patch_name":           bound.Edit.Name,
		"patch_id":             bound.Edit.ID,
		"input_fingerprint":    inputFP,
		"output_fingerprint":   outFP,
		"fingerprint_verified": fpVerified,
	}
	provBytes, err := json.MarshalIndent(prov, "", " ")
	if err != nil {
		return nil, err
	}
	provBytes = append(provBytes, '\n')
	if err := os.WriteFile(filepath.Join(tmp, provenanceFile), provBytes, 0o644); err != nil {
		return nil, err
	}

	if err := os.Rename(tmp, outAbs); err != nil {
		// Rename across devices falls back to copy.
		if err2 := copyDir(tmp, outAbs); err2 != nil {
			return nil, fmt.Errorf("rename out: %v; copy fallback: %w", err, err2)
		}
		_ = os.RemoveAll(tmp)
	}
	tmp = "" // ownership transferred

	names := make([]string, 0, len(bound.Edit.Changes))
	for _, c := range bound.Edit.Changes {
		names = append(names, c.Name)
	}
	return &ApplyResult{
		Operation:         direction,
		Out:               outAbs,
		ChangedWeights:    bound.Edit.ChangedWeights(),
		Tensors:           names,
		OutputFingerprint: outFP,
		FingerprintOK:     fpVerified,
	}, nil
}

// Restore is Apply with Side forced to "result".
func Restore(store Store, edit *Edit, out string) (*ApplyResult, error) {
	bound, err := Bind(store, edit, "result")
	if err != nil {
		return nil, err
	}
	return Apply(bound, out)
}

func checkOutPath(src, out string) error {
	if out == src {
		return fmt.Errorf("output directory must be separate from the input directory (patches never edit in place)")
	}
	sep := string(filepath.Separator)
	if strings.HasPrefix(out+sep, src+sep) || strings.HasPrefix(src+sep, out+sep) {
		return fmt.Errorf("output directory must be separate from the input directory (patches never edit in place)")
	}
	if fi, err := os.Stat(out); err == nil {
		if !fi.IsDir() {
			return fmt.Errorf("%s already exists and is not a directory", out)
		}
		entries, err := os.ReadDir(out)
		if err != nil {
			return err
		}
		if len(entries) > 0 {
			return fmt.Errorf("%s already exists and is not empty", out)
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func prepareOut(src, out string) (string, error) {
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return "", err
	}
	tmp := filepath.Join(filepath.Dir(out), "."+filepath.Base(out)+".weightedit-tmp")
	if err := os.RemoveAll(tmp); err != nil {
		return "", err
	}
	if err := os.Mkdir(tmp, 0o755); err != nil {
		return "", err
	}
	_ = src
	return tmp, nil
}

func placeFile(src, dst, mode string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	switch mode {
	case "copy":
		return copyFile(src, dst)
	case "hardlink":
		if err := os.Link(src, dst); err != nil {
			return copyFile(src, dst)
		}
		return nil
	case "symlink":
		return os.Symlink(src, dst)
	default:
		return fmt.Errorf("unknown place mode %q", mode)
	}
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	// Source may be read-only (e.g. HF cache); ensure we can splice later.
	_ = out.Chmod(0o644)
	return out.Sync()
}

func copyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		return copyFile(path, target)
	})
}

func spliceChange(srcRoot, tmpRoot string, store Store, c Change, values []byte, wantAfter string) error {
	span, err := store.Span(c.Name)
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(srcRoot, span.Path)
	if err != nil {
		return err
	}
	dstPath := filepath.Join(tmpRoot, rel)
	size, err := c.ItemSize()
	if err != nil {
		return err
	}

	f, err := os.OpenFile(dstPath, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	defer f.Close()

	indices := c.Indices
	if len(indices) == 0 {
		if _, err := f.WriteAt(values, span.Start); err != nil {
			return err
		}
	} else {
		i := 0
		for i < len(indices) {
			j := i
			for j+1 < len(indices) && indices[j+1] == indices[j]+1 {
				j++
			}
			lo, hi, pos := int(indices[i]), int(indices[j])+1, i
			chunk := values[pos*size : (pos+hi-lo)*size]
			off := span.Start + int64(lo)*int64(size)
			if _, err := f.WriteAt(chunk, off); err != nil {
				return err
			}
			i = j + 1
		}
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if wantAfter != "" {
		got, err := sha256Range(dstPath, span.Start, span.End)
		if err != nil {
			return err
		}
		if got != wantAfter {
			return fmt.Errorf("%s: written tensor does not match the expected hash (bug or disk error)", c.Name)
		}
	}
	return nil
}
