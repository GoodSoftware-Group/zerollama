// Package weightedit is a decode → bind → apply pipeline for stored weight edits.
//
// Codecs turn patch files into an Edit. Bind checks that Edit against a Store.
// Apply writes a new checkpoint directory. The Edit record is the stable
// contract; patch file formats are codecs in front of it.
package weightedit

import (
	"fmt"
)

// Identity claims the author made about the source (or result) checkpoint.
// Bind verifies every claim that is present. Omitted claims are not invented.
type Identity struct {
	LayoutSHA256 string            `json:"layout_sha256,omitempty"`
	Fingerprint  string            `json:"fingerprint,omitempty"`
	NTensors     int               `json:"n_tensors,omitempty"`
	Config       map[string]string `json:"config,omitempty"`
	Tokenizer    map[string]string `json:"tokenizer,omitempty"`
}

// Change is one stored tensor's sparse (or whole-tensor) replacement.
// Indices are row-major into Shape. Empty Indices means the whole tensor.
// Old and New are raw little-endian element bytes in the stored dtype.
type Change struct {
	Name         string
	Dtype        string
	Shape        []int64
	Indices      []uint64
	Old          []byte
	New          []byte
	SourceSHA256 string
	ResultSHA256 string
}

// Numel returns the product of Shape.
func (c Change) Numel() int64 {
	n := int64(1)
	for _, d := range c.Shape {
		n *= d
	}
	return n
}

// ItemSize returns the byte width of one element of Dtype.
func (c Change) ItemSize() (int, error) {
	return DTypeSize(c.Dtype)
}

// Edit is the intermediate form codecs produce and Apply consumes.
type Edit struct {
	Format  string
	Name    string
	ID      string
	Source  Identity
	Result  Identity
	Changes []Change
}

// ChangedWeights returns the number of stored elements this edit replaces.
func (e *Edit) ChangedWeights() int {
	n := 0
	for _, c := range e.Changes {
		if len(c.Indices) == 0 {
			n += int(c.Numel())
			continue
		}
		n += len(c.Indices)
	}
	return n
}

// TouchedRows returns the distinct first-axis row indices that contain changes
// for tensors with at least two axes. Nil when Shape has fewer than two axes.
func (c Change) TouchedRows() []int64 {
	if len(c.Shape) < 2 {
		return nil
	}
	width := int64(1)
	for _, d := range c.Shape[1:] {
		width *= d
	}
	seen := map[int64]struct{}{}
	var rows []int64
	add := func(flat uint64) {
		row := int64(flat) / width
		if _, ok := seen[row]; ok {
			return
		}
		seen[row] = struct{}{}
		rows = append(rows, row)
	}
	if len(c.Indices) == 0 {
		for i := int64(0); i < c.Shape[0]; i++ {
			rows = append(rows, i)
		}
		return rows
	}
	for _, idx := range c.Indices {
		add(idx)
	}
	return rows
}

// ValidateChange checks indices and old/new byte lengths.
func ValidateChange(c Change) error {
	size, err := c.ItemSize()
	if err != nil {
		return err
	}
	numel := c.Numel()
	if numel <= 0 {
		return fmt.Errorf("%s: invalid shape %v", c.Name, c.Shape)
	}
	k := len(c.Indices)
	if k == 0 {
		want := int(numel) * size
		if len(c.Old) != want || len(c.New) != want {
			return fmt.Errorf("%s: whole-tensor value bytes do not match dtype %s shape %v", c.Name, c.Dtype, c.Shape)
		}
		return nil
	}
	if len(c.Old) != k*size || len(c.New) != k*size {
		return fmt.Errorf("%s: value bytes do not match %d indices of dtype %s", c.Name, k, c.Dtype)
	}
	if c.Indices[0] < 0 || int64(c.Indices[k-1]) >= numel {
		return fmt.Errorf("%s: index out of range for %d elements", c.Name, numel)
	}
	for i := 1; i < k; i++ {
		if c.Indices[i] <= c.Indices[i-1] {
			return fmt.Errorf("%s: indices must be strictly increasing", c.Name)
		}
	}
	return nil
}

// DTypeSize returns the stored byte width of a safetensors dtype name.
func DTypeSize(dtype string) (int, error) {
	switch dtype {
	case "F64", "I64", "U64":
		return 8, nil
	case "F32", "I32", "U32":
		return 4, nil
	case "F16", "BF16", "I16", "U16":
		return 2, nil
	case "F8_E4M3", "F8_E5M2", "I8", "U8", "BOOL":
		return 1, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
}
