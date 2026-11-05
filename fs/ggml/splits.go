package ggml

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// tensorModel is a GGML model backed by an explicit KV map and tensor list.
// Used after merging multi-shard GGUF files so Tensors() returns every shard's weights.
type tensorModel struct {
	kv      KV
	tensors Tensors
}

func (m *tensorModel) KV() KV           { return m.kv }
func (m *tensorModel) Tensors() Tensors { return m.tensors }

// MergeSplits loads sibling GGUF shards when meta has split.count > 1 and returns
// a GGML whose Tensors() includes every shard's weights. Each Tensor has File and
// Base set so readers open the correct shard at the correct data offset.
//
// primaryPath must be the first shard (*-00001-of-NNNNN.gguf, split.no=0).
func MergeSplits(primaryPath string, g *GGML) (*GGML, error) {
	if g == nil {
		return nil, fmt.Errorf("nil ggml")
	}
	primaryPath = strings.TrimSpace(primaryPath)
	if primaryPath == "" {
		return nil, fmt.Errorf("empty gguf path")
	}

	nSplit := kvAsInt(g.KV(), "split.count")
	if nSplit <= 1 {
		tagTensorSources(primaryPath, g.Tensors().Offset, g.Tensors().Items())
		return g, nil
	}

	if no := kvAsInt(g.KV(), "split.no"); no != 0 {
		return nil, fmt.Errorf("gguf split: must open first shard (split.no=0), got split.no=%d path=%s", no, primaryPath)
	}

	paths, err := SplitPaths(primaryPath, nSplit)
	if err != nil {
		return nil, err
	}

	kv := g.KV()
	items := make([]*Tensor, 0, kvAsInt(kv, "split.tensors.count"))
	var params uint64

	addShard := func(path string, shard *GGML) {
		base := shard.Tensors().Offset
		for _, t := range shard.Tensors().Items() {
			if t == nil {
				continue
			}
			t.File = path
			t.Base = base
			items = append(items, t)
			params += t.Elements()
		}
	}

	addShard(paths[0], g)

	for i := 1; i < nSplit; i++ {
		f, err := os.Open(paths[i])
		if err != nil {
			return nil, fmt.Errorf("gguf split %d: open %s: %w", i+1, paths[i], err)
		}
		// Headers + tensorOffset only — avoid seeking through multi-GB weight regions.
		shard, err := DecodeMetadata(f)
		_ = f.Close()
		if err != nil {
			return nil, fmt.Errorf("gguf split %d: decode %s: %w", i+1, paths[i], err)
		}
		if got := kvAsInt(shard.KV(), "split.no"); got != i {
			return nil, fmt.Errorf("gguf split: %s has split.no=%d, expected %d", paths[i], got, i)
		}
		addShard(paths[i], shard)
	}

	if expected := kvAsInt(kv, "split.tensors.count"); expected > 0 && len(items) != expected {
		return nil, fmt.Errorf("gguf split: got %d tensors, split.tensors.count=%d", len(items), expected)
	}

	kv["general.parameter_count"] = params

	return &GGML{
		container: g.container,
		model: &tensorModel{
			kv: kv,
			tensors: Tensors{
				items:  items,
				Offset: g.Tensors().Offset,
			},
		},
		Length: g.Length,
	}, nil
}

func tagTensorSources(path string, base uint64, items []*Tensor) {
	for _, t := range items {
		if t == nil {
			continue
		}
		t.File = path
		t.Base = base
	}
}

func kvAsInt(kv KV, key string) int {
	switch v := kv[key].(type) {
	case uint8:
		return int(v)
	case uint16:
		return int(v)
	case uint32:
		return int(v)
	case uint64:
		return int(v)
	case int8:
		return int(v)
	case int16:
		return int(v)
	case int32:
		return int(v)
	case int64:
		return int(v)
	default:
		return 0
	}
}

// SplitPaths returns absolute paths for all shards given the first-shard path
// and split count. Naming follows llama.cpp: <prefix>-%05d-of-%05d.gguf (1-based).
func SplitPaths(primaryPath string, nSplit int) ([]string, error) {
	if nSplit < 1 {
		return nil, fmt.Errorf("invalid split count %d", nSplit)
	}
	if nSplit == 1 {
		return []string{primaryPath}, nil
	}

	base := filepath.Base(primaryPath)
	dir := filepath.Dir(primaryPath)
	idx, n, ok := ParseSplitName(base)
	if !ok {
		return nil, fmt.Errorf("gguf split: filename does not match *-NNNNN-of-NNNNN.gguf: %s", base)
	}
	if idx != 1 {
		return nil, fmt.Errorf("gguf split: expected first shard (00001), got index %d in %s", idx, base)
	}
	if n != nSplit {
		return nil, fmt.Errorf("gguf split: filename says %d parts, split.count=%d", n, nSplit)
	}

	suffix := fmt.Sprintf("-%05d-of-%05d.gguf", 1, nSplit)
	if !strings.HasSuffix(base, suffix) {
		return nil, fmt.Errorf("gguf split: could not strip suffix %q from %s", suffix, base)
	}
	prefix := strings.TrimSuffix(base, suffix)

	out := make([]string, nSplit)
	for i := 0; i < nSplit; i++ {
		name := fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, i+1, nSplit)
		p := filepath.Join(dir, name)
		if _, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("gguf split missing shard %s: %w", p, err)
		}
		out[i] = p
	}
	return out, nil
}

// ParseSplitName parses llama.cpp split filenames: prefix-0000I-of-0000N.gguf.
// Returns 1-based shard index, total count, and ok.
func ParseSplitName(base string) (idx, n int, ok bool) {
	const sep = "-of-"
	if !strings.HasSuffix(base, ".gguf") {
		return 0, 0, false
	}
	stem := strings.TrimSuffix(base, ".gguf")
	of := strings.LastIndex(stem, sep)
	if of < 0 {
		return 0, 0, false
	}
	right := stem[of+len(sep):]
	left := stem[:of]
	dash := strings.LastIndex(left, "-")
	if dash < 0 {
		return 0, 0, false
	}
	idxStr := left[dash+1:]
	i, err1 := strconv.Atoi(idxStr)
	total, err2 := strconv.Atoi(right)
	if err1 != nil || err2 != nil || i < 1 || total < 1 || i > total {
		return 0, 0, false
	}
	return i, total, true
}

// DataPath returns the GGUF file that holds t's weights.
func (t *Tensor) DataPath(defaultPath string) string {
	if t != nil && t.File != "" {
		return t.File
	}
	return defaultPath
}

// AbsoluteOffset returns the file offset of t's weight bytes.
func (t *Tensor) AbsoluteOffset(defaultBase uint64) int64 {
	if t == nil {
		return 0
	}
	if t.File != "" {
		return int64(t.Base + t.Offset)
	}
	return int64(defaultBase + t.Offset)
}
