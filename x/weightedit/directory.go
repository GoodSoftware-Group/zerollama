package weightedit

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

var (
	configSideFiles = []string{
		"config.json",
		"generation_config.json",
	}
	tokenizerSideFiles = []string{
		"tokenizer.json",
		"tokenizer_config.json",
		"vocab.json",
		"merges.txt",
		"special_tokens_map.json",
		"added_tokens.json",
		"chat_template.jinja",
		"chat_template.json",
	}
)

// DirectoryStore is a Hugging Face checkpoint: config.json + *.safetensors.
type DirectoryStore struct {
	root   string
	shards []*shardFile
	where  map[string]*shardFile
	infos  map[string]TensorInfo
	order  []string
}

// OpenDirectory opens a safetensors model directory.
func OpenDirectory(root string) (*DirectoryStore, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	fi, err := os.Stat(abs)
	if err != nil {
		return nil, err
	}
	if !fi.IsDir() {
		return nil, fmt.Errorf("%s: not a directory", abs)
	}
	if _, err := os.Stat(filepath.Join(abs, "config.json")); err != nil {
		return nil, fmt.Errorf("%s: missing config.json", abs)
	}
	paths, err := listSafetensorsShards(abs)
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("%s: no .safetensors files", abs)
	}
	ds := &DirectoryStore{
		root:  abs,
		where: make(map[string]*shardFile),
		infos: make(map[string]TensorInfo),
	}
	for _, p := range paths {
		sh, err := readShardHeader(p)
		if err != nil {
			return nil, err
		}
		ds.shards = append(ds.shards, sh)
		for name, t := range sh.Tensors {
			if _, ok := ds.where[name]; ok {
				return nil, fmt.Errorf("tensor %s appears in two shards of %s", name, abs)
			}
			ds.where[name] = sh
			ds.infos[name] = TensorInfo{Name: name, Dtype: t.Dtype, Shape: append([]int64(nil), t.Shape...)}
			ds.order = append(ds.order, name)
		}
	}
	sort.Strings(ds.order)
	return ds, nil
}

func (d *DirectoryStore) Root() string { return d.root }

func (d *DirectoryStore) TensorNames() ([]string, error) {
	out := make([]string, len(d.order))
	copy(out, d.order)
	return out, nil
}

func (d *DirectoryStore) Info(name string) (TensorInfo, error) {
	info, ok := d.infos[name]
	if !ok {
		return TensorInfo{}, fmt.Errorf("missing tensor %s", name)
	}
	return info, nil
}

func (d *DirectoryStore) Span(name string) (TensorSpan, error) {
	sh, ok := d.where[name]
	if !ok {
		return TensorSpan{}, fmt.Errorf("missing tensor %s", name)
	}
	t := sh.Tensors[name]
	return TensorSpan{
		Path:  sh.Path,
		Start: sh.DataStart + t.DataOffsets[0],
		End:   sh.DataStart + t.DataOffsets[1],
	}, nil
}

func (d *DirectoryStore) TensorHash(name string) (string, error) {
	span, err := d.Span(name)
	if err != nil {
		return "", err
	}
	return sha256Range(span.Path, span.Start, span.End)
}

func (d *DirectoryStore) LayoutDigest() (string, error) {
	infos := make([]TensorInfo, 0, len(d.order))
	for _, name := range d.order {
		infos = append(infos, d.infos[name])
	}
	return layoutDigestLines(infos), nil
}

func (d *DirectoryStore) Fingerprint() (string, map[string]string, error) {
	hashes := make(map[string]string, len(d.order))
	infos := make([]TensorInfo, 0, len(d.order))
	for _, name := range d.order {
		h, err := d.TensorHash(name)
		if err != nil {
			return "", nil, err
		}
		hashes[name] = h
		infos = append(infos, d.infos[name])
	}
	return fingerprintLines(infos, hashes), hashes, nil
}

func (d *DirectoryStore) SideFiles() (config, tokenizer map[string]string, err error) {
	config = map[string]string{}
	tokenizer = map[string]string{}
	for _, name := range configSideFiles {
		p := filepath.Join(d.root, name)
		if fi, e := os.Stat(p); e == nil && fi.Mode().IsRegular() {
			h, e := sha256File(p)
			if e != nil {
				return nil, nil, e
			}
			config[name] = h
		}
	}
	for _, name := range tokenizerSideFiles {
		p := filepath.Join(d.root, name)
		if fi, e := os.Stat(p); e == nil && fi.Mode().IsRegular() {
			h, e := sha256File(p)
			if e != nil {
				return nil, nil, e
			}
			tokenizer[name] = h
		}
	}
	return config, tokenizer, nil
}

func (d *DirectoryStore) OpenTensor(name string) (io.ReadCloser, error) {
	span, err := d.Span(name)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(span.Path)
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(span.Start, io.SeekStart); err != nil {
		f.Close()
		return nil, err
	}
	return &limitedReadCloser{f: f, left: span.End - span.Start}, nil
}

// Resolve maps a framework parameter name onto this checkpoint's key
// (HF modules vs saved names differ by a "model." prefix).
func (d *DirectoryStore) Resolve(name string) (string, error) {
	cands := []string{name, stringsTrimPrefix(name, "model."), "model." + name}
	var hits []string
	seen := map[string]struct{}{}
	for _, c := range cands {
		if _, ok := seen[c]; ok {
			continue
		}
		seen[c] = struct{}{}
		if _, ok := d.where[c]; ok {
			hits = append(hits, c)
		}
	}
	if len(hits) != 1 {
		return "", fmt.Errorf("cannot match tensor %q to the checkpoint (%d candidates among %v)", name, len(hits), cands)
	}
	return hits[0], nil
}

func stringsTrimPrefix(s, prefix string) string {
	if len(s) >= len(prefix) && s[:len(prefix)] == prefix {
		return s[len(prefix):]
	}
	return s
}

type limitedReadCloser struct {
	f    *os.File
	left int64
}

func (l *limitedReadCloser) Read(p []byte) (int, error) {
	if l.left <= 0 {
		return 0, io.EOF
	}
	if int64(len(p)) > l.left {
		p = p[:l.left]
	}
	n, err := l.f.Read(p)
	l.left -= int64(n)
	return n, err
}

func (l *limitedReadCloser) Close() error { return l.f.Close() }
