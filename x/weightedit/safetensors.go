package weightedit

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// stHeaderTensor is one entry in a safetensors JSON header.
type stHeaderTensor struct {
	Dtype       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// shardFile is one *.safetensors file opened for header inspection.
type shardFile struct {
	Path      string
	DataStart int64
	Tensors   map[string]stHeaderTensor
	Metadata  map[string]string
}

func readShardHeader(path string) (*shardFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("%s: read header size: %w", filepath.Base(path), err)
	}
	if headerSize > uint64(fi.Size()-8) || headerSize > 100<<20 {
		return nil, fmt.Errorf("%s: implausible safetensors header length %d", filepath.Base(path), headerSize)
	}
	raw := make([]byte, headerSize)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, fmt.Errorf("%s: read header: %w", filepath.Base(path), err)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw, &header); err != nil {
		return nil, fmt.Errorf("%s: parse header: %w", filepath.Base(path), err)
	}
	meta := map[string]string{}
	if mraw, ok := header["__metadata__"]; ok {
		delete(header, "__metadata__")
		_ = json.Unmarshal(mraw, &meta)
	}
	tensors := make(map[string]stHeaderTensor, len(header))
	dataStart := int64(8 + headerSize)
	for name, rawT := range header {
		var t stHeaderTensor
		if err := json.Unmarshal(rawT, &t); err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", filepath.Base(path), name, err)
		}
		if _, err := DTypeSize(t.Dtype); err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", filepath.Base(path), name, err)
		}
		numel := int64(1)
		for _, d := range t.Shape {
			numel *= d
		}
		size, _ := DTypeSize(t.Dtype)
		if t.DataOffsets[1]-t.DataOffsets[0] != numel*int64(size) {
			return nil, fmt.Errorf("%s: tensor %s: inconsistent offsets", filepath.Base(path), name)
		}
		if dataStart+t.DataOffsets[1] > fi.Size() {
			return nil, fmt.Errorf("%s: tensor %s: extends past EOF", filepath.Base(path), name)
		}
		tensors[name] = t
	}
	return &shardFile{Path: path, DataStart: dataStart, Tensors: tensors, Metadata: meta}, nil
}

func listSafetensorsShards(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".safetensors") {
			continue
		}
		if strings.Contains(name, "index") {
			continue
		}
		out = append(out, filepath.Join(dir, name))
	}
	sort.Strings(out)
	return out, nil
}
