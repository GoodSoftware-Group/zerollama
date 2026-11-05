package ggml

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// ReadTensorBytes reads raw weight bytes for a named tensor from a GGUF file.
func ReadTensorBytes(path, name string) ([]byte, *Tensor, error) {
	path = strings.TrimSpace(path)
	name = strings.TrimSpace(name)
	if path == "" {
		return nil, nil, fmt.Errorf("empty gguf path")
	}
	if name == "" {
		return nil, nil, fmt.Errorf("empty tensor name")
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	meta, err := Decode(f, 0)
	if err != nil {
		return nil, nil, err
	}

	meta, err = MergeSplits(path, meta)
	if err != nil {
		return nil, nil, err
	}

	var tensor *Tensor
	for _, t := range meta.Tensors().Items() {
		if t != nil && t.Name == name {
			tensor = t
			break
		}
	}
	if tensor == nil {
		return nil, nil, fmt.Errorf("tensor %q not found in %s", name, path)
	}

	tensorPath := tensor.DataPath(path)
	rf, err := os.Open(tensorPath)
	if err != nil {
		return nil, nil, err
	}
	defer rf.Close()

	off := tensor.AbsoluteOffset(meta.Tensors().Offset)
	if _, err := rf.Seek(off, io.SeekStart); err != nil {
		return nil, nil, fmt.Errorf("seek tensor %q: %w", name, err)
	}

	size := int64(tensor.Size())
	buf := make([]byte, size)
	if _, err := io.ReadFull(rf, buf); err != nil {
		return nil, nil, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return buf, tensor, nil
}
