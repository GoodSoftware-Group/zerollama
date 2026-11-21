package weightedit

import (
	"io"
)

// TensorInfo describes one named tensor in a Store.
type TensorInfo struct {
	Name  string
	Dtype string
	Shape []int64
}

// TensorSpan is the absolute byte range of a tensor's payload in a file.
type TensorSpan struct {
	Path  string
	Start int64 // absolute file offset of first element byte
	End   int64 // exclusive
}

// Store is a checkpoint the Bind/Apply pipeline can fingerprint and splice.
type Store interface {
	// Root is the directory (or equivalent) that holds the checkpoint files.
	Root() string
	// LayoutDigest is a hash of every tensor name, dtype, and shape.
	LayoutDigest() (string, error)
	// TensorNames returns sorted tensor names.
	TensorNames() ([]string, error)
	// Info returns dtype and shape for a tensor.
	Info(name string) (TensorInfo, error)
	// Span returns the on-disk byte range of a tensor's payload.
	Span(name string) (TensorSpan, error)
	// TensorHash returns sha256 of the tensor payload bytes.
	TensorHash(name string) (string, error)
	// SideFiles returns sha256 of present config and tokenizer files.
	SideFiles() (config, tokenizer map[string]string, err error)
	// OpenTensor returns a reader over the tensor payload.
	OpenTensor(name string) (io.ReadCloser, error)
}

// BoundEdit is an Edit that has been checked against a Store.
type BoundEdit struct {
	Edit  *Edit
	Store Store
	// Identity is the expected fingerprint side: "source" for Apply, "result" for Restore.
	Side string
}
