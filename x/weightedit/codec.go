package weightedit

import (
	"bytes"
	"errors"
	"fmt"
	"io"
)

// ErrUnknownFormat means no registered codec recognized the bytes.
var ErrUnknownFormat = errors.New("unknown weight-edit format")

// ErrInvalidPatch means the bytes matched a codec but failed to decode or validate.
var ErrInvalidPatch = errors.New("invalid weight-edit patch")

// Codec turns a patch file into an Edit. It does not open the checkpoint.
type Codec interface {
	// Format is the stable name returned on inspect (e.g. "geodesic-patch").
	Format() string
	// Detect reports whether prefix is enough to claim this format.
	Detect(prefix []byte) bool
	// Decode reads a complete patch and returns an Edit.
	Decode(r io.Reader) (*Edit, error)
}

var codecs []Codec

// RegisterCodec appends a codec to the detection list. Earlier codecs win.
func RegisterCodec(c Codec) {
	codecs = append(codecs, c)
}

// Codecs returns the registered codecs in detection order.
func Codecs() []Codec {
	out := make([]Codec, len(codecs))
	copy(out, codecs)
	return out
}

// Detect picks the first codec whose Detect returns true for prefix.
func Detect(prefix []byte) Codec {
	for _, c := range codecs {
		if c.Detect(prefix) {
			return c
		}
	}
	return nil
}

// Decode sniffs the stream and decodes with the matching codec.
// The reader is fully consumed into memory (HTTP already caps the body).
func Decode(r io.Reader) (*Edit, Codec, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, nil, err
	}
	if len(data) == 0 {
		return nil, nil, fmt.Errorf("%w: empty body", ErrUnknownFormat)
	}
	c := Detect(data)
	if c == nil {
		return nil, nil, ErrUnknownFormat
	}
	edit, err := c.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, c, fmt.Errorf("%w: %v", ErrInvalidPatch, err)
	}
	return edit, c, nil
}
