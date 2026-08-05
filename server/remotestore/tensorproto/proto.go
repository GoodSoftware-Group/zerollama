// Package tensorproto defines the per-tensor fetch wire protocol.
//
// Why ship the spec in v1 without a llama.cpp consumer: load-time stream mode
// (llama_model_init_from_user) and runtime tensor cache must share one request
// language. Specifying types/errors now avoids a second redesign when C++ patches land.
package tensorproto

import "fmt"

// Request asks for a byte range of a named tensor inside a content-addressed blob.
type Request struct {
	// Digest is the model weight blob (sha256-…).
	Digest string `json:"digest"`
	// TensorRef is a GGUF tensor name or module-role string (e.g. layer.3.expert.7).
	TensorRef string `json:"tensor_ref"`
	// Offset is relative to the start of the tensor's bytes (not the file).
	Offset int64 `json:"offset"`
	// Length is bytes to return; <=0 means the remainder of the tensor.
	Length int64 `json:"length"`
}

// ErrorCode classifies structured failures.
type ErrorCode string

const (
	ErrNotFound            ErrorCode = "not-found"
	ErrUpstreamUnavailable ErrorCode = "upstream-unavailable"
	ErrChecksumMismatch    ErrorCode = "checksum-mismatch"
	ErrInvalidRequest      ErrorCode = "invalid-request"
)

// ResponseMeta is returned as HTTP headers / JSON envelope around raw bytes.
type ResponseMeta struct {
	Digest    string    `json:"digest"`
	TensorRef string    `json:"tensor_ref"`
	Name      string    `json:"name,omitempty"`
	Role      string    `json:"role,omitempty"`
	Offset    int64     `json:"offset"`
	Length    int64     `json:"length"`
	Error     ErrorCode `json:"error,omitempty"`
	Message   string    `json:"message,omitempty"`
}

func (e ErrorCode) Error() string { return string(e) }

// Validate checks required fields on a Request.
func (r Request) Validate() error {
	if r.Digest == "" {
		return fmt.Errorf("%w: digest required", ErrInvalidRequest)
	}
	if r.TensorRef == "" {
		return fmt.Errorf("%w: tensor_ref required", ErrInvalidRequest)
	}
	if r.Offset < 0 {
		return fmt.Errorf("%w: negative offset", ErrInvalidRequest)
	}
	return nil
}
