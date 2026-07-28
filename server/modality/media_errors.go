package modality

import (
	"errors"
	"fmt"
	"net/http"
)

// ClientMediaError is a bad client payload (corrupt/unfetchable/unparseable media).
// SGLang #31417: map these to HTTP 400, not 500.
type ClientMediaError struct {
	Msg string
	Err error
}

func (e *ClientMediaError) Error() string {
	if e == nil {
		return ""
	}
	if e.Err != nil && e.Msg != "" {
		return e.Msg + ": " + e.Err.Error()
	}
	if e.Err != nil {
		return e.Err.Error()
	}
	return e.Msg
}

func (e *ClientMediaError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// ServerMediaError is a host/runtime fault (missing ffmpeg, temp IO, OOM).
// Map these to HTTP 500.
type ServerMediaError struct {
	Msg string
	Err error
}

func (e *ServerMediaError) Error() string {
	if e == nil {
		return ""
	}
	if e.Err != nil && e.Msg != "" {
		return e.Msg + ": " + e.Err.Error()
	}
	if e.Err != nil {
		return e.Err.Error()
	}
	return e.Msg
}

func (e *ServerMediaError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// ClientMedia returns a client-facing media error.
func ClientMedia(msg string) error {
	return &ClientMediaError{Msg: msg}
}

// ClientMediaf formats a client-facing media error.
func ClientMediaf(format string, args ...any) error {
	return &ClientMediaError{Msg: fmt.Sprintf(format, args...)}
}

// WrapClientMedia wraps err as a client media failure.
func WrapClientMedia(err error, msg string) error {
	if err == nil {
		return nil
	}
	var c *ClientMediaError
	if errors.As(err, &c) {
		return err
	}
	return &ClientMediaError{Msg: msg, Err: err}
}

// ServerMediaf formats a host media error.
func ServerMediaf(format string, args ...any) error {
	return &ServerMediaError{Msg: fmt.Sprintf(format, args...)}
}

// WrapServerMedia wraps err as a host media failure.
func WrapServerMedia(err error, msg string) error {
	if err == nil {
		return nil
	}
	var s *ServerMediaError
	if errors.As(err, &s) {
		return err
	}
	return &ServerMediaError{Msg: msg, Err: err}
}

// IsServerMedia reports whether err (or a wrap) is a host media fault.
func IsServerMedia(err error) bool {
	var s *ServerMediaError
	return errors.As(err, &s)
}

// IsClientMedia reports whether err (or a wrap) is a client media fault.
func IsClientMedia(err error) bool {
	var c *ClientMediaError
	return errors.As(err, &c)
}

// MediaHTTPStatus maps expand/demux errors to HTTP status.
// Unknown errors on the chat expand path default to 400 (legacy client-facing strings).
func MediaHTTPStatus(err error) int {
	if err == nil {
		return http.StatusOK
	}
	if IsServerMedia(err) {
		return http.StatusInternalServerError
	}
	return http.StatusBadRequest
}
