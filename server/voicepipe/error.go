package voicepipe

// Error is a realtime socket failure with an optional machine-readable code.
type Error struct {
	Message string
	Code    string
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	return e.Message
}
