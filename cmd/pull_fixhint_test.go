package cmd

import (
	"errors"
	"strings"
	"testing"
)

func TestPullFixHintConnectionRefused(t *testing.T) {
	err := pullFixHint(errors.New("dial tcp 127.0.0.1:11434: connection refused"))
	if err == nil || !strings.Contains(err.Error(), "zerollama serve") {
		t.Fatalf("got %v", err)
	}
}

func TestPullFixHintNotFound(t *testing.T) {
	err := pullFixHint(errors.New("model not found: 404"))
	if err == nil || !strings.Contains(err.Error(), "storage doctor") {
		t.Fatalf("got %v", err)
	}
}
