package server

import (
	"errors"
	"testing"
)

func TestShowErrStage(t *testing.T) {
	wrapped := showErr("parse_manifest", errors.New("bad digest"))
	if got := showErrorStage(wrapped); got != "parse_manifest" {
		t.Fatalf("stage = %q want parse_manifest", got)
	}
	if wrapped.Error() != "parse_manifest: bad digest" {
		t.Fatalf("Error() = %q", wrapped.Error())
	}
	if unwrapped := errors.Unwrap(wrapped); unwrapped == nil || unwrapped.Error() != "bad digest" {
		t.Fatal("Unwrap failed")
	}
}

func TestShowErrNoDoubleWrap(t *testing.T) {
	inner := showErr("load_model", errors.New("missing blob"))
	again := showErr("outer", inner)
	if got := showErrorStage(again); got != "load_model" {
		t.Fatalf("stage = %q want load_model (no double wrap)", got)
	}
}
