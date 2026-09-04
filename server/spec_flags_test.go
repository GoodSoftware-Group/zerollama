package server

import (
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestApplySpecFlags(t *testing.T) {
	off := false
	var dst llm.CompletionRequest
	applySpecFlags(&dst, map[string]any{"enable_pld": true, "enable_mtp": false}, nil, nil, nil)
	if dst.EnablePLD == nil || !*dst.EnablePLD {
		t.Fatalf("options enable_pld = %v", dst.EnablePLD)
	}
	if dst.EnableMTP == nil || *dst.EnableMTP {
		t.Fatalf("options enable_mtp = %v", dst.EnableMTP)
	}

	dst = llm.CompletionRequest{}
	applySpecFlags(&dst, map[string]any{"enable_pld": true}, &off, nil, nil)
	if dst.EnablePLD == nil || *dst.EnablePLD {
		t.Fatal("explicit field must win over options")
	}

	dst = llm.CompletionRequest{}
	applySpecFlags(&dst, nil, nil, nil, &off)
	if dst.EnableMTP == nil || *dst.EnableMTP {
		t.Fatal("enable_drafter aliases enable_mtp")
	}
}
