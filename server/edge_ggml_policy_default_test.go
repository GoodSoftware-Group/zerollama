//go:build !edge

package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestSchedSkipGgmlRunnerLoadDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_EDGE", "0")
	m := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	skip, err := schedSkipGgmlRunnerLoad(m)
	if skip || err != nil {
		t.Fatalf("skip=%v err=%v", skip, err)
	}
}
