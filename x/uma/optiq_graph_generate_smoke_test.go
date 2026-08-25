//go:build darwin && uma

package uma_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/x/uma"
)

// TestOptiqGraphGenerateRematch — F0699/F0719: in-process cgo rematch (no os/exec).
func TestOptiqGraphGenerateRematch(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE_SMOKE=1")
	}
	_ = os.Setenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE", "require")
	// Prove in-process: invalid BIN must not matter when dylib is linked.
	_ = os.Setenv("UMA_OPTIQ_GRAPH_GENERATE_BIN", "/nonexistent/optiq_graph_generate_bin")
	_ = os.Unsetenv("UMA_OPTIQ_GRAPH_GENERATE_ALLOW_EXEC")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	want := []int32{12675, 248046}

	// Dump-compat path (empty prompt → load prompt_ids.json).
	ids, err := uma.RunOptiqGraphGenerate(ctx, nil, 0)
	if err != nil {
		t.Fatalf("RunOptiqGraphGenerate dump: %v", err)
	}
	if uma.OptiqGraphGenerateLastMode != "in-process" {
		t.Fatalf("expected in-process mode, got %q", uma.OptiqGraphGenerateLastMode)
	}
	assertGenToks(t, ids, want)

	// F0719 explicit prompt from dump prompt_ids.json.
	prompt, err := loadOptiqDumpPromptIDs()
	if err != nil {
		t.Fatalf("load dump prompt: %v", err)
	}
	ids2, err := uma.RunOptiqGraphGenerate(ctx, prompt, len(want))
	if err != nil {
		t.Fatalf("RunOptiqGraphGenerate explicit prompt: %v", err)
	}
	assertGenToks(t, ids2, want)
	t.Logf("PASS: GRAPH_GEN_TOKENS rematch dump+explicit %v (in-process)", ids2)
}

func assertGenToks(t *testing.T, ids, want []int32) {
	t.Helper()
	if len(ids) != len(want) {
		t.Fatalf("len got=%d want=%d ids=%v", len(ids), len(want), ids)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("id[%d] got=%d want=%d full=%v", i, ids[i], want[i], ids)
		}
	}
}

func loadOptiqDumpPromptIDs() ([]int32, error) {
	dump := "/tmp/uma_optiq_generate_dump"
	if d := os.Getenv("ORNITH_OPTIQ_GENERATE_DIR"); d != "" {
		dump = d
	}
	b, err := os.ReadFile(filepath.Join(dump, "prompt_ids.json"))
	if err != nil {
		return nil, err
	}
	var raw []int
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	out := make([]int32, len(raw))
	for i, v := range raw {
		out[i] = int32(v)
	}
	if len(out) < 2 {
		return nil, fmt.Errorf("prompt_ids too short: %d", len(out))
	}
	return out, nil
}
