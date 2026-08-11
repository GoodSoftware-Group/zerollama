//go:build darwin && uma

package uma_test

import (
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/uma"
)

// Lab smoke: FormatGraph + Submit/Wait NOP→MARK (wishlist GRAPH-MLX 0.4 / F0624).
// Requires uma_daemon. Skip unless ZEROLLAMA_UMA_GRAPH_SMOKE=1.
func TestGraphFormatAndSubmit(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_GRAPH_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_GRAPH_SMOKE=1")
	}
	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-graph")
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()
	if !uma.Active() {
		t.Fatal("expected active broker gate")
	}

	job, err := uma.FormatGraph(1, "chain", "NOP@CPU! ; MARK@GPU?")
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if !strings.Contains(job, "GRAPH") || !strings.Contains(job, "NOP@CPU!") {
		t.Fatalf("unexpected job: %s", job)
	}
	if !strings.Contains(job, "; NOP@CPU!") {
		t.Fatalf("nodes missing leading semicolon: %s", job)
	}
	t.Logf("job=%s", job)

	resp, err := uma.Graph("xuma-graph", job, 30)
	if err != nil {
		t.Fatalf("graph: %v", err)
	}
	if !strings.Contains(resp, "NOP") || !strings.Contains(resp, "MARK") {
		t.Fatalf("unexpected wait reply: %s", resp)
	}

	jr, err := uma.FormatGraphEx("", 1, "repeat", "NOP@CPU!", 2, -1, "")
	if err != nil {
		t.Fatalf("format repeat: %v", err)
	}
	if !strings.Contains(jr, "form=repeat") || !strings.Contains(jr, "ngen=2") {
		t.Fatalf("unexpected repeat job: %s", jr)
	}
}
