package agentstats_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/agentstats"
)

func TestInitTruncatesAndRecords(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "gemma-agent.jsonl")

	if err := agentstats.Init(logPath); err != nil {
		t.Fatal(err)
	}
	agentstats.Record("request_in", map[string]any{
		"model":             "gemma4:26b-optiq",
		"prompt_cache_key":  "hermes:main",
		"prompt_tokens":     6587,
	})
	agentstats.Record("ignored", map[string]any{"model": "llama3.2:3b"})

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines=%d want 2 (serve_start + one record)", len(lines))
	}
	var start map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &start); err != nil {
		t.Fatal(err)
	}
	if start["event"] != "serve_start" {
		t.Fatalf("first event=%v", start["event"])
	}
	if start["version"] == "" {
		t.Fatal("serve_start missing version")
	}
	if _, ok := start["pid"]; !ok {
		t.Fatal("serve_start missing pid")
	}

	var row map[string]any
	if err := json.Unmarshal([]byte(lines[1]), &row); err != nil {
		t.Fatal(err)
	}
	if row["event"] != "request_in" {
		t.Fatalf("event=%v", row["event"])
	}

	if err := agentstats.Init(logPath); err != nil {
		t.Fatal(err)
	}
	agentstats.Record("request_in", map[string]any{"model": "gemma4:26b-optiq"})
	raw2, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines2 := strings.Split(strings.TrimSpace(string(raw2)), "\n")
	if len(lines2) != 2 {
		t.Fatalf("after re-init lines=%d want 2 (truncated)", len(lines2))
	}
	prevPath := logPath + ".prev"
	if _, err := os.Stat(prevPath); err != nil {
		t.Fatalf("expected rotated previous log at %s: %v", prevPath, err)
	}
	var restart map[string]any
	if err := json.Unmarshal([]byte(lines2[0]), &restart); err != nil {
		t.Fatal(err)
	}
	if restart["previous_log"] != prevPath {
		t.Fatalf("previous_log=%v want %s", restart["previous_log"], prevPath)
	}
}

func TestMaybeRecordRunnerLine(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "gemma-agent.jsonl")
	if err := agentstats.Init(logPath); err != nil {
		t.Fatal(err)
	}
	line := `time=2026-07-03T11:00:00.000-07:00 level=INFO source=cache.go:160 msg="cache hit" total=6603 matched=6583 cached=6583 left=20 utilization_pct=99.7`
	agentstats.MaybeRecordRunnerLine(line)

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "mlx_cache") {
		t.Fatalf("missing mlx_cache in %q", string(raw))
	}
}
