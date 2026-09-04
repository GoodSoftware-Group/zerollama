package mlxrunner

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/sample"
)

func TestKnobSnapshotPLDOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "off")
	t.Setenv("ZEROLLAMA_MLX_MTP", "")
	got := KnobSnapshot()
	if got[0].Value != "off" || !got[0].Overridden {
		t.Fatalf("PLD %+v", got[0])
	}
	text := FormatKnobs(got)
	if !strings.Contains(text, "ZEROLLAMA_MLX_PLD=off") || !strings.Contains(text, "Tune:") {
		t.Fatalf("format %q", text)
	}
}

func TestKnobSnapshotDefaults(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "")
	t.Setenv("ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN", "")
	t.Setenv("ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT", "")
	t.Setenv("ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT", "")
	got := KnobSnapshot()
	if got[0].Value != "on" || got[0].Overridden {
		t.Fatalf("default PLD %+v", got[0])
	}
	text := FormatKnobs(got)
	for _, want := range []string{
		"ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN=on",
		"ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT=on",
		"ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT=12288",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q in %q", want, text)
		}
	}
}

func TestRoundCostTuneReportEmpty(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	detail, warn := RoundCostTuneReport()
	if warn {
		t.Fatal(detail)
	}
	if detail == "" {
		t.Fatal("expected a note")
	}
}

func TestLastRunTuneReport(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	path := lastRunPath("qwen2.5:7b")
	if path == "" {
		t.Fatal("path")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	f := LastRun{
		Model:         "qwen2.5:7b",
		At:            time.Now().UTC(),
		Drafted:       10,
		Accepted:      1,
		Enabled:       false,
		PLD:           true,
		Hint:          "parked",
		Acceptance:    0.1,
		GreedyCoupled: true,
	}
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	detail, warn := LastRunTuneReport()
	if warn {
		t.Fatalf("single parked request should not warn: %s", detail)
	}
	if !strings.Contains(detail, "qwen2.5:7b") || !strings.Contains(detail, "parked=1/1") {
		t.Fatalf("detail %q", detail)
	}
	if !strings.Contains(detail, "greedy_coupled") {
		t.Fatalf("missing greedy_coupled in %q", detail)
	}
}

func TestLastRunTuneReportMajorityParkedWarns(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	path := lastRunPath("qwen2.5:7b")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	var runs []LastRun
	for i := 0; i < 3; i++ {
		runs = append(runs, LastRun{Model: "qwen2.5:7b", At: time.Now().UTC(), Enabled: false, PLD: true, Drafted: 10, Accepted: 1})
	}
	data, err := json.Marshal(lastRunFile{Runs: runs})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	detail, warn := LastRunTuneReport()
	if !warn {
		t.Fatal(detail)
	}
}

func TestRoundCostTuneReportSkipsLastJSON(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	path := lastRunPath("only-last")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"model":"only-last"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	detail, warn := RoundCostTuneReport()
	if warn {
		t.Fatal(detail)
	}
}

func TestTuneHintParked(t *testing.T) {
	s := &speculationSession{enabled: false, pld: true, stats: specStats{iterations: 8}}
	if got := s.tuneHint(0.1); got == "" || !strings.Contains(got, "ZEROLLAMA_MLX_PLD") {
		t.Fatalf("hint %q", got)
	}
}

func TestTuneHintGreedyTrioFence(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT", "")
	s := &speculationSession{enabled: false, promptTokens: 20000, stats: specStats{iterations: 8}}
	got := s.tuneHint(0.1)
	if !strings.Contains(got, "GREEDY_TRIO_MAX_CONTEXT") {
		t.Fatalf("long-prompt park should mention greedy fence, got %q", got)
	}
	s.promptTokens = 100
	got = s.tuneHint(0.1)
	if strings.Contains(got, "GREEDY_TRIO") {
		t.Fatalf("short prompt must not mention fence, got %q", got)
	}
}

func TestGreedyCoupled(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN", "")
	t.Setenv("ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT", "")
	t.Setenv("ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT", "")
	if (&speculationSession{promptTokens: 100}).greedyCoupled() {
		t.Fatal("nil runner is not coupled")
	}
	samp := sample.New(128)
	samp.Add(pipelineSlot, sample.Options{}, nil)
	s := &speculationSession{promptTokens: 100, spec: &speculation{r: &Runner{Sampler: samp}}}
	if !s.greedyCoupled() {
		t.Fatal("T=0 below fence must couple")
	}
	s.promptTokens = 20000
	if s.greedyCoupled() {
		t.Fatal("fence must uncouple")
	}
	s.promptTokens = 100
	samp.Remove(pipelineSlot)
	samp.Add(pipelineSlot, sample.Options{Temperature: 0.8}, nil)
	if s.greedyCoupled() {
		t.Fatal("sampled request must not couple")
	}
}
