package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/mattn/go-runewidth"
)

func TestUseCompactList(t *testing.T) {
	t.Parallel()
	if useCompactList(0) {
		t.Fatal("unknown width should stay wide (pipes/tests)")
	}
	if useCompactList(120) {
		t.Fatal("wide terminal should stay single-line")
	}
	if !useCompactList(80) {
		t.Fatal("80-col terminal should use compact 2-line")
	}
}

func TestPrintListCompactFits80(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	printListCompact(&buf, []listTableRow{{
		name:     "qwen3-coder-next:6bit",
		id:       "ffc5c8db17e8",
		size:     "64 GB",
		params:   "15.0B MoE 512x10",
		ctx:      "16k–80k",
		perf:     "--",
		modified: "4 minutes ago",
	}}, 80)

	for i, line := range strings.Split(strings.TrimRight(buf.String(), "\n"), "\n") {
		if w := runewidth.StringWidth(line); w > 80 {
			t.Fatalf("line %d width %d > 80: %q", i, w, line)
		}
	}
	out := buf.String()
	if !strings.Contains(out, "PARAMS") || !strings.Contains(out, "15.0B MoE 512x10") {
		t.Fatalf("missing params on line 2:\n%s", out)
	}
	if !strings.Contains(out, "16k–80k") {
		t.Fatalf("missing ctx:\n%s", out)
	}
}

func TestPrintPsCompactWithProjects(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	printPsCompact(&buf, []psTableRow{{
		name:      "ornith-35b-optiq:latest",
		project:   "hermes-lean/discord:dm:1516015052568793098",
		session:   "hermes:agent:main:discord:dm:1516015052568793098",
		id:        "f4df829f8a75",
		size:      "27 GB",
		processor: "100% GPU",
		context:   "262144",
		until:     "29 minutes from now",
	}}, true, 80)

	for i, line := range strings.Split(strings.TrimRight(buf.String(), "\n"), "\n") {
		if w := runewidth.StringWidth(line); w > 80 {
			t.Fatalf("line %d width %d > 80: %q", i, w, line)
		}
	}
}
