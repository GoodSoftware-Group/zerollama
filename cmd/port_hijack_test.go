package cmd

import (
	"errors"
	"net/url"
	"strings"
	"testing"
)

func TestParseLsofListenF(t *testing.T) {
	raw := "" +
		"p1672\n" +
		"czerollama\n" +
		"n*:11434\n" +
		"p45084\n" +
		"cCursor\n" +
		"n127.0.0.1:11434\n" +
		"n[::1]:11434\n"
	rows := parseLsofListenF(raw)
	if len(rows) != 3 {
		t.Fatalf("rows=%d %+v", len(rows), rows)
	}
	if rows[0].PID != "1672" || rows[0].Command != "zerollama" || rows[0].Name != "*:11434" {
		t.Fatalf("wildcard: %+v", rows[0])
	}
	if rows[1].Name != "127.0.0.1:11434" || rows[2].Name != "[::1]:11434" {
		t.Fatalf("loopback names: %+v", rows)
	}
	if n := uniqueListenPIDs(rows); len(n) != 2 {
		t.Fatalf("pids=%v", n)
	}
}

func TestFormatPortHijackSplitBind(t *testing.T) {
	u, _ := url.Parse("http://127.0.0.1:11434")
	rows := []portListener{
		{PID: "1672", Command: "zerollama", Name: "*:11434"},
		{PID: "45084", Command: "Cursor", Name: "127.0.0.1:11434"},
	}
	hb := errors.New(`Head "http://127.0.0.1:11434/": EOF`)
	if !emptyHTTPReply(hb) {
		t.Fatal("expected EOF heartbeat to count as empty reply")
	}
	got := formatPortHijack(u, hb, rows, "192.168.10.16:11434")
	for _, need := range []string{"split bind", "Cursor", "OLLAMA_HOST=192.168.10.16:11434", "EOF"} {
		if !strings.Contains(got, need) {
			t.Fatalf("missing %q in:\n%s", need, got)
		}
	}
}

func TestFormatPortHijackImpostor(t *testing.T) {
	u, _ := url.Parse("http://127.0.0.1:11434")
	rows := []portListener{{PID: "1", Command: "Cursor", Name: "127.0.0.1:11434"}}
	got := formatPortHijack(u, errors.New(`Head "http://127.0.0.1:11434/": EOF`), rows, "")
	if !strings.Contains(got, "not zerollama") {
		t.Fatalf("got %q", got)
	}
}

func TestFormatPortHijackHealthySingle(t *testing.T) {
	u, _ := url.Parse("http://127.0.0.1:11434")
	rows := []portListener{{PID: "1672", Command: "zerollama", Name: "*:11434"}}
	if got := formatPortHijack(u, nil, rows, ""); got != "" {
		t.Fatalf("unexpected hint: %s", got)
	}
}

func TestLooksLikeZerollamaListenCmd(t *testing.T) {
	if !looksLikeZerollamaListenCmd("zerollama") || !looksLikeZerollamaListenCmd("zerollama-serve") {
		t.Fatal("zerollama cmds")
	}
	if looksLikeZerollamaListenCmd("Cursor") {
		t.Fatal("Cursor must not match")
	}
}
