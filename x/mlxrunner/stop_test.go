package mlxrunner

import "testing"

func TestFlushStopHold(t *testing.T) {
	stops := []string{"</s>", "END"}
	emit, rem, matched := flushStopHold("hello", stops, false)
	if emit != "hello" || rem != "" || matched != "" {
		t.Fatalf("plain: emit=%q rem=%q matched=%q", emit, rem, matched)
	}
	emit, rem, matched = flushStopHold("hello<", stops, false)
	if emit != "" || rem != "hello<" || matched != "" {
		t.Fatalf("prefix: emit=%q rem=%q matched=%q", emit, rem, matched)
	}
	emit, rem, matched = flushStopHold("hello</s> extra", stops, false)
	if emit != "hello" || rem != "" || matched != "</s>" {
		t.Fatalf("hit: emit=%q rem=%q matched=%q", emit, rem, matched)
	}
	emit, rem, matched = flushStopHold("hello<", stops, true)
	if emit != "hello<" || rem != "" || matched != "" {
		t.Fatalf("force: emit=%q rem=%q matched=%q", emit, rem, matched)
	}
}

func TestNonemptyStops(t *testing.T) {
	got := nonemptyStops([]string{"", "</s>", ""})
	if len(got) != 1 || got[0] != "</s>" {
		t.Fatalf("got %v", got)
	}
}
