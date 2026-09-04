package api

import "testing"

func TestFilterParallelToolCalls(t *testing.T) {
	two := []ToolCall{{ID: "a"}, {ID: "b"}}
	if got := FilterParallelToolCalls(two, nil, nil); len(got) != 2 {
		t.Fatalf("nil parallel keeps both, got %d", len(got))
	}
	on := true
	if got := FilterParallelToolCalls(two, &on, nil); len(got) != 2 {
		t.Fatalf("true keeps both, got %d", len(got))
	}
	off := false
	got := FilterParallelToolCalls(two, &off, nil)
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("false keeps first, got %+v", got)
	}
	already := 1
	if got := FilterParallelToolCalls(two, &off, &already); got != nil {
		t.Fatalf("already sent drops rest, got %+v", got)
	}
}

func TestFilterToolsByName(t *testing.T) {
	tools := Tools{
		{Function: ToolFunction{Name: "a"}},
		{Function: ToolFunction{Name: "b"}},
	}
	got := FilterToolsByName(tools, "b")
	if len(got) != 1 || got[0].Function.Name != "b" {
		t.Fatalf("got %+v", got)
	}
	if FilterToolsByName(tools, "missing") != nil {
		t.Fatal("unknown name")
	}
}
