package tools

import (
	"strings"
	"testing"
	"text/template"
)

func TestTemplateToolTag_defaultsToBrace(t *testing.T) {
	if got := TemplateToolTag(nil); got != "{" {
		t.Fatalf("got %q", got)
	}
}

func TestTemplateToolTag_fromTemplate(t *testing.T) {
	tmpl, err := template.New("t").Parse(
		`{{if .ToolCalls}}<tool_call>{{end}}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	got := TemplateToolTag(tmpl)
	if !strings.HasPrefix(got, "<tool_call>") {
		t.Fatalf("got %q", got)
	}
}
