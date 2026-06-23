package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/version"
)

func TestVersionHandlerPrintsClientVersion(t *testing.T) {
	version.EdgeBuild = "false"
	t.Cleanup(func() { version.EdgeBuild = "false" })

	cmd := &cobra.Command{}
	var buf bytes.Buffer
	cmd.SetOut(&buf)
	cmd.SetErr(&buf)
	versionHandler(cmd, nil)

	out := buf.String()
	if !strings.Contains(out, "zerollama version is") {
		t.Fatalf("output=%q", out)
	}
	if strings.Contains(out, "edge build:") {
		t.Fatalf("unexpected edge marker in default build: %q", out)
	}
}

func TestVersionHandlerEdgeBuildMarker(t *testing.T) {
	version.EdgeBuild = "true"
	t.Cleanup(func() { version.EdgeBuild = "false" })

	cmd := &cobra.Command{}
	var buf bytes.Buffer
	cmd.SetOut(&buf)
	cmd.SetErr(&buf)
	versionHandler(cmd, nil)

	out := buf.String()
	if !strings.Contains(out, "edge build: true") {
		t.Fatalf("output=%q", out)
	}
}
