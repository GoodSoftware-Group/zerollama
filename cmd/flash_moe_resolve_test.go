package cmd

import (
	"testing"
)

func TestNewFlashMoEResolveCommand(t *testing.T) {
	c := NewFlashMoEResolveCommand()
	if c == nil || c.Use != "flash-moe-resolve" {
		t.Fatalf("command = %+v", c)
	}
}
