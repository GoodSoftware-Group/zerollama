package cmd

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestDoctorCheckMLXKnobsDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "")
	c := doctorCheckMLXKnobs()
	if c.Status != "ok" {
		t.Fatalf("%s %s", c.Status, c.Detail)
	}
	if !strings.Contains(c.Detail, "ZEROLLAMA_MLX_PLD=on") {
		t.Fatalf("detail %q", c.Detail)
	}
	if !strings.Contains(c.Detail, "ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN=on") {
		t.Fatalf("missing greedy trio knobs in %q", c.Detail)
	}
}

func TestDoctorCheckMLXKnobsPLDOffWarns(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_PLD", "off")
	c := doctorCheckMLXKnobs()
	if c.Status != "warn" {
		t.Fatalf("status=%s", c.Status)
	}
	if c.FixHint == "" {
		t.Fatal("expected fix hint")
	}
}

func TestDoctorCheckMLXLastRunEmptyOK(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	c := doctorCheckMLXLastRun()
	if c.Status == "fail" {
		t.Fatalf("%s", c.Detail)
	}
}

func TestDoctorCheckMLXRoundCostEmptyOK(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	c := doctorCheckMLXRoundCost()
	if c.Status == "fail" {
		t.Fatalf("%s", c.Detail)
	}
}
