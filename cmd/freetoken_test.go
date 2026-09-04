package cmd

import (
	"testing"
)

func TestNewFreetokenCommand(t *testing.T) {
	c := NewFreetokenCommand()
	if c == nil || c.Use != "freetoken" || !c.Hidden {
		t.Fatalf("command = %+v", c)
	}
	if c.Flags().Lookup("json") == nil {
		t.Fatal("missing --json")
	}
}

func TestBuildFreetokenReportProfile(t *testing.T) {
	rep, err := buildFreetokenReport()
	if err != nil {
		t.Fatal(err)
	}
	if rep.Profile == "" || rep.DoctorLine == "" {
		t.Fatalf("%+v", rep)
	}
	if len(rep.Chat) < 3 {
		t.Fatalf("expected chat lab rows, got %d", len(rep.Chat))
	}
	for _, b := range rep.Blobs {
		if b.Apply {
			t.Fatal("apply must stay false")
		}
	}
}
