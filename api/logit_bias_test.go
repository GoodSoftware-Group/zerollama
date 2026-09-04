package api

import "testing"

func TestParseLogitBias(t *testing.T) {
	got, err := ParseLogitBias(map[string]float64{"13": -100, "42": 2.5})
	if err != nil {
		t.Fatal(err)
	}
	if got[13] != -100 || got[42] != 2.5 {
		t.Fatalf("%v", got)
	}
	if _, err := ParseLogitBias(map[string]float64{"-1": 1}); err == nil {
		t.Fatal("negative id")
	}
	if _, err := ParseLogitBias(map[string]any{"nope": 1}); err == nil {
		t.Fatal("non-int key")
	}
}
