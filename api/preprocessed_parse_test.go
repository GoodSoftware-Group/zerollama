package api

import (
	"encoding/json"
	"testing"
)

func TestFlattenJSONFloats_2d(t *testing.T) {
	raw := json.RawMessage(`[[1,2],[3,4]]`)
	got, err := FlattenJSONFloats(raw)
	if err != nil || len(got) != 4 || got[2] != 3 {
		t.Fatalf("got=%v err=%v", got, err)
	}
}

func TestParseGridTHW_batch(t *testing.T) {
	got, err := ParseGridTHW(json.RawMessage(`[[1,24,32]]`), nil)
	if err != nil || len(got) != 3 || got[1] != 24 {
		t.Fatalf("got=%v err=%v", got, err)
	}
}
