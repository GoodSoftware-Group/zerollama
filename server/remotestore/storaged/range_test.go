package storaged

import "testing"

func TestParseContentRange(t *testing.T) {
	cr, err := parseContentRange("bytes 10-19/100")
	if err != nil || cr == nil || cr.start != 10 || cr.end != 19 || cr.total != 100 {
		t.Fatalf("got %+v err=%v", cr, err)
	}
	if cr, err := parseContentRange(""); err != nil || cr != nil {
		t.Fatalf("empty: %+v %v", cr, err)
	}
	if _, err := parseContentRange("bytes 5-1/10"); err == nil {
		t.Fatal("expected invalid range")
	}
}
