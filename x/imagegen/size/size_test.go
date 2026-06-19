package size

import "testing"

func TestResolveAspectPresets(t *testing.T) {
	const max = int32(384)
	cases := []struct {
		aspect       string
		wantW, wantH int32
	}{
		{"16:9", 384, 216},
		{"9:16", 216, 384},
		{"3:2", 384, 256},
		{"2:3", 256, 384},
		{"1:1", 384, 384},
	}
	for _, tc := range cases {
		w, h, err := Resolve(0, 0, tc.aspect, max)
		if err != nil {
			t.Fatalf("%s: %v", tc.aspect, err)
		}
		if w != tc.wantW || h != tc.wantH {
			t.Fatalf("%s: got %dx%d want %dx%d", tc.aspect, w, h, tc.wantW, tc.wantH)
		}
	}
}

func TestResolveExplicitWidth(t *testing.T) {
	w, h, err := Resolve(512, 0, "16:9", 384)
	if err != nil {
		t.Fatal(err)
	}
	if w != 384 || h != 216 {
		t.Fatalf("got %dx%d want 384x216 (clamped long edge)", w, h)
	}
}

func TestResolveBadAspect(t *testing.T) {
	_, _, err := Resolve(0, 0, "21:9", 384)
	if err == nil {
		t.Fatal("expected error")
	}
}
