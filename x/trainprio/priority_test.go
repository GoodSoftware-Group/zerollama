package trainprio

import "testing"

func TestParse(t *testing.T) {
	cases := []struct {
		in      string
		want    Priority
		wantErr bool
	}{
		{"", Normal, false},
		{"normal", Normal, false},
		{"low", Low, false},
		{"batch", Low, false},
		{"overnight", Low, false},
		{"high", High, false},
		{"interactive", High, false},
		{"realtime", High, false},
		{"higgh", Normal, true},
		{"turbo", Normal, true},
	}
	for _, tc := range cases {
		got, err := Parse(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Fatalf("%q: expected error", tc.in)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%q: unexpected err %v", tc.in, err)
		}
		if got != tc.want {
			t.Fatalf("%q: got %v want %v", tc.in, got, tc.want)
		}
	}
}

func TestResolve(t *testing.T) {
	p, err := Resolve("", "overnight")
	if err != nil || p != Low {
		t.Fatalf("slo overnight: got %v %v", p, err)
	}
	p, err = Resolve("high", "interactive")
	if err != nil || p != High {
		t.Fatalf("agree: got %v %v", p, err)
	}
	_, err = Resolve("high", "low")
	if err == nil {
		t.Fatal("expected conflict")
	}
	_, err = Resolve("nope", "")
	if err == nil {
		t.Fatal("expected unknown priority")
	}
}
