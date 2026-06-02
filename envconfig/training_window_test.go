package envconfig

import (
	"testing"
	"time"
)

func TestTrainingAllowedWindowParse(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_ALLOWED_WINDOW", "22:00-06:00")
	start, end, ok, valid := TrainingAllowedWindow()
	if !ok || !valid {
		t.Fatal("expected enabled valid window")
	}
	if start != 22*60 || end != 6*60 {
		t.Fatalf("got %d-%d", start, end)
	}
	if !TrainingAllowedWindowEnabled() {
		t.Fatal("expected enabled")
	}
}

func TestTrainingAllowedWindowInvalidFailsClosed(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_ALLOWED_WINDOW", "not-a-window")
	if !TrainingAllowedWindowMisconfigured() {
		t.Fatal("expected misconfigured")
	}
	if TrainingAllowedWindowEnabled() {
		t.Fatal("expected not enabled when invalid")
	}
	if WithinTrainingAllowedWindow(time.Now()) {
		t.Fatal("invalid window must fail closed")
	}
}

func TestWithinTrainingAllowedWindowMidnightSpan(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_ALLOWED_WINDOW", "22:00-06:00")
	t.Setenv("ZEROLLAMA_TRAINING_WINDOW_TZ", "UTC")
	loc := time.UTC
	cases := []struct {
		at   time.Time
		want bool
	}{
		{time.Date(2026, 5, 17, 23, 0, 0, 0, loc), true},
		{time.Date(2026, 5, 17, 5, 30, 0, 0, loc), true},
		{time.Date(2026, 5, 17, 12, 0, 0, 0, loc), false},
		{time.Date(2026, 5, 17, 6, 0, 0, 0, loc), false},
	}
	for _, tc := range cases {
		if got := WithinTrainingAllowedWindow(tc.at); got != tc.want {
			t.Fatalf("%s: got %v want %v", tc.at.Format(time.RFC3339), got, tc.want)
		}
	}
}

func TestWithinTrainingAllowedWindowSameDay(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_ALLOWED_WINDOW", "09:00-17:00")
	t.Setenv("ZEROLLAMA_TRAINING_WINDOW_TZ", "UTC")
	loc := time.UTC
	if !WithinTrainingAllowedWindow(time.Date(2026, 5, 17, 10, 0, 0, 0, loc)) {
		t.Fatal("expected inside")
	}
	if WithinTrainingAllowedWindow(time.Date(2026, 5, 17, 18, 0, 0, 0, loc)) {
		t.Fatal("expected outside")
	}
}
