package envconfig

import (
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"
)

var trainingWindowInvalidWarn sync.Once

// TrainingAllowedWindowConfigured is true when ZEROLLAMA_TRAINING_ALLOWED_WINDOW is non-empty.
func TrainingAllowedWindowConfigured() bool {
	return strings.TrimSpace(Var("ZEROLLAMA_TRAINING_ALLOWED_WINDOW")) != ""
}

// TrainingAllowedWindowMisconfigured is true when the window env is set but not HH:MM-HH:MM.
func TrainingAllowedWindowMisconfigured() bool {
	if !TrainingAllowedWindowConfigured() {
		return false
	}
	_, _, ok, valid := TrainingAllowedWindow()
	return !ok || !valid
}

// TrainingAllowedWindowEnabled reports a successfully parsed allowed window.
func TrainingAllowedWindowEnabled() bool {
	_, _, ok, valid := TrainingAllowedWindow()
	return ok && valid
}

// TrainingAllowedWindow parses ZEROLLAMA_TRAINING_ALLOWED_WINDOW (e.g. "22:00-06:00").
// ok is true when the env is non-empty; valid is true when both clock times parsed.
// Spans midnight when start > end. Empty env returns ok=false.
func TrainingAllowedWindow() (startMin, endMin int, ok bool, valid bool) {
	raw := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_ALLOWED_WINDOW"))
	if raw == "" {
		return 0, 0, false, false
	}
	parts := strings.SplitN(raw, "-", 2)
	if len(parts) != 2 {
		warnInvalidTrainingWindow(raw, "expected HH:MM-HH:MM")
		return 0, 0, true, false
	}
	startMin, err1 := parseClockMinutes(parts[0])
	endMin, err2 := parseClockMinutes(parts[1])
	if err1 != nil || err2 != nil {
		warnInvalidTrainingWindow(raw, "expected HH:MM-HH:MM")
		return 0, 0, true, false
	}
	return startMin, endMin, true, true
}

func warnInvalidTrainingWindow(raw, hint string) {
	trainingWindowInvalidWarn.Do(func() {
		slog.Warn(
			"invalid ZEROLLAMA_TRAINING_ALLOWED_WINDOW; training submit blocked until fixed",
			"value", raw,
			"hint", hint,
		)
	})
}

// TrainingWindowLocation returns the timezone for the allowed window (local by default).
func TrainingWindowLocation() *time.Location {
	tz := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_WINDOW_TZ"))
	if tz == "" || strings.EqualFold(tz, "local") {
		return time.Local
	}
	loc, err := time.LoadLocation(tz)
	if err != nil {
		slog.Warn("invalid ZEROLLAMA_TRAINING_WINDOW_TZ; using local time", "value", tz, "error", err)
		return time.Local
	}
	return loc
}

// WithinTrainingAllowedWindow reports whether t falls inside the configured window.
// Misconfigured window env fails closed (always false). Unset env is always true.
func WithinTrainingAllowedWindow(t time.Time) bool {
	start, end, ok, valid := TrainingAllowedWindow()
	if !ok {
		return true
	}
	if !valid {
		return false
	}
	loc := TrainingWindowLocation()
	t = t.In(loc)
	cur := t.Hour()*60 + t.Minute()
	if start == end {
		return true
	}
	if start < end {
		return cur >= start && cur < end
	}
	// e.g. 22:00-06:00 — allowed from 22:00 through 05:59
	return cur >= start || cur < end
}

// TrainingAllowedWindowLabel returns a human-readable window for errors.
func TrainingAllowedWindowLabel() string {
	raw := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_ALLOWED_WINDOW"))
	if raw == "" {
		return ""
	}
	tz := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_WINDOW_TZ"))
	if tz == "" {
		return raw + " (local)"
	}
	return raw + " (" + tz + ")"
}

func parseClockMinutes(s string) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty time")
	}
	parts := strings.Split(s, ":")
	if len(parts) != 2 {
		return 0, fmt.Errorf("expected HH:MM")
	}
	h, err := strconv.Atoi(parts[0])
	if err != nil || h < 0 || h > 23 {
		return 0, fmt.Errorf("invalid hour")
	}
	m, err := strconv.Atoi(parts[1])
	if err != nil || m < 0 || m > 59 {
		return 0, fmt.Errorf("invalid minute")
	}
	return h*60 + m, nil
}
