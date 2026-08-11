package mlxrunner

import (
	"testing"
)

func TestMTPRequire(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_MTP", "")
	if MTPRequire() {
		t.Fatal("empty env must not require MTP")
	}
	t.Setenv("ZEROLLAMA_MLX_MTP", "auto")
	if MTPRequire() {
		t.Fatal("auto must not require MTP")
	}
	t.Setenv("ZEROLLAMA_MLX_MTP", "require")
	if !MTPRequire() {
		t.Fatal("require must fail-closed")
	}
	t.Setenv("ZEROLLAMA_MLX_MTP", "REQUIRE")
	if !MTPRequire() {
		t.Fatal("REQUIRE must fail-closed")
	}
	t.Setenv("ZEROLLAMA_MLX_MTP", "1")
	if !MTPRequire() {
		t.Fatal("1 must fail-closed")
	}
}

func TestResolveMTPHistoryPolicy(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY", "")
	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW", "")
	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD", "")

	p, err := resolveMTPHistoryPolicy("", 1000)
	if err != nil || p.Policy != mtpHistoryCommitted || p.WindowTokens != 0 {
		t.Fatalf("default: %+v err=%v", p, err)
	}

	p, err = resolveMTPHistoryPolicy("auto", 1000)
	if err != nil || p.Policy != mtpHistoryCommitted {
		t.Fatalf("auto short: %+v err=%v", p, err)
	}
	p, err = resolveMTPHistoryPolicy("auto", 20000)
	if err != nil || p.Policy != mtpHistoryLastWindow || p.WindowTokens != 8192 {
		t.Fatalf("auto long: %+v err=%v", p, err)
	}
	// history_token_ids = prompt[1:]; keep 8192 of 19999 → dropped 11807 → HistoryStart 11808
	if p.HistoryStart != 20000-1-8192+1 {
		t.Fatalf("HistoryStart = %d want %d", p.HistoryStart, 20000-1-8192+1)
	}

	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY", "last_window")
	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW", "4096")
	p, err = resolveMTPHistoryPolicy("committed", 10000)
	if err != nil || p.Policy != mtpHistoryLastWindow || p.WindowTokens != 4096 {
		t.Fatalf("env override of committed: %+v err=%v", p, err)
	}

	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY", "")
	p, err = resolveMTPHistoryPolicy("cycle", 100)
	if err != nil || p.Policy != mtpHistoryCommitted {
		t.Fatalf("cycle→committed: %+v err=%v", p, err)
	}

	if _, err := resolveMTPHistoryPolicy("bogus", 1); err == nil {
		t.Fatal("want error for bogus policy")
	}
}

func TestMTPHistoryLastWindowWriteCursor(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY", "last_window")
	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW", "4")
	t.Setenv("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD", "")

	plan, err := resolveMTPHistoryPolicy("last_window", 10)
	if err != nil {
		t.Fatal(err)
	}
	// prompt 10 → history len 9 → keep 4 → dropped 5 → HistoryStart 6
	if plan.HistoryStart != 6 {
		t.Fatalf("HistoryStart=%d want 6", plan.HistoryStart)
	}
	s := &mtpDraftSession{history: plan, writeLookAheadAbs: plan.HistoryStart}
	// Simulate skip: first run ends before HistoryStart
	if start := s.writeLookAheadAbs - 0; start != 6 {
		t.Fatalf("start at pos 0 = %d want 6", start)
	}
	// After jumping, a run at position 4 still waits until abs 6
	if start := s.writeLookAheadAbs - 4; start != 2 {
		t.Fatalf("start at pos 4 = %d want 2", start)
	}
}
