//go:build darwin && uma

package uma

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// Optiq GRAPH token-tail on live MLX last-hidden (F0700).
// Modes via ZEROLLAMA_UMA_OPTIQ_TOKEN_TAIL (alias: ZEROLLAMA_UMA_OPTIQ_GRAPH_TOKEN):
//
//	"" / off  — disabled
//	1 / on    — replace Unembed+argmax with GRAPH; soft-fallback to MLX on failure
//	require   — same, fail-closed
//	owned     — require + count owned consumptions (F0685-style)
//
// Dump: make -C …/uma_toolkit optiq-token-tail-dump → UMA_OPTIQ_TOKEN_TAIL_DIR
// (default /tmp/uma_optiq_token_tail_dump): fnorm.bin, lm_q8.bin, meta.txt.
//
// Recipe (F0666/F0687): NORM_MUL_F16 → GEMV_Q8_G64 → ARGMAX.
// GEMV reads x after NORM (same x= name; in-place host semantics).
// Callers must feed pre-final-norm last-hidden (mlxrunner skips Norm when active).
var (
	optiqTokMu    sync.Mutex
	optiqTokReady bool
	optiqTokD     int
	optiqTokV     int
	optiqTokNw    int
	optiqTokSteps int
	optiqTokOwned int
)

func optiqTokenMode() string {
	m := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_TAIL")))
	if m == "" {
		m = strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_TOKEN")))
	}
	return m
}

// OptiqTokenTailEnabled reports whether in-process GRAPH token-tail is on.
func OptiqTokenTailEnabled() bool {
	switch optiqTokenMode() {
	case "1", "on", "true", "yes", "require", "owned":
		return true
	default:
		return false
	}
}

// OptiqGraphTokenEnabled is an alias for OptiqTokenTailEnabled.
func OptiqGraphTokenEnabled() bool { return OptiqTokenTailEnabled() }

// OptiqTokenTailRequire reports fail-closed mode.
func OptiqTokenTailRequire() bool {
	m := optiqTokenMode()
	return m == "require" || m == "owned"
}

// OptiqGraphTokenOwned reports owned mode.
func OptiqTokenTailOwned() bool {
	return optiqTokenMode() == "owned"
}

// OptiqGraphTokenOwned is an alias for OptiqTokenTailOwned.
func OptiqGraphTokenOwned() bool { return OptiqTokenTailOwned() }

func optiqTokenTailDir() string {
	d := os.Getenv("UMA_OPTIQ_TOKEN_TAIL_DIR")
	if d == "" {
		d = "/tmp/uma_optiq_token_tail_dump"
	}
	return d
}

func readTokenTailMeta(path string) (map[string]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m := map[string]string{}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		m[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	return m, nil
}

// EnsureOptiqTokenTailSession BUF_PUTs fnorm + lm_q8 once (session-resident).
func EnsureOptiqTokenTailSession() error {
	optiqTokMu.Lock()
	defer optiqTokMu.Unlock()
	if optiqTokReady {
		return nil
	}
	if !Active() {
		return fmt.Errorf("uma not active")
	}
	dir := optiqTokenTailDir()
	meta, err := readTokenTailMeta(filepath.Join(dir, "meta.txt"))
	if err != nil {
		return fmt.Errorf("token-tail dump missing at %s (make optiq-token-tail-dump): %w", dir, err)
	}
	D, _ := strconv.Atoi(meta["D"])
	V, _ := strconv.Atoi(meta["V"])
	nw, _ := strconv.Atoi(meta["nw_lm"])
	if D < 1 || V < 1 {
		return fmt.Errorf("bad token-tail meta D=%d V=%d", D, V)
	}
	fnorm, err := os.ReadFile(filepath.Join(dir, "fnorm.bin"))
	if err != nil {
		return err
	}
	lm, err := os.ReadFile(filepath.Join(dir, "lm_q8.bin"))
	if err != nil {
		return err
	}
	if nw > 0 && len(lm) != nw {
		return fmt.Errorf("lm_q8 size %d want nw_lm=%d", len(lm), nw)
	}
	if len(fnorm) != D*4 {
		return fmt.Errorf("fnorm size %d want %d", len(fnorm), D*4)
	}
	if nw == 0 {
		nw = len(lm)
	}

	names := []string{"tt_x", "tt_h", "tt_wn", "tt_w", "tt_y", "tt_tok"}
	for _, n := range names {
		BufFree(n)
	}
	nx, nh, nwn, ny, nt := D*4, D*2, D*4, V*4, 4
	for _, p := range []struct {
		n string
		z int
	}{
		{"tt_x", nx}, {"tt_h", nh}, {"tt_wn", nwn}, {"tt_w", nw}, {"tt_y", ny}, {"tt_tok", nt},
	} {
		if err := BufAlloc(p.n, p.z); err != nil {
			return fmt.Errorf("alloc %s: %w", p.n, err)
		}
	}
	zerosY := make([]byte, ny)
	zerosH := make([]byte, nh)
	if err := BufPut("tt_wn", fnorm); err != nil {
		return err
	}
	if err := BufPut("tt_w", lm); err != nil {
		return fmt.Errorf("put lm_q8 (%.0f MiB): %w", float64(nw)/(1024*1024), err)
	}
	if err := BufPut("tt_y", zerosY); err != nil {
		return err
	}
	if err := BufPut("tt_h", zerosH); err != nil {
		return err
	}
	optiqTokD = D
	optiqTokV = V
	optiqTokNw = nw
	optiqTokReady = true
	return nil
}

// OptiqTokenTailSessionReady reports whether weights are resident.
func OptiqTokenTailSessionReady() bool {
	optiqTokMu.Lock()
	defer optiqTokMu.Unlock()
	return optiqTokReady
}

// OptiqTokenTailArgmax runs F0666 GRAPH token-tail on pre-norm last-hidden x.
// Lab override: ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE=gemv_argmax skips NORM (post-norm x).
func OptiqTokenTailArgmax(x []float32) (int32, error) {
	if err := EnsureOptiqTokenTailSession(); err != nil {
		return -1, err
	}
	optiqTokMu.Lock()
	D, V := optiqTokD, optiqTokV
	optiqTokMu.Unlock()
	if len(x) < D {
		return -1, fmt.Errorf("token-tail: x len=%d want >=D=%d", len(x), D)
	}
	if len(x) > D {
		x = x[len(x)-D:]
	}
	xb := F32Bytes(x)
	if err := BufPut("tt_x", xb); err != nil {
		return -1, err
	}
	zerosH := make([]byte, D*2)
	_ = BufPut("tt_h", zerosH)
	tok0 := int32(-1)
	tb := make([]byte, 4)
	binary.LittleEndian.PutUint32(tb, uint32(tok0))
	if err := BufPut("tt_tok", tb); err != nil {
		return -1, err
	}
	zerosY := make([]byte, V*4)
	_ = BufPut("tt_y", zerosY)

	recipe := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE")))
	var nodes string
	switch recipe {
	case "gemv_argmax", "postnorm", "post":
		// Lab: post-norm activation dump → GEMV→ARGMAX only.
		nodes = strings.Join([]string{
			fmt.Sprintf("GEMV_Q8_G64@GPU! x=tt_x y=tt_y w=tt_w N=%d K=%d", V, D),
			fmt.Sprintf("ARGMAX@CPU! x=tt_y out=tt_tok N=1 V=%d", V),
			"MARK@GPU?",
		}, " ; ")
	default:
		// F0666/F0687: GEMV reads x after NORM (same x= name; do not "fix").
		nodes = strings.Join([]string{
			fmt.Sprintf("NORM_MUL_F16@GPU! x=tt_x y=tt_h w=tt_wn D=%d", D),
			fmt.Sprintf("GEMV_Q8_G64@GPU! x=tt_x y=tt_y w=tt_w N=%d K=%d", V, D),
			fmt.Sprintf("ARGMAX@CPU! x=tt_y out=tt_tok N=1 V=%d", V),
			"MARK@GPU?",
		}, " ; ")
	}
	job, err := FormatGraph(1, "chain", nodes)
	if err != nil {
		return -1, err
	}
	resp, err := Graph("mlxrunner-optiq-token-tail", job, 600)
	if err != nil {
		return -1, err
	}
	needNorm := recipe == "" || recipe == "norm_gemv_argmax" || recipe == "norm"
	if needNorm && !strings.Contains(resp, "NORM_MUL_F16") {
		return -1, fmt.Errorf("token-tail unexpected reply (want NORM): %s", truncate(resp, 200))
	}
	if !strings.Contains(resp, "GEMV_Q8_G64") || !strings.Contains(resp, "ARGMAX") {
		return -1, fmt.Errorf("token-tail unexpected reply: %s", truncate(resp, 200))
	}
	gotb := make([]byte, 4)
	n, err := BufGet("tt_tok", gotb)
	if err != nil {
		return -1, err
	}
	if n != 4 {
		return -1, fmt.Errorf("token-tail tok nbytes=%d", n)
	}
	tok := int32(binary.LittleEndian.Uint32(gotb))
	optiqTokMu.Lock()
	optiqTokSteps++
	if OptiqTokenTailOwned() {
		optiqTokOwned++
	}
	optiqTokMu.Unlock()
	return tok, nil
}

// OwnedTokenTailArgmax is an alias for OptiqTokenTailArgmax.
func OwnedTokenTailArgmax(x []float32) (int32, error) {
	return OptiqTokenTailArgmax(x)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// OptiqTokenTailSteps returns how many GRAPH token-tail steps ran.
func OptiqTokenTailSteps() int {
	optiqTokMu.Lock()
	defer optiqTokMu.Unlock()
	return optiqTokSteps
}

// OptiqTokenTailOwnedCount returns owned-mode consumptions.
func OptiqTokenTailOwnedCount() int {
	optiqTokMu.Lock()
	defer optiqTokMu.Unlock()
	return optiqTokOwned
}

// ResetOptiqTokenTailForTest clears session state (smokes only).
func ResetOptiqTokenTailForTest() {
	optiqTokMu.Lock()
	defer optiqTokMu.Unlock()
	for _, n := range []string{"tt_x", "tt_h", "tt_wn", "tt_w", "tt_y", "tt_tok"} {
		BufFree(n)
	}
	optiqTokReady = false
	optiqTokSteps = 0
	optiqTokOwned = 0
}
