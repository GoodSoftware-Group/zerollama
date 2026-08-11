//go:build darwin && uma

package uma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// Optiq decode always-on (F0685). Modes via ZEROLLAMA_UMA_OPTIQ_GRAPH_DECODE:
//
//	"" / off  — disabled
//	1 / on    — every decode gap: session-resident Wz→Wo GRAPH rematch (shadow)
//	require   — same, abort caller on failure
//	owned     — require + broker y is consumed by TakeOwnedY (fail closed if unused)
var (
	optiqDecodeMu      sync.Mutex
	optiqDecodeReady   bool
	optiqDecodeMeta    map[string]string
	optiqDecodeN       int
	optiqDecodeK       int
	optiqDecodeSteps   int
	optiqOwnedY        []float32
	optiqOwnedPending  bool
	optiqOwnedConsumed int
	optiqOwnedFwdErr   error
	optiqOwnedTarget   any // *nn.QuantizedLinear matched by pointer in mlxrunner
)

func optiqDecodeMode() string {
	return strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_DECODE")))
}

// OptiqGraphDecodeEnabled reports whether always-on decode GRAPH is on.
func OptiqGraphDecodeEnabled() bool {
	switch optiqDecodeMode() {
	case "1", "on", "true", "yes", "require", "owned":
		return true
	default:
		return false
	}
}

func optiqDecodeRequire() bool {
	m := optiqDecodeMode()
	return m == "require" || m == "owned"
}

func optiqDecodeOwned() bool {
	return optiqDecodeMode() == "owned"
}

// EnsureOptiqDecodeSession BUF_PUTs live dump packs once (session-resident).
func EnsureOptiqDecodeSession() error {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	if optiqDecodeReady {
		return nil
	}
	if !Active() {
		return fmt.Errorf("uma not active")
	}
	dump := os.Getenv("UMA_OPTIQ_DUMP_DIR")
	if dump == "" {
		dump = "/tmp/uma_optiq_live_dump"
	}
	meta, err := readOptiqMeta(filepath.Join(dump, "meta.txt"))
	if err != nil {
		return fmt.Errorf("optiq decode dump missing at %s (make optiq-live-dump): %w", dump, err)
	}
	N, _ := strconv.Atoi(meta["N"])
	K, _ := strconv.Atoi(meta["K"])
	nw, _ := strconv.Atoi(meta["nw"])
	if N < 1 || K < 1 || nw < 1 {
		return fmt.Errorf("bad optiq meta N=%d K=%d nw=%d", N, K, nw)
	}
	x, err := os.ReadFile(filepath.Join(dump, "x.bin"))
	if err != nil {
		return err
	}
	wz, err := os.ReadFile(filepath.Join(dump, "wz.bin"))
	if err != nil {
		return err
	}
	wo, err := os.ReadFile(filepath.Join(dump, "wo.bin"))
	if err != nil {
		return err
	}
	yHostB, err := os.ReadFile(filepath.Join(dump, "y_host.bin"))
	if err != nil {
		return err
	}
	if len(x) != K*4 || len(wz) != nw || len(wo) != nw || len(yHostB) != N*4 {
		return fmt.Errorf("optiq dump size mismatch")
	}

	names := []string{"od_x", "od_mid", "od_y", "od_wz", "od_wo"}
	for _, n := range names {
		BufFree(n)
	}
	if err := BufAlloc("od_x", len(x)); err != nil {
		return err
	}
	if err := BufAlloc("od_mid", N*4); err != nil {
		return err
	}
	if err := BufAlloc("od_y", N*4); err != nil {
		return err
	}
	if err := BufAlloc("od_wz", len(wz)); err != nil {
		return err
	}
	if err := BufAlloc("od_wo", len(wo)); err != nil {
		return err
	}
	zeros := make([]byte, N*4)
	for _, p := range []struct {
		n string
		b []byte
	}{
		{"od_x", x}, {"od_wz", wz}, {"od_wo", wo}, {"od_mid", zeros}, {"od_y", zeros},
	} {
		if err := BufPut(p.n, p.b); err != nil {
			return fmt.Errorf("put %s: %w", p.n, err)
		}
	}
	optiqDecodeMeta = meta
	optiqDecodeN = N
	optiqDecodeK = K
	optiqDecodeReady = true
	return nil
}

// MaybeOptiqGraphDecodeStep runs session-resident Wz→Wo GRAPH in the RELEASE gap.
// Call every decode step (not once). Under owned, stashes y for TakeOwnedY.
func MaybeOptiqGraphDecodeStep() error {
	if !OptiqGraphDecodeEnabled() {
		return nil
	}
	if err := EnsureOptiqDecodeSession(); err != nil {
		if optiqDecodeRequire() {
			return err
		}
		return nil
	}
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()

	N, K := optiqDecodeN, optiqDecodeK
	nodes := strings.Join([]string{
		"GEMV_Q4_G64@GPU! x=od_x y=od_mid w=od_wz N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"GEMV_Q4_G64@GPU! x=od_mid y=od_y w=od_wo N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"MARK@GPU?",
	}, " ; ")
	job, err := FormatGraph(1, "chain", nodes)
	if err != nil {
		return err
	}
	resp, err := Graph("mlxrunner-optiq-decode", job, 180)
	if err != nil {
		return err
	}
	if !strings.Contains(resp, "GEMV_Q4_G64") {
		return fmt.Errorf("unexpected graph reply: %s", resp)
	}
	gotb := make([]byte, N*4)
	n, err := BufGet("od_y", gotb)
	if err != nil {
		return err
	}
	if n != len(gotb) {
		return fmt.Errorf("buf get nbytes=%d", n)
	}
	got := optiqBytesF32(gotb)

	dump := os.Getenv("UMA_OPTIQ_DUMP_DIR")
	if dump == "" {
		dump = "/tmp/uma_optiq_live_dump"
	}
	yHostB, err := os.ReadFile(filepath.Join(dump, "y_host.bin"))
	if err != nil {
		return err
	}
	yRef := optiqBytesF32(yHostB)
	var maxe float32
	for i := 0; i < N; i++ {
		e := float32(math.Abs(float64(got[i] - yRef[i])))
		if e > maxe {
			maxe = e
		}
	}
	if maxe > 1e-3 {
		return fmt.Errorf("optiq decode maxerr=%g", maxe)
	}
	optiqDecodeSteps++
	// Owned mode: live InProjZ Forward consumes broker GEMV (OwnedInProjZGemv).
	// Gap rematch is health-only on dump x — do not set pending y.
	return nil
}

// OptiqDecodeOwned reports owned mode (broker result must feed Forward).
func OptiqDecodeOwned() bool {
	return optiqDecodeOwned()
}

// RegisterOwnedLinearTarget marks the QuantizedLinear pointer that Forward should replace.
func RegisterOwnedLinearTarget(ql any) {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	optiqOwnedTarget = ql
}

// OwnedTargetMatch reports whether ql is the registered owned linear.
func OwnedTargetMatch(ql any) bool {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	return optiqOwnedTarget != nil && optiqOwnedTarget == ql
}

// OwnedInProjZGemv runs session-resident GEMV_Q4 (Wz) on live x (K floats).
// Used from QuantizedLinear.Forward under owned mode — broker result is the forward output.
func OwnedInProjZGemv(x []float32) ([]float32, error) {
	if err := EnsureOptiqDecodeSession(); err != nil {
		return nil, err
	}
	optiqDecodeMu.Lock()
	N, K := optiqDecodeN, optiqDecodeK
	optiqDecodeMu.Unlock()
	if len(x) < K {
		return nil, fmt.Errorf("owned gemv: x len=%d want >=K=%d", len(x), K)
	}
	if len(x) > K {
		x = x[len(x)-K:] // last token row
	}
	xb := F32Bytes(x)
	if err := BufPut("od_x", xb); err != nil {
		return nil, err
	}
	zeros := make([]byte, N*4)
	_ = BufPut("od_mid", zeros)
	nodes := "GEMV_Q4_G64@GPU! x=od_x y=od_mid w=od_wz N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K) + " ; MARK@GPU?"
	job, err := FormatGraph(1, "chain", nodes)
	if err != nil {
		return nil, err
	}
	resp, err := Graph("mlxrunner-optiq-owned-z", job, 180)
	if err != nil {
		return nil, err
	}
	if !strings.Contains(resp, "GEMV_Q4_G64") {
		return nil, fmt.Errorf("owned gemv unexpected reply: %s", resp)
	}
	gotb := make([]byte, N*4)
	n, err := BufGet("od_mid", gotb)
	if err != nil {
		return nil, err
	}
	if n != len(gotb) {
		return nil, fmt.Errorf("owned gemv nbytes=%d", n)
	}
	optiqDecodeMu.Lock()
	optiqOwnedConsumed++
	optiqOwnedPending = false
	optiqDecodeMu.Unlock()
	return optiqBytesF32(gotb), nil
}

// SetOwnedForwardErr records a fail-closed error from the Forward hook.
func SetOwnedForwardErr(err error) {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	optiqOwnedFwdErr = err
}

// TakeOwnedForwardErr returns and clears the last owned-forward error.
func TakeOwnedForwardErr() error {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	err := optiqOwnedFwdErr
	optiqOwnedFwdErr = nil
	return err
}

// TakeOwnedY returns the last broker GRAPH y and clears the pending flag.
// Under owned mode, pipeline must call this so the forward path consumes broker output.
func TakeOwnedY() ([]float32, bool) {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	if !optiqOwnedPending || len(optiqOwnedY) == 0 {
		return nil, false
	}
	y := append([]float32(nil), optiqOwnedY...)
	optiqOwnedPending = false
	optiqOwnedConsumed++
	return y, true
}

// OwnedYPending reports whether owned mode still has an unconsumed y (fail-closed check).
func OwnedYPending() bool {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	return optiqOwnedPending
}

// OptiqDecodeSteps returns how many always-on GRAPH decode steps ran.
func OptiqDecodeSteps() int {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	return optiqDecodeSteps
}

// OptiqOwnedConsumed returns how many times TakeOwnedY succeeded.
func OptiqOwnedConsumed() int {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	return optiqOwnedConsumed
}

// ResetOptiqDecodeForTest clears session state (smokes only).
func ResetOptiqDecodeForTest() {
	optiqDecodeMu.Lock()
	defer optiqDecodeMu.Unlock()
	for _, n := range []string{"od_x", "od_mid", "od_y", "od_wz", "od_wo"} {
		BufFree(n)
	}
	optiqDecodeReady = false
	optiqDecodeMeta = nil
	optiqDecodeSteps = 0
	optiqOwnedY = nil
	optiqOwnedPending = false
	optiqOwnedConsumed = 0
	optiqOwnedFwdErr = nil
	optiqOwnedTarget = nil
}

// F32Bytes encodes float32 slice as little-endian bytes (tests / inject helpers).
func F32Bytes(xs []float32) []byte {
	b := make([]byte, len(xs)*4)
	for i, v := range xs {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}
