//go:build !uma || !darwin

package uma

import (
	"context"
	"fmt"
	"math"
)

// BuildEnabled reports that this binary was compiled with -tags uma.
func BuildEnabled() bool { return false }

// RuntimeEnabled is always false in the stub.
func RuntimeEnabled() bool { return false }

// Active is always false in the stub.
func Active() bool { return false }

// Acquire is a no-op.
func Acquire() error { return nil }

// Release is a no-op.
func Release() {}

// LeaseBegin is a no-op.
func LeaseBegin(phase string) error { return nil }

// LeaseEnd is a no-op.
func LeaseEnd() {}

// LeaseBeginUnit is a no-op.
func LeaseBeginUnit(unit, phase string) error { return nil }

// LeaseEndUnit is a no-op.
func LeaseEndUnit(unit string) {}

// RunGPU calls fn directly.
func RunGPU(fn func()) error {
	return RunUnit("gpu", fn)
}

// RunUnit calls fn directly.
func RunUnit(unit string, fn func()) error {
	_ = unit
	if fn != nil {
		fn()
	}
	return nil
}

// Grain reports "phase" in the stub.
func Grain() string { return "phase" }

// FormatGraph is unavailable without -tags uma.
func FormatGraph(ntok int, form, nodes string) (string, error) {
	return "", fmt.Errorf("uma: build with -tags uma on Darwin")
}

// FormatGraphEx is unavailable without -tags uma.
func FormatGraphEx(level string, ntok int, form, nodes string, ngen, eos int, toks string) (string, error) {
	return "", fmt.Errorf("uma: build with -tags uma on Darwin")
}

// Submit is unavailable without -tags uma.
func Submit(project, job string) (uint64, error) {
	return 0, fmt.Errorf("uma: build with -tags uma on Darwin")
}

// Wait is unavailable without -tags uma.
func Wait(ticket uint64, timeoutSec float64) (string, error) {
	return "", fmt.Errorf("uma: build with -tags uma on Darwin")
}

// Graph is unavailable without -tags uma.
func Graph(project, job string, timeoutSec float64) (string, error) {
	return "", fmt.Errorf("uma: build with -tags uma on Darwin")
}

func BufAlloc(name string, nbytes int) error {
	return fmt.Errorf("uma: build with -tags uma on Darwin")
}
func BufFree(name string) {}
func BufPut(name string, data []byte) error {
	return fmt.Errorf("uma: build with -tags uma on Darwin")
}
func BufGet(name string, dst []byte) (int, error) {
	return 0, fmt.Errorf("uma: build with -tags uma on Darwin")
}
func BufExport(name string) (uint32, uint32, error) {
	return 0, 0, fmt.Errorf("uma: build with -tags uma on Darwin")
}
func BufReclaim(name string, token uint32) error {
	return fmt.Errorf("uma: build with -tags uma on Darwin")
}

// MaybeProbeOptiqLiveChain is a no-op without -tags uma.
func MaybeProbeOptiqLiveChain() (bool, error) { return false, nil }

// OptiqGraphGenerateEnabled is false without -tags uma.
func OptiqGraphGenerateEnabled() bool { return false }

// OptiqGraphGenerateRequire is false without -tags uma.
func OptiqGraphGenerateRequire() bool { return false }

// RunOptiqGraphGenerate is unavailable without -tags uma.
func RunOptiqGraphGenerate(ctx context.Context, prompt []int32, nGen int) ([]int32, error) {
	return nil, fmt.Errorf("uma: build with -tags uma on Darwin")
}

// OptiqGraphDecodeEnabled is false without -tags uma.
func OptiqGraphDecodeEnabled() bool { return false }

// EnsureOptiqDecodeSession is a no-op without -tags uma.
func EnsureOptiqDecodeSession() error { return nil }

// MaybeOptiqGraphDecodeStep is a no-op without -tags uma.
func MaybeOptiqGraphDecodeStep() error { return nil }

// TakeOwnedY is a no-op without -tags uma.
func TakeOwnedY() ([]float32, bool) { return nil, false }

// OwnedYPending is false without -tags uma.
func OwnedYPending() bool { return false }

// OptiqDecodeSteps is 0 without -tags uma.
func OptiqDecodeSteps() int { return 0 }

// OptiqOwnedConsumed is 0 without -tags uma.
func OptiqOwnedConsumed() int { return 0 }

// ResetOptiqDecodeForTest is a no-op without -tags uma.
func ResetOptiqDecodeForTest() {}

// OptiqDecodeOwned is false without -tags uma.
func OptiqDecodeOwned() bool { return false }

// RegisterOwnedLinearTarget is a no-op without -tags uma.
func RegisterOwnedLinearTarget(ql any) {}

// OwnedTargetMatch is false without -tags uma.
func OwnedTargetMatch(ql any) bool { return false }

// OwnedInProjZGemv is unavailable without -tags uma.
func OwnedInProjZGemv(x []float32) ([]float32, error) {
	return nil, fmt.Errorf("uma: build with -tags uma on Darwin")
}

// SetOwnedForwardErr is a no-op without -tags uma.
func SetOwnedForwardErr(err error) {}

// TakeOwnedForwardErr is a no-op without -tags uma.
func TakeOwnedForwardErr() error { return nil }

// OptiqGraphTokenEnabled is false without -tags uma.
func OptiqGraphTokenEnabled() bool { return false }

// OptiqGraphTokenOwned is false without -tags uma.
func OptiqGraphTokenOwned() bool { return false }

// OptiqTokenTailEnabled is false without -tags uma.
func OptiqTokenTailEnabled() bool { return false }

// OptiqTokenTailRequire is false without -tags uma.
func OptiqTokenTailRequire() bool { return false }

// OptiqTokenTailOwned is false without -tags uma.
func OptiqTokenTailOwned() bool { return false }

// OptiqTokenTailSessionReady is false without -tags uma.
func OptiqTokenTailSessionReady() bool { return false }

// EnsureOptiqTokenTailSession is a no-op without -tags uma.
func EnsureOptiqTokenTailSession() error { return nil }

// OptiqTokenTailArgmax is unavailable without -tags uma.
func OptiqTokenTailArgmax(x []float32) (int32, error) {
	return -1, fmt.Errorf("uma: build with -tags uma on Darwin")
}

// OwnedTokenTailArgmax is unavailable without -tags uma.
func OwnedTokenTailArgmax(x []float32) (int32, error) {
	return -1, fmt.Errorf("uma: build with -tags uma on Darwin")
}

// OptiqTokenTailSteps is 0 without -tags uma.
func OptiqTokenTailSteps() int { return 0 }

// OptiqTokenTailOwnedCount is 0 without -tags uma.
func OptiqTokenTailOwnedCount() int { return 0 }

// ResetOptiqTokenTailForTest is a no-op without -tags uma.
func ResetOptiqTokenTailForTest() {}

// F32Bytes encodes float32 little-endian (available in stub for tests).
func F32Bytes(xs []float32) []byte {
	b := make([]byte, len(xs)*4)
	for i, v := range xs {
		u := math.Float32bits(v)
		b[i*4] = byte(u)
		b[i*4+1] = byte(u >> 8)
		b[i*4+2] = byte(u >> 16)
		b[i*4+3] = byte(u >> 24)
	}
	return b
}
