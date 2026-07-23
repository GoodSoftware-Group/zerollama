//go:build !uma || !darwin

package uma

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

// RunGPU calls fn directly.
func RunGPU(fn func()) error {
	if fn != nil {
		fn()
	}
	return nil
}
