//go:build !darwin

package discover

import (
	"context"
	"fmt"
	"io"
	"runtime"
)

// RunANEProbe is unsupported off Darwin.
func RunANEProbe(_ context.Context, _ io.Writer) error {
	return fmt.Errorf("ane-probe: unsupported on %s", runtime.GOOS)
}

// ProbeANE is unsupported off Darwin.
func ProbeANE(_ context.Context) (ANEProbeResult, error) {
	return ANEProbeResult{}, fmt.Errorf("ane-probe: unsupported on %s", runtime.GOOS)
}
