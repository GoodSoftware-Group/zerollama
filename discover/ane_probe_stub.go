//go:build !darwin

package discover

import (
	"context"
	"fmt"
	"io"
	"runtime"
)

// ANEProbeResult is JSON emitted by tools/ane-probe/ane-probe (Darwin builds only).
type ANEProbeResult struct {
	OK           bool    `json:"ok"`
	EvalMS       float64 `json:"eval_ms"`
	CompileCount int     `json:"compile_count"`
	Channels     int     `json:"channels"`
	Spatial      int     `json:"spatial"`
	Source       string  `json:"source"`
	Error        string  `json:"error,omitempty"`
}

// FindANEProbeBin locates the ane-probe helper (Darwin only).
func FindANEProbeBin() string {
	return ""
}

// ANERepoPath returns the maderix/ane checkout (Darwin lab tooling only).
func ANERepoPath() string {
	return ""
}

// RunANEProbe is unsupported off Darwin.
func RunANEProbe(_ context.Context, _ io.Writer) error {
	return fmt.Errorf("ane-probe: unsupported on %s", runtime.GOOS)
}

// ProbeANE is unsupported off Darwin.
func ProbeANE(_ context.Context) (ANEProbeResult, error) {
	return ANEProbeResult{}, fmt.Errorf("ane-probe: unsupported on %s", runtime.GOOS)
}
