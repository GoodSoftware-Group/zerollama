//go:build darwin

// ANE probe integration — subprocess smoke test for maderix/ANE libane_bridge.
//
// Why not CGO in zerollama: private _ANEClient APIs break across macOS updates;
// isolating compile/eval in tools/ane-probe keeps the main binary stable and lets
// doctor warn without blocking serve. See docs/ane-probe.md.
package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/reporoots"
)

// ANEProbeResult is JSON emitted by tools/ane-probe/ane-probe.
type ANEProbeResult struct {
	OK           bool    `json:"ok"`
	EvalMS       float64 `json:"eval_ms"`
	CompileCount int     `json:"compile_count"`
	Channels     int     `json:"channels"`
	Spatial      int     `json:"spatial"`
	Source       string  `json:"source"`
	Error        string  `json:"error,omitempty"`
}

const aneProbeTimeout = 45 * time.Second

// FindANEProbeBin locates the ane-probe helper binary under build/ or tools/.
func FindANEProbeBin() string {
	if p := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_PROBE")); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
	}

	candidates := []string{
		"build/ane-probe-darwin/bin/ane-probe",
		"tools/ane-probe/ane-probe",
	}
	for _, root := range reporoots.SearchRoots() {
		for _, rel := range candidates {
			p := filepath.Join(root, rel)
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p
			}
		}
	}
	return ""
}

// ANERepoPath returns the maderix/ane checkout used to build libane_bridge.
func ANERepoPath() string {
	return envconfig.ANERepo()
}

// RunANEProbe executes the ane-probe binary and writes its JSON stdout to w.
func RunANEProbe(ctx context.Context, w io.Writer) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("ane-probe: darwin only (got %s)", runtime.GOOS)
	}

	bin := FindANEProbeBin()
	if bin == "" {
		return fmt.Errorf("ane-probe binary not found — run ./scripts/ane/ane_probe_build.sh (set ANE_REPO if needed)")
	}

	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithTimeout(ctx, aneProbeTimeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, bin)
	cmd.Env = os.Environ()
	// Why Dir = bin dir: dyld resolves @loader_path/libane_bridge.dylib next to the executable.
	cmd.Dir = filepath.Dir(bin)
	out, err := cmd.Output()
	if err != nil {
		if len(out) > 0 {
			_, _ = w.Write(out)
		}
		return fmt.Errorf("ane-probe %s: %w", bin, err)
	}
	_, err = w.Write(out)
	return err
}

// ProbeANE runs ane-probe and parses the JSON result.
func ProbeANE(ctx context.Context) (ANEProbeResult, error) {
	var buf strings.Builder
	if err := RunANEProbe(ctx, &buf); err != nil {
		return ANEProbeResult{}, err
	}
	var res ANEProbeResult
	if err := json.Unmarshal([]byte(strings.TrimSpace(buf.String())), &res); err != nil {
		return ANEProbeResult{}, fmt.Errorf("ane-probe json: %w", err)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "probe returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}
