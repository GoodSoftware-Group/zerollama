package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/ollama/ollama/internal/reporoots"
)

// ANEHandoffResult is JSON from ane-iosurface-smoke (CPU producer).
type ANEHandoffResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	Channels     int     `json:"channels"`
	Spatial      int     `json:"spatial"`
	SurfaceBytes int     `json:"surface_bytes"`
	WriteMS      float64 `json:"write_ms"`
	EvalMS       float64 `json:"eval_ms"`
	ReadMS       float64 `json:"read_ms"`
	TotalMS      float64 `json:"total_ms"`
	Source       string  `json:"source"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// ANEMetalHandoffResult is JSON from ane-metal-handoff-smoke (Metal producer).
type ANEMetalHandoffResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	Channels     int     `json:"channels"`
	Spatial      int     `json:"spatial"`
	SurfaceID    uint32  `json:"surface_id"`
	SurfaceBytes int     `json:"surface_bytes"`
	MetalFillMS  float64 `json:"metal_fill_ms"`
	EvalMS       float64 `json:"eval_ms"`
	ReadMS       float64 `json:"read_ms"`
	TotalMS      float64 `json:"total_ms"`
	Source       string  `json:"source"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// FindANEIOSurfaceSmokeBin locates the IOSurface handoff smoke binary.
func FindANEIOSurfaceSmokeBin() string {
	return aneToolBin("ane-iosurface-smoke")
}

// RunANEIOSurfaceSmoke executes IOSurface handoff timing smoke.
func RunANEIOSurfaceSmoke(ctx context.Context, w io.Writer, quick bool) error {
	bin := FindANEIOSurfaceSmokeBin()
	args := []string{}
	if quick {
		args = append(args, "--quick")
	}
	out, err := runANETool(ctx, bin, args)
	if len(out) > 0 {
		_, _ = w.Write(out)
	}
	return err
}

// ProbeANEIOSurfaceSmoke parses handoff smoke JSON.
func ProbeANEIOSurfaceSmoke(ctx context.Context, quick bool) (ANEHandoffResult, error) {
	bin := FindANEIOSurfaceSmokeBin()
	if bin == "" {
		return ANEHandoffResult{}, fmt.Errorf("ane-iosurface-smoke not found — run ./scripts/ane/ane_probe_build.sh")
	}
	args := []string{}
	if quick {
		args = append(args, "--quick")
	}
	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEHandoffResult{}, err
	}
	var res ANEHandoffResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEHandoffResult{}, fmt.Errorf("ane-iosurface-smoke json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "handoff smoke returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

// FindANEMetalHandoffSmokeBin locates the Metal IOSurface handoff smoke binary.
func FindANEMetalHandoffSmokeBin() string {
	return aneToolBin("ane-metal-handoff-smoke")
}

// RunANEMetalHandoffSmoke executes Metal→IOSurface→ANE timing smoke.
func RunANEMetalHandoffSmoke(ctx context.Context, w io.Writer, quick bool) error {
	return RunANEMetalHandoffDims(ctx, w, 0, 0, quick)
}

func aneMetalHandoffArgs(channels, spatial int, quick bool) []string {
	args := []string{}
	if channels > 0 {
		args = append(args, "--channels", fmt.Sprintf("%d", channels))
	}
	if spatial > 0 {
		args = append(args, "--spatial", fmt.Sprintf("%d", spatial))
	}
	if quick {
		args = append(args, "--quick")
	}
	return args
}

// RunANEMetalHandoffDims runs Metal handoff at explicit conv dimensions.
func RunANEMetalHandoffDims(ctx context.Context, w io.Writer, channels, spatial int, quick bool) error {
	bin := FindANEMetalHandoffSmokeBin()
	out, err := runANETool(ctx, bin, aneMetalHandoffArgs(channels, spatial, quick))
	if len(out) > 0 {
		_, _ = w.Write(out)
	}
	return err
}

// ProbeANEMetalHandoffSmoke parses Metal handoff smoke JSON.
func ProbeANEMetalHandoffSmoke(ctx context.Context, quick bool) (ANEMetalHandoffResult, error) {
	return ProbeANEMetalHandoffDims(ctx, 0, 0, quick)
}

// ProbeANEMetalHandoffDims parses Metal handoff JSON at explicit dimensions.
func ProbeANEMetalHandoffDims(ctx context.Context, channels, spatial int, quick bool) (ANEMetalHandoffResult, error) {
	bin := FindANEMetalHandoffSmokeBin()
	if bin == "" {
		return ANEMetalHandoffResult{}, fmt.Errorf("ane-metal-handoff-smoke not found — run ./scripts/ane/ane_probe_build.sh")
	}
	args := aneMetalHandoffArgs(channels, spatial, quick)
	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEMetalHandoffResult{}, err
	}
	var res ANEMetalHandoffResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEMetalHandoffResult{}, fmt.Errorf("ane-metal-handoff-smoke json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "metal handoff smoke returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

// RunANEDraftResolveJSON writes draft inventory as JSON (for scripts/CLI).
func RunANEDraftResolveJSON(w io.Writer, preferred string) error {
	entries, err := ListANEDraftInventory()
	if err != nil {
		return err
	}
	if preferred != "" {
		if e, ok := SelectANEDraftModel(entries, preferred); ok {
			return json.NewEncoder(w).Encode(e)
		}
		return fmt.Errorf("no ANE draft target matching %q", preferred)
	}
	return json.NewEncoder(w).Encode(entries)
}

// RunANEHandoffSuite runs probe + draft + iosurface smokes sequentially (lab only).
func RunANEHandoffSuite(ctx context.Context, w io.Writer, quick bool) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("ane handoff suite: darwin only")
	}
	type step struct {
		name string
		fn   func() error
	}
	steps := []step{
		{"ane-probe", func() error { return RunANEProbe(ctx, w) }},
		{"ane-draft-bench", func() error { return RunANEDraftBench(ctx, w, quick) }},
		{"ane-iosurface-smoke", func() error { return RunANEIOSurfaceSmoke(ctx, w, quick) }},
		{"ane-metal-handoff-smoke", func() error { return RunANEMetalHandoffSmoke(ctx, w, quick) }},
	}
	for _, s := range steps {
		if _, err := fmt.Fprintf(w, "\n"); err != nil {
			return err
		}
		if err := s.fn(); err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}
	}
	return nil
}

// HandoffBinDir returns the directory containing ANE lab binaries.
func HandoffBinDir() string {
	if b := FindANEIOSurfaceSmokeBin(); b != "" {
		return filepath.Dir(b)
	}
	for _, root := range reporoots.SearchRoots() {
		p := filepath.Join(root, "build", "ane-probe-darwin", "bin")
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			return p
		}
	}
	return ""
}
