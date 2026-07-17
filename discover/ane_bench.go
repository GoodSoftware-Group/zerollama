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

	"github.com/ollama/ollama/internal/reporoots"
)

// ANEBenchResult is JSON from ane-matmul-bench (conv peak proxy).
type ANEBenchResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	Channels     int     `json:"channels"`
	Spatial      int     `json:"spatial"`
	Depth        int     `json:"depth"`
	EvalMS       float64 `json:"eval_ms"`
	GFLOP        float64 `json:"gflop"`
	TFLOPS       float64 `json:"tflops"`
	CompileCount int     `json:"compile_count"`
	Source       string  `json:"source"`
	Error        string  `json:"error,omitempty"`
}

// ANEDraftBenchResult is JSON from ane-draft-bench (draft conv proxy).
type ANEDraftBenchResult struct {
	OK           bool    `json:"ok"`
	Mode         string  `json:"mode"`
	Channels     int     `json:"channels"`
	Spatial      int     `json:"spatial"`
	EvalMS       float64 `json:"eval_ms"`
	CompileCount int     `json:"compile_count"`
	Source       string  `json:"source"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

const aneBenchTimeout = 120 * time.Second

func aneToolBin(name string) string {
	if name == "ane-probe" {
		if p := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_PROBE")); p != "" {
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p
			}
		}
	}
	envKey := "ZEROLLAMA_" + strings.ToUpper(strings.ReplaceAll(name, "-", "_"))
	if p := strings.TrimSpace(os.Getenv(envKey)); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
	}

	candidates := []string{
		filepath.Join("build", "ane-probe-darwin", "bin", name),
		filepath.Join("tools", "ane-inprocess", name),
		filepath.Join("tools", "ane-metal", name),
		filepath.Join("tools", "ane-ggml-map", name),
		filepath.Join("tools", "ane-prefill", name),
		filepath.Join("tools", strings.TrimSuffix(name, "-bench"), name),
		filepath.Join("tools", "ane-probe", name),
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

// FindANEMatmulBenchBin locates the peak throughput bench binary.
func FindANEMatmulBenchBin() string {
	return aneToolBin("ane-matmul-bench")
}

// FindANEDraftBenchBin locates the draft-step latency bench binary.
func FindANEDraftBenchBin() string {
	return aneToolBin("ane-draft-bench")
}

func runANETool(ctx context.Context, bin string, args []string) ([]byte, error) {
	if runtime.GOOS != "darwin" {
		return nil, fmt.Errorf("ane tools: darwin only (got %s)", runtime.GOOS)
	}
	if bin == "" {
		return nil, fmt.Errorf("ane tool binary not found")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithTimeout(ctx, aneBenchTimeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, bin, args...)
	cmd.Env = os.Environ()
	cmd.Dir = filepath.Dir(bin)
	out, err := cmd.Output()
	if err != nil {
		if len(out) > 0 {
			return out, fmt.Errorf("%s: %w", filepath.Base(bin), err)
		}
		return nil, fmt.Errorf("%s: %w", filepath.Base(bin), err)
	}
	return out, nil
}

// RunANEMatmulBench runs peak conv-stack bench; quick=true uses --quick.
func RunANEMatmulBench(ctx context.Context, w io.Writer, quick bool) error {
	bin := FindANEMatmulBenchBin()
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

// ProbeANEMatmulBench parses peak bench JSON.
func ProbeANEMatmulBench(ctx context.Context, quick bool) (ANEBenchResult, error) {
	bin := FindANEMatmulBenchBin()
	if bin == "" {
		return ANEBenchResult{}, fmt.Errorf("ane-matmul-bench not found — run ./scripts/ane/ane_probe_build.sh")
	}
	args := []string{}
	if quick {
		args = append(args, "--quick")
	}
	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEBenchResult{}, err
	}
	var res ANEBenchResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEBenchResult{}, fmt.Errorf("ane-matmul-bench json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "bench returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

func aneDraftBenchArgs(channels, spatial int, quick bool) []string {
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

// RunANEDraftBench runs draft-step matmul proxy bench.
func RunANEDraftBench(ctx context.Context, w io.Writer, quick bool) error {
	return RunANEDraftBenchDims(ctx, w, 0, 0, quick)
}

// RunANEDraftBenchDims runs draft bench at explicit conv proxy dimensions.
func RunANEDraftBenchDims(ctx context.Context, w io.Writer, channels, spatial int, quick bool) error {
	bin := FindANEDraftBenchBin()
	out, err := runANETool(ctx, bin, aneDraftBenchArgs(channels, spatial, quick))
	if len(out) > 0 {
		_, _ = w.Write(out)
	}
	return err
}

// RunANEDraftBenchForModel resolves a local dflash tag and benches at GGUF-derived proxy dims.
func RunANEDraftBenchForModel(ctx context.Context, w io.Writer, preferred string, quick bool) error {
	entries, err := ListANEDraftInventory()
	if err != nil {
		return err
	}
	entry, ok := SelectANEDraftModel(entries, preferred)
	if !ok {
		return fmt.Errorf("no ANE draft target in local inventory")
	}
	ch, sp := entry.ProxyChannels, entry.ProxySpatial
	if ch <= 0 || sp <= 0 {
		if info, err := ProbeANEDraftGGUF(entry.BaseGGUF, entry.DraftGGUF); err == nil {
			ch, sp = info.ProxyChannels, info.ProxySpatial
		}
	}
	if ch <= 0 {
		ch, sp = DraftANEProxyDims(entry.EmbeddingLength)
	}
	return RunANEDraftBenchDims(ctx, w, ch, sp, quick)
}

// ProbeANEDraftBench parses draft bench JSON.
func ProbeANEDraftBench(ctx context.Context, quick bool) (ANEDraftBenchResult, error) {
	return ProbeANEDraftBenchDims(ctx, 0, 0, quick)
}

// ProbeANEDraftBenchDims parses draft bench JSON at explicit dimensions.
func ProbeANEDraftBenchDims(ctx context.Context, channels, spatial int, quick bool) (ANEDraftBenchResult, error) {
	bin := FindANEDraftBenchBin()
	if bin == "" {
		return ANEDraftBenchResult{}, fmt.Errorf("ane-draft-bench not found — run ./scripts/ane/ane_probe_build.sh")
	}
	args := aneDraftBenchArgs(channels, spatial, quick)
	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEDraftBenchResult{}, err
	}
	var res ANEDraftBenchResult
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEDraftBenchResult{}, fmt.Errorf("ane-draft-bench json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "draft bench returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}
