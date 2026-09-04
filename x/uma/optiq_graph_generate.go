//go:build darwin && uma

package uma

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Optiq GRAPH generate (F0698/F0699/F0719). Modes via ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE:
//
//	"" / off  — disabled
//	1 / on    — in-process F0697 cascade L0..31 (cgo); soft-fail → MLX decode
//	require   — same, abort caller on failure
//
// Default path is in-process via libuma_optiq_graph_gen.dylib (F0699; no os/exec).
// Soft exec fallback only when UMA_OPTIQ_GRAPH_GENERATE_ALLOW_EXEC=1.
// F0719: pass request prompt token ids (any-prompt). Freeze rematch when
// prompt matches dump prompt_ids (got_gen [12675,248046]).
func optiqGraphGenerateMode() string {
	return strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE")))
}

// OptiqGraphGenerateEnabled reports whether opt-in GRAPH generate is on.
func OptiqGraphGenerateEnabled() bool {
	switch optiqGraphGenerateMode() {
	case "1", "on", "true", "yes", "require":
		return true
	default:
		return false
	}
}

// OptiqGraphGenerateRequire reports fail-closed mode.
func OptiqGraphGenerateRequire() bool {
	return optiqGraphGenerateMode() == "require"
}

func optiqGraphGenerateDumpDir() string {
	if d := strings.TrimSpace(os.Getenv("ORNITH_OPTIQ_GENERATE_DIR")); d != "" {
		return d
	}
	return "/tmp/uma_optiq_generate_dump"
}

// OptiqGraphGenerateLastMode is "in-process" or "exec" after a successful Run.
var OptiqGraphGenerateLastMode string

func resolveOptiqToolkit() string {
	if tk := strings.TrimSpace(os.Getenv("UMA_TOOLKIT")); tk != "" {
		return tk
	}
	if wd, err := os.Getwd(); err == nil {
		for _, p := range []string{
			filepath.Join(wd, "..", "bmtl", "hardware_lab", "lanes", "m4", "uma_toolkit"),
			filepath.Join(wd, "hardware_lab", "lanes", "m4", "uma_toolkit"),
		} {
			if st, err := os.Stat(p); err == nil && st.IsDir() {
				return p
			}
		}
	}
	home, _ := os.UserHomeDir()
	if home != "" {
		p := filepath.Join(home, "Sites", "inference", "bmtl", "hardware_lab", "lanes", "m4", "uma_toolkit")
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			return p
		}
	}
	return "/Users/user1/Sites/inference/bmtl/hardware_lab/lanes/m4/uma_toolkit"
}

func resolveOptiqGraphGenerateDylib() string {
	name := "libuma_optiq_graph_gen.dylib"
	tk := resolveOptiqToolkit()
	p := filepath.Join(tk, name)
	if st, err := os.Stat(p); err == nil && !st.IsDir() {
		return p
	}
	p = filepath.Join(tk, "src", name)
	if st, err := os.Stat(p); err == nil && !st.IsDir() {
		return p
	}
	return ""
}

// resolveOptiqGraphGenerateBin locates the F0697 cascade L0 smoke binary (exec fallback).
func resolveOptiqGraphGenerateBin() (string, error) {
	if p := strings.TrimSpace(os.Getenv("UMA_OPTIQ_GRAPH_GENERATE_BIN")); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
		return "", fmt.Errorf("UMA_OPTIQ_GRAPH_GENERATE_BIN=%s not found", p)
	}
	name := "test_uma_optiq_live_cascade_l0_multit_generate_smoke"
	tk := resolveOptiqToolkit()
	p := filepath.Join(tk, name)
	if st, err := os.Stat(p); err == nil && !st.IsDir() {
		return p, nil
	}
	p = filepath.Join(tk, "tests", name)
	if st, err := os.Stat(p); err == nil && !st.IsDir() {
		return p, nil
	}
	return "", fmt.Errorf("optiq GRAPH generate binary not found (set UMA_OPTIQ_GRAPH_GENERATE_BIN or UMA_TOOLKIT; build: make -C uma_toolkit libuma_optiq_graph_gen.dylib)")
}

// parseGraphGenTokens extracts ids from a GRAPH_GEN_TOKENS=a,b,... line.
func parseGraphGenTokens(out []byte) ([]int32, error) {
	for _, line := range bytes.Split(out, []byte{'\n'}) {
		s := strings.TrimSpace(string(line))
		if !strings.HasPrefix(s, "GRAPH_GEN_TOKENS=") {
			continue
		}
		body := strings.TrimPrefix(s, "GRAPH_GEN_TOKENS=")
		if body == "" {
			return nil, fmt.Errorf("empty GRAPH_GEN_TOKENS")
		}
		parts := strings.Split(body, ",")
		ids := make([]int32, 0, len(parts))
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			v, err := strconv.ParseInt(p, 10, 32)
			if err != nil {
				return nil, fmt.Errorf("parse GRAPH_GEN_TOKENS %q: %w", p, err)
			}
			ids = append(ids, int32(v))
		}
		if len(ids) == 0 {
			return nil, fmt.Errorf("empty GRAPH_GEN_TOKENS body")
		}
		return ids, nil
	}
	return nil, fmt.Errorf("no GRAPH_GEN_TOKENS= line in binary output")
}

func runOptiqGraphGenerateExec(ctx context.Context) ([]int32, error) {
	bin, err := resolveOptiqGraphGenerateBin()
	if err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, bin)
	cmd.Dir = filepath.Dir(bin)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("optiq GRAPH generate binary failed: %w\nstderr: %s\nstdout: %s", err, stderr.String(), stdout.String())
	}
	ids, err := parseGraphGenTokens(stdout.Bytes())
	if err != nil {
		return nil, fmt.Errorf("%w\nstdout: %s", err, stdout.String())
	}
	OptiqGraphGenerateLastMode = "exec"
	slog.Info("optiq GRAPH generate via exec fallback", "bin", bin, "n", len(ids))
	return ids, nil
}

// RunOptiqGraphGenerate obtains generate-suffix tokens via in-process
// C API (libuma_optiq_graph_gen.dylib / cgo). F0719: pass prompt token ids
// from the request (any-prompt Forward). Empty prompt falls back to dump
// prompt_ids (F0699 compat). nGen<=0 uses dump meta npred.
// No os/exec unless UMA_OPTIQ_GRAPH_GENERATE_ALLOW_EXEC=1 and in-process fails.
func RunOptiqGraphGenerate(ctx context.Context, prompt []int32, nGen int) ([]int32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dump := optiqGraphGenerateDumpDir()
	meta := filepath.Join(dump, "meta.json")
	if _, err := os.Stat(meta); err != nil {
		return nil, fmt.Errorf("optiq generate dump missing at %s (run: go run ./x/mlxrunner/cmd/ornith_generate_parity): %w", meta, err)
	}
	ids, err := runOptiqGraphGenerateInProcess(prompt, nGen)
	if err == nil {
		return ids, nil
	}
	allowExec := strings.TrimSpace(os.Getenv("UMA_OPTIQ_GRAPH_GENERATE_ALLOW_EXEC")) == "1"
	if !allowExec {
		return nil, fmt.Errorf("in-process GRAPH generate failed (set UMA_OPTIQ_GRAPH_GENERATE_ALLOW_EXEC=1 for smoke-binary fallback): %w", err)
	}
	slog.Warn("optiq GRAPH generate in-process failed; trying exec", "err", err)
	return runOptiqGraphGenerateExec(ctx)
}
