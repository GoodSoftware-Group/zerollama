package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/x/trainingworker"
)

type doctorCheck struct {
	Name    string `json:"name"`
	Status  string `json:"status"` // ok, warn, fail
	Detail  string `json:"detail,omitempty"`
	FixHint string `json:"fix_hint,omitempty"`
}

type doctorReport struct {
	Checks   []doctorCheck `json:"checks"`
	Failures int           `json:"failures"`
	Warnings int           `json:"warnings"`
	OK       bool          `json:"ok"`
}

func NewDoctorCommand() *cobra.Command {
	var jsonOut bool
	var fix bool
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check local zerollama / Apple Silicon runtime readiness",
		Long:  "Validate uv venv, Metal libllama, sidecar health, and autoconfig on Darwin.",
		RunE: func(_ *cobra.Command, _ []string) error {
			repo := doctorRepoRoot()
			if fix {
				if err := runDoctorFix(repo); err != nil {
					return err
				}
			}
			report := buildDoctorReport(repo)
			if jsonOut {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				if err := enc.Encode(report); err != nil {
					return err
				}
				if !report.OK {
					return fmt.Errorf("doctor: %d check(s) failed, %d warning(s)", report.Failures, report.Warnings)
				}
				return nil
			}
			return printDoctorHuman(report)
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "Print results as JSON")
	cmd.Flags().BoolVar(&fix, "fix", false, "Run safe auto-fixes (uv venv; on Darwin build Metal llama.cpp when missing)")
	return cmd
}

func buildDoctorReport(repo string) doctorReport {
	checks := runDoctorChecks(repo)
	report := doctorReport{Checks: checks}
	for _, c := range checks {
		switch c.Status {
		case "fail":
			report.Failures++
		case "warn":
			report.Warnings++
		}
	}
	report.OK = report.Failures == 0
	return report
}

func printDoctorHuman(report doctorReport) error {
	for _, c := range report.Checks {
		fmt.Printf("[%s] %s\n", c.Status, c.Name)
		if c.Detail != "" {
			fmt.Printf("      %s\n", c.Detail)
		}
		if c.FixHint != "" && c.Status != "ok" {
			fmt.Printf("      → %s\n", c.FixHint)
		}
	}
	fmt.Println()
	if report.Failures > 0 {
		return fmt.Errorf("doctor: %d check(s) failed, %d warning(s)", report.Failures, report.Warnings)
	}
	if report.Warnings > 0 {
		fmt.Printf("doctor: ok with %d warning(s)\n", report.Warnings)
		return nil
	}
	fmt.Println("doctor: all checks passed")
	return nil
}

func runDoctorFix(repo string) error {
	fmt.Println("== doctor --fix: runtime venv ==")
	script := filepath.Join(repo, "scripts", "runtime_uv_venv.sh")
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("missing %s", script)
	}
	cmd := exec.Command("bash", script)
	cmd.Dir = repo
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("runtime_uv_venv: %w", err)
	}
	if runtime.GOOS == "darwin" && os.Getenv("DOCTOR_FIX_BUILD") != "0" {
		if doctorFindLibLlama(repo) == "" {
			build := filepath.Join(repo, "scripts", "build_llama_server.sh")
			if _, err := os.Stat(build); err == nil {
				fmt.Println("== doctor --fix: build Metal llama.cpp ==")
				bcmd := exec.Command("bash", build)
				bcmd.Dir = repo
				bcmd.Stdout = os.Stdout
				bcmd.Stderr = os.Stderr
				if err := bcmd.Run(); err != nil {
					return fmt.Errorf("build_llama_server: %w", err)
				}
			}
		}
	}
	return nil
}

func runDoctorChecks(repo string) []doctorCheck {
	var out []doctorCheck

	out = append(out, doctorCheckGo())
	out = append(out, doctorCheckZerollamaBinary(repo))
	out = append(out, doctorCheckUV())
	out = append(out, doctorCheckRuntimeVenv(repo))
	out = append(out, doctorCheckLibLlama(repo))

	if runtime.GOOS == "darwin" {
		out = append(out, doctorCheckServeModes())
		out = append(out, doctorCheckTrainingVenv(repo))
		out = append(out, doctorCheckSidecarHealth())
		out = append(out, doctorCheckTextGGUF(repo))
	} else {
		out = append(out, doctorCheck{
			Name:   "darwin runtime smoke",
			Status: "warn",
			Detail: "full sidecar checks run on darwin only",
		})
	}
	return out
}

func doctorRepoRoot() string {
	for _, key := range []string{"ZEROLLAMA_REPO", "OLLAMA_TRAINING_PYTHONPATH"} {
		if p := strings.TrimSpace(os.Getenv(key)); p != "" {
			return filepath.Clean(p)
		}
	}
	if root, err := trainingworker.RepoRoot(); err == nil && root != "" {
		return root
	}
	if wd, err := os.Getwd(); err == nil {
		for dir := wd; ; dir = filepath.Dir(dir) {
			if st, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !st.IsDir() {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
		}
	}
	return "."
}

func doctorCheckGo() doctorCheck {
	return doctorCheck{
		Name:   "platform",
		Status: "ok",
		Detail: fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
	}
}

func doctorCheckZerollamaBinary(repo string) doctorCheck {
	bin := filepath.Join(repo, "zerollama")
	if _, err := os.Stat(bin); err != nil {
		if p, err := exec.LookPath("zerollama"); err == nil {
			bin = p
		} else {
			return doctorCheck{
				Name:    "zerollama binary",
				Status:  "warn",
				Detail:  "not found in repo or PATH",
				FixHint: "go build -o zerollama .",
			}
		}
	}
	cmd := exec.Command(bin, "serve", "--help")
	if err := cmd.Run(); err != nil {
		return doctorCheck{
			Name:    "zerollama binary",
			Status:  "fail",
			Detail:  fmt.Sprintf("%s serve --help failed: %v", bin, err),
			FixHint: "rebuild with go build -o zerollama .",
		}
	}
	// Stale binary may lack doctor subcommand.
	docCmd := exec.Command(bin, "doctor", "--help")
	if err := docCmd.Run(); err != nil {
		return doctorCheck{
			Name:    "zerollama binary",
			Status:  "warn",
			Detail:  fmt.Sprintf("%s lacks doctor subcommand (stale build?)", bin),
			FixHint: "go build -o zerollama .",
		}
	}
	return doctorCheck{
		Name:   "zerollama binary",
		Status: "ok",
		Detail: bin,
	}
}

func doctorCheckUV() doctorCheck {
	if _, err := exec.LookPath("uv"); err != nil {
		return doctorCheck{
			Name:    "uv",
			Status:  "fail",
			Detail:  "not on PATH",
			FixHint: "install from https://docs.astral.sh/uv/",
		}
	}
	out, _ := exec.Command("uv", "--version").CombinedOutput()
	return doctorCheck{
		Name:   "uv",
		Status: "ok",
		Detail: strings.TrimSpace(string(out)),
	}
}

func doctorCheckRuntimeVenv(repo string) doctorCheck {
	py := filepath.Join(repo, "runtime", ".venv", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		return doctorCheck{
			Name:    "runtime/.venv",
			Status:  "fail",
			Detail:  "missing",
			FixHint: "./scripts/runtime_uv_venv.sh or zerollama doctor --fix",
		}
	}
	cmd := exec.Command(py, "-c", "import fastapi")
	if err := cmd.Run(); err != nil {
		return doctorCheck{
			Name:    "runtime/.venv",
			Status:  "fail",
			Detail:  "fastapi import failed",
			FixHint: "RUNTIME_UV_SYNC=1 ./scripts/runtime_uv_venv.sh",
		}
	}
	return doctorCheck{
		Name:   "runtime/.venv",
		Status: "ok",
		Detail: py,
	}
}

func doctorFindLibLlama(repo string) string {
	candidates := doctorLibLlamaCandidates(repo)
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && st.Size() > 0 {
			return p
		}
	}
	return ""
}

func doctorLibLlamaCandidates(repo string) []string {
	candidates := []string{}
	if p := strings.TrimSpace(os.Getenv("LLAMA_CPP_LIB")); p != "" {
		candidates = append(candidates, p)
	}
	root := strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT"))
	if root == "" {
		root = filepath.Clean(filepath.Join(repo, "..", "llama.cpp"))
	}
	if runtime.GOOS == "darwin" {
		candidates = append(candidates, filepath.Join(root, "build", "bin", "libllama.dylib"))
	} else {
		candidates = append(candidates, filepath.Join(root, "build", "bin", "libllama.so"))
	}
	return candidates
}

func doctorCheckLibLlama(repo string) doctorCheck {
	if p := doctorFindLibLlama(repo); p != "" {
		return doctorCheck{
			Name:   "libllama",
			Status: "ok",
			Detail: p,
		}
	}
	return doctorCheck{
		Name:    "libllama",
		Status:  "fail",
		Detail:  "Metal/CUDA libllama not found",
		FixHint: "LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh or zerollama doctor --fix",
	}
}

func doctorOllamaHost() string {
	if u := strings.TrimSpace(os.Getenv("OLLAMA_HOST")); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return "http://127.0.0.1:11434"
}

func doctorRuntimeURL() string {
	if u := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_URL")); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return "http://127.0.0.1:8081"
}

func doctorHTTPReachable(url string) bool {
	client := &http.Client{Timeout: 1500 * time.Millisecond}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 500
}

func doctorCheckServeModes() doctorCheck {
	goHost := doctorOllamaHost()
	sidecarURL := doctorRuntimeURL()
	sidecarUp := doctorHTTPReachable(sidecarURL + "/health")
	goUp := doctorHTTPReachable(goHost + "/api/tags")
	legacyGoUp := false
	if goHost == "http://127.0.0.1:11434" {
		legacyGoUp = doctorHTTPReachable("http://127.0.0.1:8080/api/tags")
	}

	var parts []string
	if goUp {
		parts = append(parts, "Go API "+goHost)
	} else if legacyGoUp {
		parts = append(parts, "Go API http://127.0.0.1:8080")
	}
	if sidecarUp {
		parts = append(parts, "runtime sidecar "+sidecarURL)
	}
	detail := "none detected"
	if len(parts) > 0 {
		detail = strings.Join(parts, "; ")
	}

	status := "ok"
	var fixes []string
	switch {
	case !goUp && !legacyGoUp && !sidecarUp:
		status = "warn"
		fixes = append(fixes, "run zerollama serve (Mac auto-starts sidecar on :8081)")
	case goUp && sidecarUp && strings.Contains(goHost, ":11434"):
		detail += " (Mac default: Go :11434 + sidecar :8081)"
	case legacyGoUp && sidecarUp && !goUp:
		detail += " (CI/smoke layout: Go :8080 + sidecar :8081)"
	case (goUp || legacyGoUp) && !sidecarUp:
		status = "warn"
		if runtime.GOOS == "darwin" {
			fixes = append(fixes, "runtime sidecar missing — zerollama serve auto-starts :8081 on Mac")
		} else {
			fixes = append(fixes, "start runtime sidecar or set ZEROLLAMA_RUNTIME_URL")
		}
	case sidecarUp && !goUp && !legacyGoUp:
		status = "warn"
		fixes = append(fixes, "sidecar without Go proxy — run zerollama serve")
	}
	return doctorCheck{
		Name:    "serve mode",
		Status:  status,
		Detail:  detail,
		FixHint: strings.Join(fixes, "; "),
	}
}

func doctorCheckTrainingVenv(repo string) doctorCheck {
	py := filepath.Join(repo, ".venv-training", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		return doctorCheck{
			Name:    "training/.venv-training",
			Status:  "warn",
			Detail:  "missing (optional for /api/train MPS LoRA)",
			FixHint: "./scripts/training_uv_venv.sh --verify or MAC_SETUP_TRAINING=1 ./scripts/mac_setup.sh",
		}
	}
	cmd := exec.Command(py, "-c", "import torch, peft")
	if err := cmd.Run(); err != nil {
		return doctorCheck{
			Name:    "training/.venv-training",
			Status:  "warn",
			Detail:  "torch/peft import failed",
			FixHint: "TRAINING_UV_SYNC=1 ./scripts/training_uv_venv.sh --verify",
		}
	}
	return doctorCheck{
		Name:   "training/.venv-training",
		Status: "ok",
		Detail: py,
	}
}

func doctorCheckSidecarHealth() doctorCheck {
	url := doctorRuntimeURL() + "/health"
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return doctorCheck{
			Name:    "runtime sidecar",
			Status:  "warn",
			Detail:  fmt.Sprintf("%s unreachable", url),
			FixHint: "zerollama serve (Mac auto-starts sidecar) or ./scripts/serve_mac_runtime.sh for CI",
		}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return doctorCheck{
			Name:    "runtime sidecar",
			Status:  "warn",
			Detail:  fmt.Sprintf("%s returned HTTP %d", url, resp.StatusCode),
			FixHint: "zerollama serve (Mac auto-starts sidecar) or ./scripts/serve_mac_runtime.sh for CI",
		}
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return doctorCheck{
			Name:   "runtime sidecar",
			Status: "warn",
			Detail: "could not read /health",
		}
	}
	var h map[string]any
	if err := json.Unmarshal(body, &h); err != nil {
		return doctorCheck{
			Name:   "runtime sidecar",
			Status: "warn",
			Detail: "invalid /health JSON",
		}
	}
	return doctorEvaluateSidecarHealth(h)
}

func doctorEvaluateSidecarHealth(h map[string]any) doctorCheck {
	pick := ""
	if ac, ok := h["autoconfig"].(map[string]any); ok {
		pick, _ = ac["pick"].(string)
	}
	backend, _ := h["llama_backend"].(string)
	source, _ := h["llama_backend_source"].(string)
	probe, _ := h["vram_probe_effective"].(string)
	fallback, _ := h["llama_backend_fallback"].(bool)
	requested, _ := h["llama_backend_requested"].(string)
	detail := fmt.Sprintf(
		"pick=%s backend=%s source=%s probe=%s requested=%s fallback=%v",
		pick, backend, source, probe, requested, fallback,
	)
	status := "ok"
	var fixes []string
	if pick != "" && pick != "apple_silicon" && pick != "custom" {
		status = "warn"
		fixes = append(fixes, "unset ZEROLLAMA_RUNTIME_CONFIG or use apple_silicon.yaml autoconfig")
	}
	if source == "env" && backend != "inprocess" {
		status = "warn"
		fixes = append(fixes, "unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND to use apple_silicon.yaml inprocess default")
	}
	if backend == "inprocess" && source == "default" {
		status = "warn"
		fixes = append(fixes, "inprocess without yaml/env — load apple_silicon.yaml or set ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess")
	}
	if fallback {
		status = "warn"
		fixes = append(fixes, "inprocess load failed; use a text-only GGUF or check libllama (see /health llama_backend_fallback)")
	}
	return doctorCheck{
		Name:    "runtime sidecar",
		Status:  status,
		Detail:  detail,
		FixHint: strings.Join(fixes, "; "),
	}
}

func doctorCheckTextGGUF(repo string) doctorCheck {
	py := filepath.Join(repo, "runtime", ".venv", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		py = "python3"
	}
	cmd := exec.Command(py, "-c", doctorPickTextGGUFSnippet())
	out, err := cmd.Output()
	if err != nil {
		return doctorCheck{
			Name:    "text GGUF model",
			Status:  "warn",
			Detail:  "could not scan ~/.ollama/models",
			FixHint: "zerollama pull a small text model; set M3_LLAMA_MODEL for sign-off",
		}
	}
	line := strings.TrimSpace(string(out))
	if line == "" {
		return doctorCheck{
			Name:    "text GGUF model",
			Status:  "warn",
			Detail:  "no local text GGUF found",
			FixHint: "zerollama pull llama3.2:3b (or similar text-only model)",
		}
	}
	return doctorCheck{
		Name:   "text GGUF model",
		Status: "ok",
		Detail: line,
	}
}

func doctorPickTextGGUFSnippet() string {
	return `
import json
from pathlib import Path
root = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library"
best = None
for mf in sorted(root.rglob("latest")):
    try:
        m = json.loads(mf.read_text())
        if any("projector" in (layer.get("mediaType") or "") for layer in m.get("layers", [])):
            continue
        cfg_path = Path.home() / ".ollama/models/blobs" / m["config"]["digest"].replace("sha256:", "sha256-")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        fam = (cfg.get("model_family") or "").lower()
        if fam in ("nomic-bert", "bert", "embed"):
            continue
        if "gemma" in fam and cfg.get("model_type") not in (None, "", "llama"):
            continue
        for layer in m.get("layers", []):
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue
            d = layer["digest"].replace("sha256:", "sha256-")
            path = Path.home() / ".ollama/models/blobs" / d
            size = int(layer.get("size") or 0)
            if path.is_file() and (best is None or size < best[0]):
                best = (size, str(path), mf.parent.name)
            break
    except Exception:
        pass
if best:
    print(f"{best[2]} -> {best[1]}")
`
}
