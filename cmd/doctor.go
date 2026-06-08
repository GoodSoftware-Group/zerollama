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
	Name    string
	Status  string // ok, warn, fail
	Detail  string
	FixHint string
}

func NewDoctorCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "doctor",
		Short: "Check local zerollama / Apple Silicon runtime readiness",
		Long:  "Validate uv venv, Metal libllama, sidecar health, and autoconfig on Darwin.",
		RunE:  runDoctor,
	}
}

func runDoctor(_ *cobra.Command, _ []string) error {
	checks := runDoctorChecks()
	failures := 0
	warns := 0
	for _, c := range checks {
		switch c.Status {
		case "fail":
			failures++
		case "warn":
			warns++
		}
		fmt.Printf("[%s] %s\n", c.Status, c.Name)
		if c.Detail != "" {
			fmt.Printf("      %s\n", c.Detail)
		}
		if c.FixHint != "" && c.Status != "ok" {
			fmt.Printf("      → %s\n", c.FixHint)
		}
	}
	fmt.Println()
	if failures > 0 {
		return fmt.Errorf("doctor: %d check(s) failed, %d warning(s)", failures, warns)
	}
	if warns > 0 {
		fmt.Printf("doctor: ok with %d warning(s)\n", warns)
		return nil
	}
	fmt.Println("doctor: all checks passed")
	return nil
}

func runDoctorChecks() []doctorCheck {
	var out []doctorCheck
	repo := doctorRepoRoot()

	out = append(out, doctorCheckGo())
	out = append(out, doctorCheckZerollamaBinary(repo))
	out = append(out, doctorCheckUV())
	out = append(out, doctorCheckRuntimeVenv(repo))
	out = append(out, doctorCheckLibLlama(repo))

	if runtime.GOOS == "darwin" {
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
			FixHint: "rebuild with go build -o zerollama .; on Mac prefer sidecar (./scripts/serve_mac_runtime.sh)",
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
			FixHint: "./scripts/runtime_uv_venv.sh",
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

func doctorCheckLibLlama(repo string) doctorCheck {
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
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && st.Size() > 0 {
			return doctorCheck{
				Name:   "libllama",
				Status: "ok",
				Detail: p,
			}
		}
	}
	return doctorCheck{
		Name:    "libllama",
		Status:  "fail",
		Detail:  "Metal/CUDA libllama not found",
		FixHint: "LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh",
	}
}

func doctorRuntimeURL() string {
	if u := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_URL")); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return "http://127.0.0.1:8081"
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
			FixHint: "./scripts/serve_mac_runtime.sh",
		}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return doctorCheck{
			Name:    "runtime sidecar",
			Status:  "warn",
			Detail:  fmt.Sprintf("%s returned HTTP %d", url, resp.StatusCode),
			FixHint: "./scripts/serve_mac_runtime.sh",
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
