// Darwin serve bootstrap: uv sidecar on loopback, venv ensure, autoconfig defaults.
// On macOS, embedded CPython in the Go binary is tied to system Python 3.9 — too old for
// the runtime stack. zerollama serve spawns runtime/.venv as a child on :8081 instead.

package server

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/trainingworker"
)

const (
	darwinVenvTimeout   = 10 * time.Minute
	darwinHealthTimeout = 45 * time.Second
)

// DarwinSidecar manages a child runtime sidecar started by zerollama serve on macOS.
type DarwinSidecar struct {
	cmd     *exec.Cmd
	stopOnce sync.Once
}

// Stop terminates a child sidecar process started by BootstrapDarwinSidecar when
// ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=managed. Default (persist) leaves the sidecar running
// so the next zerollama serve can reuse it without a cold Python startup.
func (s *DarwinSidecar) Stop() {
	if s == nil || !envconfig.DarwinSidecarKillOnServeExit() {
		return
	}
	s.stopOnce.Do(func() {
		if s.cmd == nil || s.cmd.Process == nil {
			return
		}
		_ = s.cmd.Process.Kill()
		_, _ = s.cmd.Process.Wait()
	})
}

// BootstrapDarwinSidecar ensures uv venvs, sets Darwin defaults, and starts or reuses
// a loopback runtime sidecar. Returns (nil, nil) when bootstrap does not apply.
func BootstrapDarwinSidecar(ctx context.Context) (*DarwinSidecar, error) {
	if !darwinSidecarEnabled() {
		if runtime.GOOS == "darwin" {
			slog.Info("darwin runtime sidecar bootstrap skipped", "reason", darwinSidecarSkipReason())
		}
		return nil, nil
	}

	slog.Info("darwin runtime sidecar bootstrap starting")

	repoRoot, err := trainingworker.RepoRoot()
	if err != nil || repoRoot == "" {
		return nil, fmt.Errorf("darwin sidecar: repo root: %w", err)
	}
	slog.Info("darwin sidecar: repo root", "path", repoRoot)

	applyDarwinServeDefaults(repoRoot)

	venvCtx, venvCancel := context.WithTimeout(ctx, darwinVenvTimeout)
	defer venvCancel()
	slog.Info("darwin sidecar: ensuring runtime/.venv (uv; first run can take several minutes)")
	if err := ensureDarwinRuntimeVenv(venvCtx, repoRoot); err != nil {
		return nil, fmt.Errorf("darwin sidecar: runtime venv: %w", err)
	}
	slog.Info("darwin sidecar: runtime/.venv ready")

	host, port := darwinSidecarListen()
	baseURL := fmt.Sprintf("http://%s:%d", host, port)
	_ = os.Setenv("ZEROLLAMA_RUNTIME_URL", baseURL)

	slog.Info("darwin sidecar: checking runtime health", "url", baseURL)
	probeCtx, probeCancel := context.WithTimeout(ctx, 3*time.Second)
	if waitRuntimeHealth(probeCtx, baseURL, 2*time.Second) == nil {
		probeCancel()
		slog.Info("darwin runtime sidecar already listening", "url", baseURL)
		return &DarwinSidecar{}, nil
	}
	probeCancel()

	py := filepath.Join(repoRoot, "runtime", ".venv", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		return nil, fmt.Errorf("darwin sidecar: %s missing after venv ensure", py)
	}

	logPath := strings.TrimSpace(os.Getenv("MACOS_RT_LOG"))
	if logPath == "" {
		logPath = filepath.Join(os.TempDir(), "zerollama-runtime-sidecar.log")
	}
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("darwin sidecar: log %s: %w", logPath, err)
	}

	cmd := exec.Command(py, "-m", "runtime", "serve", "--host", host, "--port", strconv.Itoa(port))
	cmd.Dir = filepath.Join(repoRoot, "runtime")
	cmd.Env = darwinSubprocessEnv()
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	slog.Info("darwin sidecar: starting python runtime", "log", logPath, "port", port)
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return nil, fmt.Errorf("darwin sidecar: start: %w", err)
	}
	_ = logFile.Close()

	healthCtx, healthCancel := context.WithTimeout(ctx, darwinHealthTimeout)
	defer healthCancel()
	if err := waitRuntimeHealth(healthCtx, baseURL, 2*time.Second); err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return nil, fmt.Errorf("darwin sidecar: health on %s (see %s): %w", baseURL, logPath, err)
	}

	slog.Info("darwin runtime sidecar started", "url", baseURL, "log", logPath)
	return &DarwinSidecar{cmd: cmd}, nil
}

// EnsureDarwinTrainingEnv prepares .venv-training PYTHONPATH for embedded training on Darwin.
// Called from Serve before trainingworker.Start (not sidecar bootstrap).
func EnsureDarwinTrainingEnv(ctx context.Context, repoRoot string) error {
	venvCtx, cancel := context.WithTimeout(ctx, darwinVenvTimeout)
	defer cancel()
	return ensureDarwinTrainingVenv(venvCtx, repoRoot)
}

func darwinSidecarEnabled() bool {
	if runtime.GOOS != "darwin" {
		return false
	}
	return darwinSidecarEnvEnabled()
}

func darwinSidecarEnvEnabled() bool {
	v := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_URL")) != "" {
		return false
	}
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME")); v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	return true
}

func darwinSidecarSkipReason() string {
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR")); v == "0" || strings.EqualFold(v, "false") {
		return "ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0"
	}
	if u := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_URL")); u != "" {
		return "ZEROLLAMA_RUNTIME_URL already set (external sidecar expected): " + u
	}
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME")); v == "0" || strings.EqualFold(v, "false") {
		return "ZEROLLAMA_RUNTIME=0"
	}
	return "unknown"
}

func applyDarwinServeDefaults(repoRoot string) {
	if strings.TrimSpace(os.Getenv("ZEROLLAMA_REPO")) == "" {
		_ = os.Setenv("ZEROLLAMA_REPO", repoRoot)
	}
	if strings.TrimSpace(os.Getenv("OLLAMA_TRAINING_PYTHONPATH")) == "" {
		_ = os.Setenv("OLLAMA_TRAINING_PYTHONPATH", repoRoot)
	}
	if strings.TrimSpace(os.Getenv("ZEROLLAMA_AUTO_CONFIG")) == "" &&
		strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_CONFIG")) == "" {
		_ = os.Setenv("ZEROLLAMA_AUTO_CONFIG", "1")
	}
	// Do not set ZEROLLAMA_RUNTIME=1 here: sidecar stays up for tokenize/VRAM probes,
	// but plain GGUF chat stays on ggml Metal unless the operator opts in.
	if !envconfig.LlamaCppBackend() {
		_ = os.Unsetenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND")
	}
	envconfig.ApplyLlamaCppBackendDefaults()
	applyDarwinLlamaCppEnv(repoRoot)
	if !envconfig.LlamaCppBackend() {
		lib := strings.TrimSpace(os.Getenv("LLAMA_CPP_LIB"))
		if lib != "" {
			if st, err := os.Stat(lib); err == nil && !st.IsDir() {
				_ = os.Setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "inprocess")
			}
		}
	}
}

func applyDarwinLlamaCppEnv(repoRoot string) {
	if strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT")) != "" {
		return
	}
	llamaRoot := filepath.Clean(filepath.Join(repoRoot, "..", "llama.cpp"))
	if _, err := os.Stat(filepath.Join(llamaRoot, "CMakeLists.txt")); err != nil {
		return
	}
	_ = os.Setenv("LLAMA_CPP_ROOT", llamaRoot)
	lib := filepath.Join(llamaRoot, "build", "bin", "libllama.dylib")
	if _, err := os.Stat(lib); err == nil && strings.TrimSpace(os.Getenv("LLAMA_CPP_LIB")) == "" {
		_ = os.Setenv("LLAMA_CPP_LIB", lib)
	}
	serverBin := filepath.Join(llamaRoot, "build", "bin", "llama-server")
	if _, err := os.Stat(serverBin); err == nil && strings.TrimSpace(os.Getenv("LLAMA_SERVER_BIN")) == "" {
		_ = os.Setenv("LLAMA_SERVER_BIN", serverBin)
	}
}

func darwinSidecarListen() (host string, port int) {
	host = "127.0.0.1"
	port = envconfig.RuntimeEmbedPort()
	if u := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_URL")); u != "" {
		if parsed, err := url.Parse(u); err == nil && parsed.Host != "" {
			h, p, err := net.SplitHostPort(parsed.Host)
			if err == nil {
				if h != "" {
					host = h
				}
				if n, err := strconv.Atoi(p); err == nil && n > 0 {
					port = n
				}
			}
		}
	}
	return host, port
}

func ensureDarwinRuntimeVenv(ctx context.Context, repoRoot string) error {
	py := filepath.Join(repoRoot, "runtime", ".venv", "bin", "python")
	if darwinVenvImportOK(ctx, py, "fastapi") {
		slog.Info("darwin sidecar: runtime/.venv already ready", "python", py)
		return nil
	}
	return runRepoBash(ctx, repoRoot, "source scripts/runtime_uv_venv.sh && runtime_uv_venv")
}

func darwinVenvImportOK(ctx context.Context, py, module string) bool {
	if py == "" {
		return false
	}
	if _, err := os.Stat(py); err != nil {
		return false
	}
	cmd := exec.CommandContext(ctx, py, "-c", "import "+module)
	cmd.Env = darwinSubprocessEnv()
	return cmd.Run() == nil
}

func darwinSubprocessEnv() []string {
	path := os.Getenv("PATH")
	for _, dir := range []string{
		filepath.Join(os.Getenv("HOME"), ".local", "bin"),
		"/opt/homebrew/bin",
		"/usr/local/bin",
	} {
		if dir == "" {
			continue
		}
		if !strings.Contains(path+":", dir+":") {
			path = dir + ":" + path
		}
	}
	env := os.Environ()
	set := false
	for i, e := range env {
		if strings.HasPrefix(e, "PATH=") {
			env[i] = "PATH=" + path
			set = true
			break
		}
	}
	if !set {
		env = append(env, "PATH="+path)
	}
	return env
}

func ensureDarwinTrainingVenv(ctx context.Context, repoRoot string) error {
	py := filepath.Join(repoRoot, ".venv-training", "bin", "python")
	if darwinVenvImportOK(ctx, py, "torch") && darwinVenvImportOK(ctx, py, "peft") {
		slog.Info("darwin training: .venv-training already ready", "python", py)
		return applyDarwinTrainingPYTHONPATH(ctx, repoRoot)
	}
	if err := runRepoBash(ctx, repoRoot, "source scripts/training_uv_venv.sh && training_uv_venv"); err != nil {
		return err
	}
	return applyDarwinTrainingPYTHONPATH(ctx, repoRoot)
}

func applyDarwinTrainingPYTHONPATH(ctx context.Context, repoRoot string) error {
	py := filepath.Join(repoRoot, ".venv-training", "bin", "python")
	cmd := exec.CommandContext(ctx, py, "-c", "import site; print(site.getsitepackages()[0])")
	cmd.Env = darwinSubprocessEnv()
	out, err := cmd.Output()
	if err != nil {
		return err
	}
	site := strings.TrimSpace(string(out))
	if site != "" {
		_ = os.Setenv("PYTHONPATH", site)
	}
	return nil
}

func runRepoBash(ctx context.Context, repoRoot, script string) error {
	cmd := exec.CommandContext(ctx, "bash", "-c", script)
	cmd.Dir = repoRoot
	cmd.Env = darwinSubprocessEnv()
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func waitRuntimeHealth(ctx context.Context, baseURL string, perReq time.Duration) error {
	client := &http.Client{Timeout: perReq}
	url := strings.TrimSuffix(baseURL, "/") + "/health"
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			if err != nil {
				return fmt.Errorf("%w: last error: %v", ctx.Err(), err)
			}
			return ctx.Err()
		case <-ticker.C:
		}
	}
}
