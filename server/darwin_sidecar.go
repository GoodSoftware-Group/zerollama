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

// Stop terminates a sidecar process started by BootstrapDarwinSidecar.
func (s *DarwinSidecar) Stop() {
	if s == nil {
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
		return nil, nil
	}

	repoRoot, err := trainingworker.RepoRoot()
	if err != nil || repoRoot == "" {
		return nil, fmt.Errorf("darwin sidecar: repo root: %w", err)
	}

	applyDarwinServeDefaults(repoRoot)

	venvCtx, venvCancel := context.WithTimeout(ctx, darwinVenvTimeout)
	defer venvCancel()
	if err := ensureDarwinRuntimeVenv(venvCtx, repoRoot); err != nil {
		return nil, fmt.Errorf("darwin sidecar: runtime venv: %w", err)
	}
	if envconfig.TrainingEnabled(true) {
		if err := ensureDarwinTrainingVenv(venvCtx, repoRoot); err != nil {
			slog.Warn("darwin training venv not ready (OLLAMA_TRAINING may fail)", "error", err)
		}
	}

	host, port := darwinSidecarListen()
	baseURL := fmt.Sprintf("http://%s:%d", host, port)
	_ = os.Setenv("ZEROLLAMA_RUNTIME_URL", baseURL)

	if waitRuntimeHealth(ctx, baseURL, 2*time.Second) == nil {
		slog.Info("darwin runtime sidecar already listening", "url", baseURL)
		return &DarwinSidecar{}, nil
	}

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
	cmd.Env = os.Environ()
	cmd.Stdout = logFile
	cmd.Stderr = logFile
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

func applyDarwinServeDefaults(repoRoot string) {
	if strings.TrimSpace(os.Getenv("ZEROLLAMA_AUTO_CONFIG")) == "" &&
		strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_CONFIG")) == "" {
		_ = os.Setenv("ZEROLLAMA_AUTO_CONFIG", "1")
	}
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME")); v == "" {
		_ = os.Setenv("ZEROLLAMA_RUNTIME", "1")
	}
	_ = os.Unsetenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND")
	applyDarwinLlamaCppEnv(repoRoot)
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
	return runRepoBash(ctx, repoRoot, "source scripts/runtime_uv_venv.sh && runtime_uv_venv")
}

func ensureDarwinTrainingVenv(ctx context.Context, repoRoot string) error {
	if strings.TrimSpace(os.Getenv("OLLAMA_TRAINING_PYTHONPATH")) == "" &&
		strings.TrimSpace(os.Getenv("ZEROLLAMA_REPO")) == "" {
		_ = os.Setenv("OLLAMA_TRAINING_PYTHONPATH", repoRoot)
	}
	cmd := exec.CommandContext(ctx, "bash", "-c",
		"source scripts/training_uv_venv.sh && training_uv_venv && printf '%s' \"$PYTHONPATH\"",
	)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		return err
	}
	if pyPath := strings.TrimSpace(string(out)); pyPath != "" {
		_ = os.Setenv("PYTHONPATH", pyPath)
	}
	return nil
}

func runRepoBash(ctx context.Context, repoRoot, script string) error {
	cmd := exec.CommandContext(ctx, "bash", "-c", script)
	cmd.Dir = repoRoot
	cmd.Stdout = io.Discard
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
