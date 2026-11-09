package qwen3asr

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// ErrUnavailable means the resident process could not start. Callers use Whisper.
var ErrUnavailable = errors.New("qwen3-asr unavailable")

// ErrBusy means one clip already holds the GPU. Callers wait and retry.
// It is not a missing install.
var ErrBusy = errors.New("qwen3-asr busy")

// wedgeHealthWait is how long /health may take before a live process is
// treated as stuck. A healthy process answers on loopback in milliseconds.
const wedgeHealthWait = time.Second

// wedgeBusy is how long one clip may hold the GPU before the process is
// restarted. Speculative guesses pile up behind a call that never returns.
const wedgeBusy = 15 * time.Second

// Server is one python3 -m runtime.asr.qwen3_serve process.
type Server struct {
	mu        sync.Mutex
	cmd       *exec.Cmd
	baseURL   string
	client    *http.Client
	health    *http.Client
	ready     chan struct{}
	done      chan struct{}
	startErr  error
	retired   bool
	everReady bool
	gen       int
	reviveMu  sync.Mutex
	errMu     sync.Mutex
	errBuf    bytes.Buffer
	modelDir  string
	device    string
	python    string
	pyPath    string
}

// StartServer launches the serve module on a free loopback port.
func StartServer(ctx context.Context, modelDir string) (*Server, error) {
	_ = ctx // caller Wait uses the request context; load health polling is detached.
	modelDir = strings.TrimSpace(modelDir)
	if modelDir == "" {
		return nil, fmt.Errorf("%w: model directory empty", ErrUnavailable)
	}
	st, err := os.Stat(modelDir)
	if err != nil || !st.IsDir() {
		return nil, fmt.Errorf("%w: model directory missing", ErrUnavailable)
	}
	python, err := lookPython()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	pyPath := runtimePythonPath()
	if pyPath == "" {
		return nil, fmt.Errorf("%w: runtime package not found (set ZEROLLAMA_REPO)", ErrUnavailable)
	}
	s := &Server{
		client:   &http.Client{Timeout: 10 * time.Minute},
		health:   &http.Client{Timeout: 200 * time.Millisecond},
		modelDir: modelDir,
		device:   envconfig.Qwen3ASRDevice(),
		python:   python,
		pyPath:   pyPath,
	}
	if err := s.spawn(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	return s, nil
}

func lookPython() (string, error) {
	if root := findRepoRoot(); root != "" {
		for _, rel := range []string{
			filepath.Join("runtime", ".venv", "bin", "python3"),
			filepath.Join("runtime", ".venv", "bin", "python"),
		} {
			p := filepath.Join(root, rel)
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p, nil
			}
		}
	}
	for _, name := range []string{"python3", "python"} {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("python3 not found on PATH")
}

func runtimePythonPath() string {
	if root := findRepoRoot(); root != "" {
		p := filepath.Join(root, "runtime")
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			if _, err := os.Stat(filepath.Join(p, "runtime", "asr", "qwen3_serve.py")); err == nil {
				return p
			}
		}
	}
	return ""
}

// findRepoRoot looks for the checkout that contains runtime/runtime/asr/qwen3_serve.py.
func findRepoRoot() string {
	for _, key := range []string{"ZEROLLAMA_REPO", "OLLAMA_TRAINING_PYTHONPATH"} {
		root := strings.TrimSpace(envconfig.Var(key))
		if root == "" {
			continue
		}
		if isRepoRoot(root) {
			return root
		}
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		if isRepoRoot(dir) {
			return dir
		}
		for _, rel := range []string{"..", "../..", "../../..", "../../../.."} {
			cand := filepath.Clean(filepath.Join(dir, rel))
			if isRepoRoot(cand) {
				return cand
			}
		}
	}
	if wd, err := os.Getwd(); err == nil {
		for dir := wd; ; dir = filepath.Dir(dir) {
			if isRepoRoot(dir) {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		for _, name := range []string{"zerollama", "Sites/inference/zerollama"} {
			cand := filepath.Join(home, name)
			if isRepoRoot(cand) {
				return cand
			}
		}
	}
	return ""
}

func isRepoRoot(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "runtime", "runtime", "asr", "qwen3_serve.py"))
	return err == nil
}

func (s *Server) spawn() error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	args := []string{
		"-m", "runtime.asr.qwen3_serve",
		"--model", s.modelDir,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--device", s.device,
	}
	cmd := exec.Command(s.python, args...)
	env := filterEnv(os.Environ(), "PYTHONPATH", "CUDA_VISIBLE_DEVICES")
	env = append(env, "PYTHONPATH="+s.pyPath)
	if strings.EqualFold(s.device, "cpu") || s.device == "" {
		env = append(env, "CUDA_VISIBLE_DEVICES=")
	}
	cmd.Env = env
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	cmd.Stdout = io.Discard
	if err := cmd.Start(); err != nil {
		return err
	}

	ready := make(chan struct{})
	done := make(chan struct{})
	s.mu.Lock()
	s.gen++
	gen := s.gen
	s.cmd = cmd
	s.baseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	s.ready = ready
	s.done = done
	s.startErr = nil
	s.errMu.Lock()
	s.errBuf.Reset()
	s.errMu.Unlock()
	s.mu.Unlock()

	go io.Copy(&limitWriter{dst: &s.errBuf, mu: &s.errMu, max: 4096}, stderr)
	// Load is independent of the caller request so a barge-in does not SIGTERM
	// a cold start and then fall back to Whisper.
	go s.waitReady(ready, gen)
	go s.watchExit(cmd, ready, done, gen)
	return nil
}

func filterEnv(env []string, drop ...string) []string {
	skip := make(map[string]struct{}, len(drop))
	for _, k := range drop {
		skip[k] = struct{}{}
	}
	out := make([]string, 0, len(env))
	for _, e := range env {
		key, _, ok := strings.Cut(e, "=")
		if ok {
			if _, drop := skip[key]; drop {
				continue
			}
		}
		out = append(out, e)
	}
	return out
}

func (s *Server) watchExit(cmd *exec.Cmd, ready, done chan struct{}, gen int) {
	err := cmd.Wait()
	s.mu.Lock()
	if s.gen == gen {
		if s.cmd == cmd {
			s.cmd = nil
		}
		select {
		case <-ready:
			if s.startErr == nil && !s.retired {
				s.startErr = fmt.Errorf("qwen3-asr exited: %v", err)
			}
		default:
			if s.startErr == nil {
				if err != nil {
					s.startErr = fmt.Errorf("qwen3-asr exited before ready: %w: %s", err, s.stderr())
				} else {
					s.startErr = fmt.Errorf("qwen3-asr exited before ready: %s", s.stderr())
				}
			}
			close(ready)
		}
	}
	s.mu.Unlock()
	close(done)
}

func (s *Server) waitReady(ready chan struct{}, gen int) {
	defer func() {
		s.mu.Lock()
		if s.gen != gen {
			s.mu.Unlock()
			return
		}
		select {
		case <-ready:
		default:
			close(ready)
		}
		failed := s.startErr != nil
		cmd := s.cmd
		s.mu.Unlock()
		if failed {
			s.signal(cmd)
		}
	}()
	deadline := time.Now().Add(10 * time.Minute)
	for {
		s.mu.Lock()
		if s.gen != gen {
			s.mu.Unlock()
			return
		}
		base := s.baseURL
		client := s.health
		s.mu.Unlock()

		if time.Now().After(deadline) {
			s.failStart(ready, gen, fmt.Errorf("qwen3-asr health timed out: %s", s.stderr()))
			return
		}
		reqCtx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, base+"/health", nil)
		if err != nil {
			cancel()
			s.failStart(ready, gen, err)
			return
		}
		res, err := client.Do(req)
		cancel()
		if err == nil {
			body, _ := io.ReadAll(io.LimitReader(res.Body, 256))
			_ = res.Body.Close()
			if res.StatusCode == http.StatusOK && healthOK(body) {
				s.mu.Lock()
				if s.gen == gen {
					s.everReady = true
					select {
					case <-ready:
					default:
						close(ready)
					}
				}
				s.mu.Unlock()
				return
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func (s *Server) failStart(ready chan struct{}, gen int, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.gen != gen {
		return
	}
	if s.startErr == nil {
		s.startErr = err
	}
	select {
	case <-ready:
	default:
		close(ready)
	}
}

func healthOK(body []byte) bool {
	ok, _ := parseHealth(body)
	return ok
}

func parseHealth(body []byte) (ok bool, busy time.Duration) {
	var parsed struct {
		Status string `json:"status"`
		BusyMS int64  `json:"busy_ms"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return false, 0
	}
	if parsed.Status != "ok" {
		return false, 0
	}
	if parsed.BusyMS > 0 {
		busy = time.Duration(parsed.BusyMS) * time.Millisecond
	}
	return true, busy
}

// Wait blocks until the model is ready or the start failed.
func (s *Server) Wait(ctx context.Context) error {
	if s == nil {
		return ErrUnavailable
	}
	s.mu.Lock()
	ready := s.ready
	s.mu.Unlock()
	if ready == nil {
		return ErrUnavailable
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-ready:
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.retired {
			return fmt.Errorf("qwen3-asr closed")
		}
		if s.startErr != nil {
			if !s.everReady {
				return fmt.Errorf("%w: %v", ErrUnavailable, s.startErr)
			}
			return s.startErr
		}
		return nil
	}
}

// Ready is true when /health answered and the process is still up.
func (s *Server) Ready() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	ready := s.ready
	s.mu.Unlock()
	if ready == nil {
		return false
	}
	select {
	case <-ready:
	default:
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.startErr == nil && !s.retired && s.cmd != nil && s.everReady
}

// Loading is true while the child is still starting and has not failed.
func (s *Server) Loading() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.retired || s.cmd == nil {
		return false
	}
	if s.everReady {
		return false
	}
	ready := s.ready
	if ready == nil {
		return true
	}
	select {
	case <-ready:
		return s.startErr == nil
	default:
		return true
	}
}

// Transcribe posts one WAV. language is a canonical name such as English.
func (s *Server) Transcribe(ctx context.Context, wav []byte, language string) (string, error) {
	if s == nil {
		return "", ErrUnavailable
	}
	if err := s.Wait(ctx); err != nil {
		return "", err
	}
	s.mu.Lock()
	base := s.baseURL
	client := s.client
	alive := s.cmd != nil && !s.retired
	s.mu.Unlock()
	if !alive {
		return "", fmt.Errorf("qwen3-asr is not running")
	}
	if err := s.reviveIfWedged(ctx); err != nil {
		return "", err
	}
	s.mu.Lock()
	base = s.baseURL
	client = s.client
	s.mu.Unlock()
	return PostTranscribe(ctx, client, base, wav, language)
}

// reviveIfWedged restarts a process that is still running but no longer
// answering, or that has held the GPU longer than wedgeBusy. A dead process
// is left for ensure to replace.
func (s *Server) reviveIfWedged(ctx context.Context) error {
	if !s.Ready() {
		return nil
	}
	s.reviveMu.Lock()
	defer s.reviveMu.Unlock()
	if !s.Ready() {
		return nil
	}
	ok, busy := s.probeHealth(wedgeHealthWait)
	if ok && busy < wedgeBusy {
		return nil
	}
	slog.Warn("qwen3-asr wedged; restarting", "health_ok", ok, "busy", busy.String())
	return s.Restart(ctx)
}

// probeHealth calls /health. ok is false when the process does not answer.
func (s *Server) probeHealth(wait time.Duration) (ok bool, busy time.Duration) {
	s.mu.Lock()
	base := s.baseURL
	s.mu.Unlock()
	if base == "" || wait <= 0 {
		return false, 0
	}
	ctx, cancel := context.WithTimeout(context.Background(), wait)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/health", nil)
	if err != nil {
		return false, 0
	}
	client := &http.Client{Timeout: wait}
	res, err := client.Do(req)
	if err != nil {
		return false, 0
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 256))
	if err != nil || res.StatusCode != http.StatusOK {
		return false, 0
	}
	return parseHealth(body)
}

// Restart stops the current process and starts another on a new port.
func (s *Server) Restart(ctx context.Context) error {
	if s == nil {
		return fmt.Errorf("qwen3-asr is not running")
	}
	s.mu.Lock()
	if s.retired {
		s.mu.Unlock()
		return fmt.Errorf("qwen3-asr closed")
	}
	s.everReady = false
	s.mu.Unlock()
	s.kill()
	if err := s.spawn(); err != nil {
		return err
	}
	return s.Wait(ctx)
}

// Close stops the process.
func (s *Server) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.retired = true
	if s.startErr == nil {
		s.startErr = fmt.Errorf("qwen3-asr closed")
	}
	ready := s.ready
	if ready != nil {
		select {
		case <-ready:
		default:
			close(ready)
		}
	}
	s.mu.Unlock()
	s.kill()
}

func (s *Server) kill() {
	s.mu.Lock()
	cmd := s.cmd
	done := s.done
	s.cmd = nil
	s.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)
	if done != nil {
		select {
		case <-done:
			return
		case <-time.After(2 * time.Second):
		}
	}
	_ = cmd.Process.Kill()
	if done != nil {
		<-done
	}
}

func (s *Server) signal(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)
}

func (s *Server) stderr() string {
	s.errMu.Lock()
	defer s.errMu.Unlock()
	return strings.TrimSpace(s.errBuf.String())
}

// PostTranscribe posts multipart file + optional language to /transcribe.
func PostTranscribe(ctx context.Context, client *http.Client, baseURL string, wav []byte, language string) (string, error) {
	if client == nil {
		client = http.DefaultClient
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	part, err := w.CreateFormFile("file", "clip.wav")
	if err != nil {
		return "", err
	}
	if _, err := part.Write(wav); err != nil {
		return "", err
	}
	if language = strings.TrimSpace(language); language != "" {
		if err := w.WriteField("language", language); err != nil {
			return "", err
		}
	}
	if err := w.Close(); err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+"/transcribe", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	res, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return "", err
	}
	if res.StatusCode == http.StatusServiceUnavailable {
		return "", ErrUnavailable
	}
	if res.StatusCode == http.StatusTooManyRequests {
		return "", ErrBusy
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", fmt.Errorf("qwen3-asr HTTP %d: %s", res.StatusCode, strings.TrimSpace(string(body)))
	}
	var parsed struct {
		Text  string `json:"text"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("qwen3-asr response: %w", err)
	}
	if parsed.Error != "" && parsed.Text == "" {
		return "", fmt.Errorf("qwen3-asr: %s", parsed.Error)
	}
	return strings.TrimSpace(parsed.Text), nil
}

type limitWriter struct {
	dst *bytes.Buffer
	mu  *sync.Mutex
	max int
}

// Write stores up to max bytes and always reports the full input consumed.
// A short count makes io.Copy stop reading. The child's stderr pipe then
// fills and the next request, including /health, blocks in its access log.
func (w *limitWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.dst.Len() < w.max {
		remain := w.max - w.dst.Len()
		chunk := p
		if len(chunk) > remain {
			chunk = chunk[:remain]
		}
		_, _ = w.dst.Write(chunk)
	}
	return len(p), nil
}
