package neuralvad

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// ErrUnavailable means the process could not start. Callers stay on amplitude.
var ErrUnavailable = errors.New("neural vad unavailable")

// scoreTimeout is how long one chunk may wait. A miss uses amplitude.
const scoreTimeout = 40 * time.Millisecond

// Server is one python3 -m runtime.asr.vad_serve process.
type Server struct {
	mu       sync.Mutex
	cmd      *exec.Cmd
	baseURL  string
	client   *http.Client
	health   *http.Client
	ready    chan struct{}
	done     chan struct{}
	startErr error
	retired  bool
	gen      int
	errMu    sync.Mutex
	errBuf   bytes.Buffer
	model    string
	python   string
	pyPath   string
}

// Shared is the process used by every live socket.
type Shared struct {
	mu       sync.Mutex
	proc     *Server
	booting  bool
	bootDone chan struct{}
}

var defaultShared = &Shared{}

// NeuralVADPath is OLLAMA_NEURAL_VAD when that path is an existing file.
func NeuralVADPath() string {
	p := strings.TrimSpace(envconfig.NeuralVAD())
	if p == "" {
		return ""
	}
	st, err := os.Stat(p)
	if err != nil || st.IsDir() {
		return ""
	}
	return p
}

var sessionSeq uint64

// Attach starts the shared process when a local ONNX file is configured and
// returns a session id plus a Gate scorer. Nil score means amplitude.
func Attach() (sessionID string, score func(pcm []byte, rate int, threshold float64) (int, error)) {
	if !Available() {
		return "", nil
	}
	id := fmt.Sprintf("vad-%d", atomic.AddUint64(&sessionSeq, 1))
	sc := Scorer(id)
	if sc == nil {
		return "", nil
	}
	return id, sc
}

// Available is true when a local ONNX file is configured.
func Available() bool {
	return NeuralVADPath() != ""
}

// Score posts one PCM16 chunk. last is the last speech sample index, or -1.
func Score(ctx context.Context, session string, pcm []byte, rate int, threshold float64) (int, error) {
	return defaultShared.Score(ctx, session, pcm, rate, threshold)
}

// Reset clears recurrent state for session.
func Reset(session string) {
	defaultShared.Reset(session)
}

// Scorer returns a voicepipe-compatible function for session, or nil when
// no ONNX path is configured. Kick off the shared process in the background
// so Accept is not held for model load. A timed-out or failed score returns
// an error so the gate falls back to amplitude for that packet.
func Scorer(session string) func(pcm []byte, rate int, threshold float64) (int, error) {
	if !Available() {
		return nil
	}
	sid := strings.TrimSpace(session)
	if sid == "" {
		return nil
	}
	go func() {
		_, _ = defaultShared.ensure(context.Background())
	}()
	return func(pcm []byte, rate int, threshold float64) (int, error) {
		ctx, cancel := context.WithTimeout(context.Background(), scoreTimeout)
		defer cancel()
		return defaultShared.Score(ctx, sid, pcm, rate, threshold)
	}
}

// Score implements Shared.Score.
func (s *Shared) Score(ctx context.Context, session string, pcm []byte, rate int, threshold float64) (int, error) {
	if s == nil {
		return -1, ErrUnavailable
	}
	if rate != 8000 && rate != 16000 {
		return -1, fmt.Errorf("neural vad rate %d", rate)
	}
	proc, err := s.ensure(ctx)
	if err != nil {
		return -1, err
	}
	return proc.Score(ctx, session, pcm, rate, threshold)
}

// Reset implements Shared.Reset.
func (s *Shared) Reset(session string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	proc := s.proc
	s.mu.Unlock()
	if proc == nil || !proc.Ready() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), scoreTimeout)
	defer cancel()
	_ = proc.Reset(ctx, session)
}

func (s *Shared) ensure(ctx context.Context) (*Server, error) {
	for {
		s.mu.Lock()
		if s.booting {
			done := s.bootDone
			s.mu.Unlock()
			if done == nil {
				continue
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-done:
			}
			continue
		}
		if s.proc != nil {
			proc := s.proc
			s.mu.Unlock()
			if err := proc.Wait(ctx); err != nil {
				return nil, err
			}
			if !proc.Ready() {
				return nil, ErrUnavailable
			}
			return proc, nil
		}
		path := NeuralVADPath()
		if path == "" {
			s.mu.Unlock()
			return nil, ErrUnavailable
		}
		s.booting = true
		done := make(chan struct{})
		s.bootDone = done
		s.mu.Unlock()

		proc, err := StartServer(ctx, path)
		if err == nil {
			err = proc.Wait(ctx)
		}
		if err != nil {
			if proc != nil {
				// Keep a still-loading process when the caller timed out so
				// the next score can Wait for the same boot.
				if (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) && !proc.Ready() && proc.Loading() {
					s.mu.Lock()
					s.proc = proc
					s.booting = false
					close(done)
					s.bootDone = nil
					s.mu.Unlock()
					return nil, err
				}
				proc.Close()
			}
			s.mu.Lock()
			s.booting = false
			close(done)
			s.bootDone = nil
			s.mu.Unlock()
			return nil, err
		}

		s.mu.Lock()
		s.proc = proc
		s.booting = false
		close(done)
		s.bootDone = nil
		s.mu.Unlock()
		return proc, nil
	}
}

// StartServer launches the serve module on a free loopback port.
func StartServer(ctx context.Context, modelPath string) (*Server, error) {
	_ = ctx
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("%w: model empty", ErrUnavailable)
	}
	if st, err := os.Stat(modelPath); err != nil || st.IsDir() {
		return nil, fmt.Errorf("%w: model missing", ErrUnavailable)
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
		client: &http.Client{Timeout: 5 * time.Second},
		health: &http.Client{Timeout: 200 * time.Millisecond},
		model:  modelPath,
		python: python,
		pyPath: pyPath,
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
		if _, err := os.Stat(filepath.Join(p, "runtime", "asr", "vad_serve.py")); err == nil {
			return p
		}
	}
	return ""
}

func findRepoRoot() string {
	for _, key := range []string{"ZEROLLAMA_REPO", "OLLAMA_TRAINING_PYTHONPATH"} {
		root := strings.TrimSpace(envconfig.Var(key))
		if root != "" && isRepoRoot(root) {
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
	return ""
}

func isRepoRoot(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "runtime", "runtime", "asr", "vad_serve.py"))
	return err == nil
}

func (s *Server) spawn() error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	cmd := exec.Command(s.python, "-m", "runtime.asr.vad_serve",
		"--model", s.model,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
	)
	env := filterEnv(os.Environ(), "PYTHONPATH", "CUDA_VISIBLE_DEVICES")
	env = append(env, "PYTHONPATH="+s.pyPath, "CUDA_VISIBLE_DEVICES=")
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
	go s.waitReady(ready, gen)
	go s.watchExit(cmd, ready, done, gen)
	return nil
}

func filterEnv(env []string, drop ...string) []string {
	skip := map[string]struct{}{}
	for _, k := range drop {
		skip[k] = struct{}{}
	}
	out := make([]string, 0, len(env))
	for _, e := range env {
		key, _, ok := strings.Cut(e, "=")
		if ok {
			if _, d := skip[key]; d {
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
				s.startErr = fmt.Errorf("neural-vad exited: %v", err)
			}
		default:
			if s.startErr == nil {
				s.startErr = fmt.Errorf("neural-vad exited before ready: %v: %s", err, s.stderr())
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
		if failed && cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Signal(syscall.SIGTERM)
		}
	}()
	deadline := time.Now().Add(2 * time.Minute)
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
			s.failStart(ready, gen, fmt.Errorf("neural-vad health timed out: %s", s.stderr()))
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
	var parsed struct {
		Status string `json:"status"`
	}
	return json.Unmarshal(body, &parsed) == nil && parsed.Status == "ok"
}

// Wait blocks until ready or start failed.
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
			return ErrUnavailable
		}
		if s.startErr != nil {
			return fmt.Errorf("%w: %v", ErrUnavailable, s.startErr)
		}
		return nil
	}
}

// Ready is true when /health answered and the process is up.
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
	return s.startErr == nil && !s.retired && s.cmd != nil
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
	ready := s.ready
	if ready == nil {
		return true
	}
	select {
	case <-ready:
		return false
	default:
		return true
	}
}

// Score posts one chunk.
func (s *Server) Score(ctx context.Context, session string, pcm []byte, rate int, threshold float64) (int, error) {
	if err := s.Wait(ctx); err != nil {
		return -1, err
	}
	s.mu.Lock()
	base := s.baseURL
	client := s.client
	alive := s.cmd != nil && !s.retired
	s.mu.Unlock()
	if !alive {
		return -1, errors.New("neural-vad is not running")
	}
	payload, _ := json.Marshal(map[string]any{
		"session":   session,
		"rate":      rate,
		"threshold": threshold,
		"pcm":       base64.StdEncoding.EncodeToString(pcm),
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/score", bytes.NewReader(payload))
	if err != nil {
		return -1, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return -1, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return -1, err
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return -1, fmt.Errorf("neural-vad HTTP %d: %s", res.StatusCode, strings.TrimSpace(string(body)))
	}
	var parsed struct {
		Last  int    `json:"last"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return -1, err
	}
	if parsed.Error != "" {
		return -1, fmt.Errorf("neural-vad: %s", parsed.Error)
	}
	return parsed.Last, nil
}

// Reset clears one socket's state.
func (s *Server) Reset(ctx context.Context, session string) error {
	if err := s.Wait(ctx); err != nil {
		return err
	}
	s.mu.Lock()
	base := s.baseURL
	client := s.client
	s.mu.Unlock()
	payload, _ := json.Marshal(map[string]string{"session": session})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/reset", bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	io.Copy(io.Discard, res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return fmt.Errorf("neural-vad reset HTTP %d", res.StatusCode)
	}
	return nil
}

// Close stops the process.
func (s *Server) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.retired = true
	if s.startErr == nil {
		s.startErr = fmt.Errorf("neural-vad closed")
	}
	ready := s.ready
	if ready != nil {
		select {
		case <-ready:
		default:
			close(ready)
		}
	}
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

func (s *Server) stderr() string {
	s.errMu.Lock()
	defer s.errMu.Unlock()
	return strings.TrimSpace(s.errBuf.String())
}

type limitWriter struct {
	dst *bytes.Buffer
	mu  *sync.Mutex
	max int
}

func (w *limitWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.dst.Len() >= w.max {
		return len(p), nil
	}
	remain := w.max - w.dst.Len()
	if len(p) > remain {
		p = p[:remain]
	}
	return w.dst.Write(p)
}
