package whispercli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
)

// ErrBadAudio means this whisper-server cannot decode a posted WAV.
// whisper.cpp before 1.9.1 treats the upload as a filename. Its error
// handler then replaces the JSON body with "Invalid request".
var ErrBadAudio = errors.New("whisper-server cannot read posted audio")

// Server is one whisper-server process for one duplex call.
// It loads the model once and posts each clip to /inference.
type Server struct {
	mu       sync.Mutex
	cmd      *exec.Cmd
	baseURL  string
	client   *http.Client
	health   *http.Client
	ready    chan struct{}
	done     chan struct{}
	startErr error
	waitErr  error
	retired  bool
	gen      int
	errMu    sync.Mutex
	errBuf   bytes.Buffer
	bin      string
	model    string
}

// StartServer launches bin with the model on a free loopback port and --no-gpu.
// It returns as soon as the process starts. Wait until Ready or Wait returns.
func StartServer(ctx context.Context, bin, modelPath string) (*Server, error) {
	bin = strings.TrimSpace(bin)
	modelPath = strings.TrimSpace(modelPath)
	if bin == "" {
		return nil, fmt.Errorf("whisper-server binary is empty")
	}
	if modelPath == "" {
		return nil, fmt.Errorf("whisper model path is empty")
	}
	s := &Server{
		client: &http.Client{Timeout: 10 * time.Minute},
		health: &http.Client{Timeout: 200 * time.Millisecond},
		bin:    bin,
		model:  modelPath,
	}
	if err := s.spawn(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Server) spawn(ctx context.Context) error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	args := []string{
		"-m", s.model,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--no-gpu",
	}
	cmd := exec.Command(s.bin, args...)
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
	s.waitErr = nil
	s.errMu.Lock()
	s.errBuf.Reset()
	s.errMu.Unlock()
	s.mu.Unlock()

	go io.Copy(&limitWriter{dst: &s.errBuf, mu: &s.errMu, max: 4096}, stderr)
	go s.waitReady(ctx, ready, gen)
	go s.watchExit(cmd, ready, done, gen)
	return nil
}

func (s *Server) watchExit(cmd *exec.Cmd, ready, done chan struct{}, gen int) {
	err := cmd.Wait()
	s.mu.Lock()
	if s.gen == gen {
		s.waitErr = err
		if s.cmd == cmd {
			s.cmd = nil
		}
		select {
		case <-ready:
			if s.startErr == nil && !s.retired {
				s.startErr = fmt.Errorf("whisper-server exited: %v", err)
			}
		default:
			if s.startErr == nil {
				if err != nil {
					s.startErr = fmt.Errorf("whisper-server exited before ready: %w: %s", err, s.stderr())
				} else {
					s.startErr = fmt.Errorf("whisper-server exited before ready: %s", s.stderr())
				}
			}
			close(ready)
		}
	}
	s.mu.Unlock()
	close(done)
}

func (s *Server) waitReady(ctx context.Context, ready chan struct{}, gen int) {
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
	deadline := time.Now().Add(2 * time.Minute)
	if dl, ok := ctx.Deadline(); ok && dl.Before(deadline) {
		deadline = dl
	}
	for {
		s.mu.Lock()
		if s.gen != gen {
			s.mu.Unlock()
			return
		}
		base := s.baseURL
		client := s.health
		s.mu.Unlock()

		if ctx.Err() != nil {
			s.failStart(ready, gen, ctx.Err())
			return
		}
		if time.Now().After(deadline) {
			s.failStart(ready, gen, fmt.Errorf("whisper-server health timed out: %s", s.stderr()))
			return
		}
		reqCtx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
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
		select {
		case <-ctx.Done():
			s.failStart(ready, gen, ctx.Err())
			return
		case <-time.After(50 * time.Millisecond):
		}
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
	if err := json.Unmarshal(body, &parsed); err != nil {
		return false
	}
	return parsed.Status == "ok"
}

// Wait blocks until the model is ready or the start failed.
func (s *Server) Wait(ctx context.Context) error {
	if s == nil {
		return fmt.Errorf("whisper-server is not running")
	}
	s.mu.Lock()
	ready := s.ready
	s.mu.Unlock()
	if ready == nil {
		return fmt.Errorf("whisper-server is not running")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-ready:
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.retired {
			return fmt.Errorf("whisper-server retired")
		}
		return s.startErr
	}
}

// Ready is true when the process answered /health and has not been retired.
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

// Retired is true after a binary that cannot read a posted WAV.
func (s *Server) Retired() bool {
	if s == nil {
		return true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.retired
}

// Transcribe posts one WAV to /inference.
func (s *Server) Transcribe(ctx context.Context, wav []byte, language string) (string, error) {
	if s == nil {
		return "", fmt.Errorf("whisper-server is not running")
	}
	if err := s.Wait(ctx); err != nil {
		return "", err
	}
	s.mu.Lock()
	base := s.baseURL
	client := s.client
	retired := s.retired
	alive := s.cmd != nil
	s.mu.Unlock()
	if retired {
		return "", fmt.Errorf("whisper-server retired")
	}
	if !alive {
		return "", fmt.Errorf("whisper-server is not running")
	}
	text, err := PostInference(ctx, client, base, wav, language)
	if errors.Is(err, ErrBadAudio) {
		s.retire()
		return "", err
	}
	if err != nil && isDeadConn(err) {
		s.markDown(err)
	}
	return text, err
}

// Restart stops the current process and starts another on a new port.
func (s *Server) Restart(ctx context.Context) error {
	if s == nil {
		return fmt.Errorf("whisper-server is not running")
	}
	s.mu.Lock()
	if s.retired {
		s.mu.Unlock()
		return fmt.Errorf("whisper-server retired")
	}
	s.mu.Unlock()
	s.kill()
	return s.spawn(ctx)
}

// Close stops the process and marks it retired.
func (s *Server) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.retired = true
	if s.startErr == nil {
		s.startErr = fmt.Errorf("whisper-server closed")
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
	s.signal(cmd)
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		if cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		<-done
	}
}

func (s *Server) signal(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)
}

func (s *Server) retire() {
	s.Close()
}

func (s *Server) markDown(err error) {
	s.mu.Lock()
	if s.startErr == nil {
		s.startErr = err
	}
	cmd := s.cmd
	s.cmd = nil
	s.mu.Unlock()
	s.signal(cmd)
}

func (s *Server) stderr() string {
	s.errMu.Lock()
	defer s.errMu.Unlock()
	return strings.TrimSpace(s.errBuf.String())
}

func isDeadConn(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "connection reset") ||
		errors.Is(err, io.EOF)
}

// PostInference posts one clip to a whisper-server /inference endpoint.
func PostInference(ctx context.Context, client *http.Client, baseURL string, wav []byte, language string) (string, error) {
	if client == nil {
		client = http.DefaultClient
	}
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("file", "clip.wav")
	if err != nil {
		return "", err
	}
	if _, err := part.Write(wav); err != nil {
		return "", err
	}
	fields := map[string]string{
		"response_format": "json",
		"beam_size":       "1",
		"best_of":         "1",
		"no_timestamps":   "true",
	}
	if language = strings.TrimSpace(language); language != "" {
		fields["language"] = language
	}
	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			return "", err
		}
	}
	if err := w.Close(); err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+"/inference", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	res, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 4<<20))
	if err != nil {
		return "", err
	}
	// whisper-server's error_handler replaces a 400 JSON body with "Invalid request".
	if res.StatusCode == http.StatusBadRequest {
		return "", ErrBadAudio
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", fmt.Errorf("whisper-server: HTTP %d: %s", res.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Text  string `json:"text"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("whisper-server: %w: %s", err, strings.TrimSpace(string(raw)))
	}
	if parsed.Error != "" {
		if strings.Contains(parsed.Error, "failed to read audio data") {
			return "", ErrBadAudio
		}
		return "", fmt.Errorf("whisper-server: %s", parsed.Error)
	}
	return strings.TrimSpace(parsed.Text), nil
}

// LookServer returns bin when it is executable, otherwise empty.
// A bare name uses PATH. A path with a separator is checked as written.
func LookServer(bin string) string {
	bin = strings.TrimSpace(bin)
	if bin == "" {
		return ""
	}
	if filepath.Base(bin) != bin {
		info, err := os.Stat(bin)
		if err != nil || info.IsDir() {
			return ""
		}
		return bin
	}
	path, err := exec.LookPath(bin)
	if err != nil {
		return ""
	}
	return path
}
