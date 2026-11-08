package kokoro

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
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

// ErrUnavailable means the process could not start.
var ErrUnavailable = errors.New("kokoro unavailable")

// ErrBadRequest is an empty utterance, a speed outside [0.5, 2], or a voice
// the library rejected.
var ErrBadRequest = errors.New("kokoro request")

// BadRequestError is the message returned to the speech client.
type BadRequestError struct {
	Msg string
}

func (e *BadRequestError) Error() string {
	if e == nil || e.Msg == "" {
		return ErrBadRequest.Error()
	}
	return e.Msg
}

func (e *BadRequestError) Unwrap() error { return ErrBadRequest }

// SampleRate is Kokoro v1.0 playback rate.
const SampleRate = 24000

// DefaultVoice is used when the request and config omit a voice.
const DefaultVoice = "af_heart"

// Server is one python3 -m runtime.tts.kokoro_serve process.
type Server struct {
	mu           sync.Mutex
	cmd          *exec.Cmd
	baseURL      string
	client       *http.Client
	health       *http.Client
	ready        chan struct{}
	done         chan struct{}
	startErr     error
	retired      bool
	gen          int
	errMu        sync.Mutex
	errBuf       bytes.Buffer
	model        string
	voices       string
	defaultVoice string
	python       string
	pyPath       string
}

// Shared keeps one process for a model+voices pair.
type Shared struct {
	mu       sync.Mutex
	proc     *Server
	model    string
	voices   string
	booting  bool
	bootDone chan struct{}
}

var defaultShared = &Shared{}

// Paths resolves on-disk Kokoro files. Empty means unavailable.
func Paths(modelPath, voicesPath string) (model, voices string) {
	model = existingFile(modelPath)
	voices = existingFile(voicesPath)
	if model == "" {
		model = existingFile(envconfig.KokoroModel())
	}
	if voices == "" {
		voices = existingFile(envconfig.KokoroVoices())
	}
	if model == "" || voices == "" {
		return "", ""
	}
	return model, voices
}

func existingFile(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	st, err := os.Stat(p)
	if err != nil || st.IsDir() {
		return ""
	}
	return p
}

// Available is true when both local files are configured.
func Available(modelPath, voicesPath string) bool {
	m, v := Paths(modelPath, voicesPath)
	return m != "" && v != ""
}

// Warm starts the shared process in the background.
func Warm(modelPath, voicesPath, defaultVoice string) {
	m, v := Paths(modelPath, voicesPath)
	if m == "" {
		return
	}
	go func() {
		_, _ = defaultShared.ensure(context.Background(), m, v, defaultVoice)
	}()
}

// Speak synthesizes one utterance as a 16-bit mono WAV.
func Speak(ctx context.Context, modelPath, voicesPath, text, voice, lang, phonemes, emotion string, speed float64, defaultVoice string) (wav []byte, rate int, err error) {
	return defaultShared.Speak(ctx, modelPath, voicesPath, text, voice, lang, phonemes, emotion, speed, defaultVoice)
}

// SpeakStream synthesizes one utterance and calls emit for each PCM16 chunk.
func SpeakStream(ctx context.Context, modelPath, voicesPath, text, voice, lang, phonemes, emotion string, speed float64, defaultVoice string, emit func(rate int, pcm []byte) error) error {
	return defaultShared.SpeakStream(ctx, modelPath, voicesPath, text, voice, lang, phonemes, emotion, speed, defaultVoice, emit)
}

func (s *Shared) Speak(ctx context.Context, modelPath, voicesPath, text, voice, lang, phonemes, emotion string, speed float64, defaultVoice string) ([]byte, int, error) {
	m, v := Paths(modelPath, voicesPath)
	if m == "" {
		return nil, 0, ErrUnavailable
	}
	voice = resolveVoice(voice, defaultVoice)
	lang, err := ResolveLang(voice, lang)
	if err != nil {
		return nil, 0, err
	}
	if err := validateSpeed(speed); err != nil {
		return nil, 0, err
	}
	if _, emotion, err = ApplyEmotion(speed, emotion); err != nil {
		return nil, 0, err
	}
	s.mu.Lock() // hold across ensure+speak so a second utterance waits
	defer s.mu.Unlock()
	proc, err := s.ensureLocked(ctx, m, v, defaultVoice)
	if err != nil {
		return nil, 0, err
	}
	return proc.Speak(ctx, text, voice, lang, phonemes, emotion, speed)
}

func (s *Shared) SpeakStream(ctx context.Context, modelPath, voicesPath, text, voice, lang, phonemes, emotion string, speed float64, defaultVoice string, emit func(rate int, pcm []byte) error) error {
	m, v := Paths(modelPath, voicesPath)
	if m == "" {
		return ErrUnavailable
	}
	voice = resolveVoice(voice, defaultVoice)
	lang, err := ResolveLang(voice, lang)
	if err != nil {
		return err
	}
	if err := validateSpeed(speed); err != nil {
		return err
	}
	if _, emotion, err = ApplyEmotion(speed, emotion); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	proc, err := s.ensureLocked(ctx, m, v, defaultVoice)
	if err != nil {
		return err
	}
	return proc.SpeakStream(ctx, text, voice, lang, phonemes, emotion, speed, emit)
}

func resolveVoice(voice, defaultVoice string) string {
	if v := strings.TrimSpace(voice); v != "" {
		return v
	}
	if v := strings.TrimSpace(defaultVoice); v != "" {
		return v
	}
	return DefaultVoice
}

func validateSpeed(speed float64) error {
	if math.IsNaN(speed) || speed < 0.5 || speed > 2 {
		return &BadRequestError{Msg: "speed must be between 0.5 and 2"}
	}
	return nil
}

func (s *Shared) ensure(ctx context.Context, model, voices, defaultVoice string) (*Server, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ensureLocked(ctx, model, voices, defaultVoice)
}

func (s *Shared) ensureLocked(ctx context.Context, model, voices, defaultVoice string) (*Server, error) {
	for {
		if s.booting {
			done := s.bootDone
			s.mu.Unlock()
			if done == nil {
				s.mu.Lock()
				continue
			}
			select {
			case <-ctx.Done():
				s.mu.Lock()
				return nil, ctx.Err()
			case <-done:
			}
			s.mu.Lock()
			continue
		}
		if s.proc != nil && s.model == model && s.voices == voices {
			proc := s.proc
			err := proc.Wait(ctx)
			if proc.Ready() {
				if err != nil {
					return nil, err
				}
				return proc, nil
			}
			// A cancel while the child is still loading leaves it in place.
			// A crash or a failed health check drops it so the next speak
			// can start a new process. This request starts one when ctx is live.
			if proc.Loading() && ctx.Err() != nil {
				if err != nil {
					return nil, err
				}
				return nil, ctx.Err()
			}
			s.proc = nil
			s.model = ""
			s.voices = ""
			s.mu.Unlock()
			proc.Close()
			s.mu.Lock()
			if ctx.Err() != nil {
				if err != nil {
					return nil, err
				}
				return nil, ctx.Err()
			}
			continue
		}
		old := s.proc
		s.proc = nil
		s.model = ""
		s.voices = ""
		s.booting = true
		done := make(chan struct{})
		s.bootDone = done
		s.mu.Unlock()

		if old != nil {
			old.Close()
		}
		proc, err := StartServer(ctx, model, voices, defaultVoice)
		if err == nil {
			err = proc.Wait(ctx)
		}
		s.mu.Lock()
		if err != nil {
			if proc != nil {
				if (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) && proc.Loading() {
					s.proc = proc
					s.model = model
					s.voices = voices
					s.booting = false
					close(done)
					s.bootDone = nil
					return nil, err
				}
				proc.Close()
			}
			s.booting = false
			close(done)
			s.bootDone = nil
			return nil, err
		}
		s.proc = proc
		s.model = model
		s.voices = voices
		s.booting = false
		close(done)
		s.bootDone = nil
		return proc, nil
	}
}

// StartServer launches the serve module on a free loopback port.
func StartServer(ctx context.Context, modelPath, voicesPath, defaultVoice string) (*Server, error) {
	_ = ctx
	modelPath = strings.TrimSpace(modelPath)
	voicesPath = strings.TrimSpace(voicesPath)
	if modelPath == "" || voicesPath == "" {
		return nil, fmt.Errorf("%w: model or voices empty", ErrUnavailable)
	}
	if st, err := os.Stat(modelPath); err != nil || st.IsDir() {
		return nil, fmt.Errorf("%w: model missing", ErrUnavailable)
	}
	if st, err := os.Stat(voicesPath); err != nil || st.IsDir() {
		return nil, fmt.Errorf("%w: voices missing", ErrUnavailable)
	}
	python, err := lookPython()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	pyPath := runtimePythonPath()
	if pyPath == "" {
		return nil, fmt.Errorf("%w: runtime package not found (set ZEROLLAMA_REPO)", ErrUnavailable)
	}
	dv := strings.TrimSpace(defaultVoice)
	if dv == "" {
		dv = DefaultVoice
	}
	s := &Server{
		client:       &http.Client{}, // request context bounds speak
		health:       &http.Client{Timeout: 200 * time.Millisecond},
		model:        modelPath,
		voices:       voicesPath,
		defaultVoice: dv,
		python:       python,
		pyPath:       pyPath,
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
		if _, err := os.Stat(filepath.Join(p, "runtime", "tts", "kokoro_serve.py")); err == nil {
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
	_, err := os.Stat(filepath.Join(dir, "runtime", "runtime", "tts", "kokoro_serve.py"))
	return err == nil
}

func (s *Server) spawn() error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	cmd := exec.Command(s.python, "-m", "runtime.tts.kokoro_serve",
		"--model", s.model,
		"--voices", s.voices,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--default-voice", s.defaultVoice,
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
				s.startErr = fmt.Errorf("kokoro exited: %v", err)
			}
		default:
			if s.startErr == nil {
				s.startErr = fmt.Errorf("kokoro exited before ready: %v: %s", err, s.stderr())
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
	deadline := time.Now().Add(envconfig.ModalityKokoroTimeout())
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
			s.failStart(ready, gen, fmt.Errorf("kokoro health timed out: %s", s.stderr()))
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

// Loading is true while the child is still starting.
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

// CloseShared stops the shared process. Tests use it so the next speak starts clean.
func CloseShared() {
	defaultShared.mu.Lock()
	proc := defaultShared.proc
	defaultShared.proc = nil
	defaultShared.model = ""
	defaultShared.voices = ""
	defaultShared.mu.Unlock()
	if proc != nil {
		proc.Close()
	}
}

// Speak posts one full WAV utterance.
func (s *Server) Speak(ctx context.Context, text, voice, lang, phonemes, emotion string, speed float64) ([]byte, int, error) {
	voice = resolveVoice(voice, s.defaultVoice)
	lang, err := ResolveLang(voice, lang)
	if err != nil {
		return nil, 0, err
	}
	if err := validateSpeed(speed); err != nil {
		return nil, 0, err
	}
	if _, emotion, err = ApplyEmotion(speed, emotion); err != nil {
		return nil, 0, err
	}
	if err := s.Wait(ctx); err != nil {
		return nil, 0, err
	}
	s.mu.Lock()
	base := s.baseURL
	client := s.client
	alive := s.cmd != nil && !s.retired
	s.mu.Unlock()
	if !alive {
		return nil, 0, errors.New("kokoro is not running")
	}
	payload, _ := json.Marshal(speakPayload(text, voice, lang, phonemes, emotion, speed, false))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/speak", bytes.NewReader(payload))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 32<<20))
	if err != nil {
		return nil, 0, err
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return nil, 0, httpStatusError(res.StatusCode, body)
	}
	var parsed struct {
		Rate  int    `json:"rate"`
		Wav   string `json:"wav"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, 0, err
	}
	if parsed.Error != "" {
		return nil, 0, fmt.Errorf("kokoro: %s", parsed.Error)
	}
	raw, err := base64.StdEncoding.DecodeString(parsed.Wav)
	if err != nil {
		return nil, 0, err
	}
	rate := parsed.Rate
	if rate <= 0 {
		rate = SampleRate
	}
	return raw, rate, nil
}

// SpeakStream reads NDJSON PCM chunks from the loopback process.
func (s *Server) SpeakStream(ctx context.Context, text, voice, lang, phonemes, emotion string, speed float64, emit func(rate int, pcm []byte) error) error {
	if emit == nil {
		return errors.New("emit required")
	}
	voice = resolveVoice(voice, s.defaultVoice)
	lang, err := ResolveLang(voice, lang)
	if err != nil {
		return err
	}
	if err := validateSpeed(speed); err != nil {
		return err
	}
	if _, emotion, err = ApplyEmotion(speed, emotion); err != nil {
		return err
	}
	if err := s.Wait(ctx); err != nil {
		return err
	}
	s.mu.Lock()
	base := s.baseURL
	client := s.client
	alive := s.cmd != nil && !s.retired
	s.mu.Unlock()
	if !alive {
		return errors.New("kokoro is not running")
	}
	payload, _ := json.Marshal(speakPayload(text, voice, lang, phonemes, emotion, speed, true))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/speak", bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(res.Body, 4096))
		return httpStatusError(res.StatusCode, body)
	}
	sc := bufio.NewScanner(res.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 8<<20)
	emitted := false
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var parsed struct {
			Rate  int    `json:"rate"`
			PCM   string `json:"pcm"`
			Done  bool   `json:"done"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal(line, &parsed); err != nil {
			return err
		}
		if parsed.Error != "" {
			if !emitted {
				return &BadRequestError{Msg: parsed.Error}
			}
			return fmt.Errorf("kokoro: %s", parsed.Error)
		}
		if parsed.Done {
			return nil
		}
		if parsed.PCM == "" {
			continue
		}
		pcm, err := base64.StdEncoding.DecodeString(parsed.PCM)
		if err != nil {
			return err
		}
		rate := parsed.Rate
		if rate <= 0 {
			rate = SampleRate
		}
		if err := emit(rate, pcm); err != nil {
			return err
		}
		emitted = true
	}
	if err := sc.Err(); err != nil {
		return err
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
		s.startErr = fmt.Errorf("kokoro closed")
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

func speakPayload(text, voice, lang, phonemes, emotion string, speed float64, stream bool) map[string]any {
	body := map[string]any{
		"text":   text,
		"voice":  voice,
		"lang":   lang,
		"speed":  speed,
		"stream": stream,
	}
	if emotion != "" {
		body["emotion"] = emotion
	}
	if phonemes != "" {
		body["phoneme_silence"] = json.RawMessage(phonemes)
	}
	return body
}

func httpStatusError(status int, body []byte) error {
	msg := strings.TrimSpace(string(body))
	var parsed struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(body, &parsed) == nil && strings.TrimSpace(parsed.Error) != "" {
		msg = strings.TrimSpace(parsed.Error)
	}
	if status == http.StatusBadRequest {
		return &BadRequestError{Msg: msg}
	}
	return fmt.Errorf("kokoro HTTP %d: %s", status, msg)
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
