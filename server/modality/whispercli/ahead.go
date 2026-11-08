package whispercli

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

// Ahead is one whisper.cpp process started before the clip is ready.
// The CLI loads the model, then blocks on stdin. Finish writes the WAV and
// reads the text file.
type Ahead struct {
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	out     string
	cancel  context.CancelFunc
	stop    sync.Once
	waitErr error
	errMu   sync.Mutex
	errBuf  bytes.Buffer
}

// StartAhead launches bin with args. outBase is the -of path; the text is
// outBase + ".txt". Close stops the process.
func StartAhead(ctx context.Context, bin string, args []string, outBase string) (*Ahead, error) {
	ctx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(ctx, bin, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		_ = stdin.Close()
		return nil, err
	}
	cmd.Stdout = io.Discard
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, err
	}
	a := &Ahead{cmd: cmd, stdin: stdin, out: outBase, cancel: cancel}
	go func() {
		_, _ = io.Copy(&limitWriter{dst: &a.errBuf, mu: &a.errMu, max: 4096}, stderr)
	}()
	return a, nil
}

// Finish writes the finished WAV and returns the transcript.
func (a *Ahead) Finish(ctx context.Context, wav []byte) (string, error) {
	if a == nil {
		return "", fmt.Errorf("whisper is not running")
	}
	done := make(chan error, 1)
	go func() {
		_, err := a.stdin.Write(wav)
		if err == nil {
			err = a.stdin.Close()
		}
		if err == nil {
			err = a.wait()
		}
		done <- err
	}()
	select {
	case <-ctx.Done():
		a.Close()
		return "", ctx.Err()
	case err := <-done:
		if err != nil && ctx.Err() != nil {
			return "", ctx.Err()
		}
		if err != nil {
			return "", fmt.Errorf("whisper: %w: %s", err, a.stderr())
		}
		raw, err := os.ReadFile(a.out + ".txt")
		if err != nil {
			return "", fmt.Errorf("read whisper output: %w", err)
		}
		return strings.TrimSpace(string(raw)), nil
	}
}

// Close stops the process.
func (a *Ahead) Close() {
	if a == nil {
		return
	}
	a.cancel()
	_ = a.stdin.Close()
	_ = a.wait()
}

func (a *Ahead) wait() error {
	a.stop.Do(func() {
		a.waitErr = a.cmd.Wait()
	})
	return a.waitErr
}

func (a *Ahead) stderr() string {
	a.errMu.Lock()
	defer a.errMu.Unlock()
	return strings.TrimSpace(a.errBuf.String())
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
	room := w.max - w.dst.Len()
	if len(p) > room {
		p = p[:room]
	}
	_, _ = w.dst.Write(p)
	return len(p), nil
}
