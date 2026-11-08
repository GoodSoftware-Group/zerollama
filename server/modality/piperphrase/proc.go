package piperphrase

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"

	"golang.org/x/sys/unix"
)

// Proc is one Piper process with the ONNX voice already loaded.
// Speak sends one JSON line, the same shape as Piper --json-input, and reads
// the WAV path Piper prints when that line is done.
type Proc struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	rawOut *os.File
	rawErr *os.File
	outFd  int
	errFd  int
	raw    bool
	cancel context.CancelFunc
	stop   sync.Once
	mu     sync.Mutex
}

type piperLine struct {
	Text       string `json:"text"`
	OutputFile string `json:"output_file"`
}

// Start launches bin. args are the Piper flags, including --json-input.
// Close stops the process.
func Start(bin string, args []string) (*Proc, error) {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, bin, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		_ = stdin.Close()
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		_ = stdin.Close()
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, err
	}
	if rawOutput(args) {
		outFile, errFile, outFd, errFd, err := rawPipes(stdout, stderr)
		if err != nil {
			cancel()
			return nil, err
		}
		return &Proc{cmd: cmd, stdin: stdin, rawOut: outFile, rawErr: errFile, outFd: outFd, errFd: errFd, raw: true, cancel: cancel}, nil
	}
	go io.Copy(io.Discard, stderr)
	return &Proc{cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout), cancel: cancel}, nil
}

func rawPipes(stdout, stderr io.ReadCloser) (*os.File, *os.File, int, int, error) {
	outFile, ok := stdout.(*os.File)
	errFile, errOK := stderr.(*os.File)
	if !ok || !errOK {
		return nil, nil, 0, 0, errors.New("piper stream is not a pipe")
	}
	// Fd puts the descriptor into blocking mode. Set nonblock after that and
	// keep the integers: a later Fd call would clear nonblock again.
	outFd := int(outFile.Fd())
	errFd := int(errFile.Fd())
	if err := unix.SetNonblock(outFd, true); err != nil {
		return nil, nil, 0, 0, err
	}
	if err := unix.SetNonblock(errFd, true); err != nil {
		return nil, nil, 0, 0, err
	}
	return outFile, errFile, outFd, errFd, nil
}

func rawOutput(args []string) bool {
	for _, arg := range args {
		if arg == "--output_raw" || arg == "--output-raw" {
			return true
		}
	}
	return false
}

// Speak writes one utterance and returns the WAV bytes Piper wrote.
// A cancelled ctx stops this process so the next Speak starts a new one.
func (p *Proc) Speak(ctx context.Context, text, wavPath string) ([]byte, error) {
	if p == nil {
		return nil, errors.New("piper is not running")
	}
	if p.raw {
		return nil, errors.New("piper raw output has no wav path")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(piperLine{Text: text, OutputFile: wavPath})
	if err != nil {
		return nil, err
	}
	if _, err := p.stdin.Write(append(payload, '\n')); err != nil {
		p.halt()
		return nil, err
	}
	type got struct {
		line string
		err  error
	}
	ch := make(chan got, 1)
	go func() {
		line, err := p.stdout.ReadString('\n')
		ch <- got{line, err}
	}()
	select {
	case <-ctx.Done():
		p.halt()
		return nil, ctx.Err()
	case r := <-ch:
		if r.err != nil {
			p.halt()
			return nil, r.err
		}
		path := strings.TrimSpace(r.line)
		if path == "" {
			p.halt()
			return nil, errors.New("piper returned no audio path")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			p.halt()
			return nil, err
		}
		return data, nil
	}
}

// SpeakPCM writes one JSON line and emits little-endian s16le mono as Piper
// synthesizes it. The process must have been started with --output_raw.
// Piper logs "Real-time factor:" on stderr when that line's audio is done.
// A cancelled ctx stops this process.
func (p *Proc) SpeakPCM(ctx context.Context, text string, onChunk func([]byte) error) error {
	if p == nil || !p.raw {
		return errors.New("piper is not streaming")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	payload, err := json.Marshal(struct {
		Text string `json:"text"`
	}{Text: text})
	if err != nil {
		return err
	}
	if _, err := p.stdin.Write(append(payload, '\n')); err != nil {
		p.halt()
		return err
	}
	var errBuf []byte
	var hold []byte
	buf := make([]byte, 4096)
	for {
		if err := ctx.Err(); err != nil {
			p.halt()
			return err
		}
		got := false
		n, err := readPipe(p.outFd, buf)
		if n > 0 {
			got = true
			hold = append(hold, buf[:n]...)
			if err := flushPCM(&hold, onChunk); err != nil {
				p.halt()
				return err
			}
		} else if err != nil {
			p.halt()
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		n, err = readPipe(p.errFd, buf)
		if n > 0 {
			got = true
			errBuf = append(errBuf, buf[:n]...)
		} else if err != nil {
			p.halt()
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		if !bytes.Contains(errBuf, []byte("Real-time factor:")) {
			if !got {
				if err := waitPipes(p.outFd, p.errFd); err != nil {
					p.halt()
					return err
				}
			}
			continue
		}
		for {
			n, err = readPipe(p.outFd, buf)
			if n > 0 {
				hold = append(hold, buf[:n]...)
				continue
			}
			if err != nil {
				p.halt()
				if ctx.Err() != nil {
					return ctx.Err()
				}
				return err
			}
			break
		}
		return flushPCM(&hold, onChunk)
	}
}

func flushPCM(hold *[]byte, onChunk func([]byte) error) error {
	n := len(*hold)
	if n%2 != 0 {
		n--
	}
	if n == 0 {
		return nil
	}
	chunk := make([]byte, n)
	copy(chunk, (*hold)[:n])
	*hold = append([]byte(nil), (*hold)[n:]...)
	if onChunk == nil {
		return nil
	}
	return onChunk(chunk)
}

func readPipe(fd int, buf []byte) (int, error) {
	n, err := unix.Read(fd, buf)
	if errors.Is(err, unix.EAGAIN) || errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EINTR) {
		return 0, nil
	}
	if err != nil {
		return n, err
	}
	if n == 0 {
		return 0, io.EOF
	}
	return n, nil
}

func waitPipes(outFd, errFd int) error {
	fds := []unix.PollFd{
		{Fd: int32(outFd), Events: unix.POLLIN},
		{Fd: int32(errFd), Events: unix.POLLIN},
	}
	_, err := unix.Poll(fds, 15)
	if err != nil && !errors.Is(err, unix.EINTR) {
		return err
	}
	return nil
}

// Close stops the process. It is safe while Speak is waiting.
func (p *Proc) Close() {
	if p == nil {
		return
	}
	p.cancel()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.halt()
}

func (p *Proc) halt() {
	p.stop.Do(func() {
		p.cancel()
		_ = p.stdin.Close()
		_ = p.cmd.Wait()
	})
}
