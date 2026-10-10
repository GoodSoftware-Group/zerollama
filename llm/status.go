package llm

import (
	"bytes"
	"io"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/agentstats"
)

// StatusWriter is a writer that captures error messages from the llama runner process
type StatusWriter struct {
	out io.Writer
	// Subprocess wrappers may wire both stdout and stderr to the same
	// StatusWriter, and os/exec serializes Write calls in that case.
	lastErrMsg atomic.Value
	// lastWriteNs is updated on every non-empty Write so load waits can treat
	// progress logs as stall-timeout activity (MLX materialize can exceed 5m).
	lastWriteNs atomic.Int64
	// LastErrMsg is read by the in-process ggml runner in server.go.
	LastErrMsg string
}

const maxCapturedErrorBytes = 8 * 1024

func NewStatusWriter(out io.Writer) *StatusWriter {
	return &StatusWriter{
		out: out,
	}
}

func (w *StatusWriter) LastError() string {
	if w == nil {
		return ""
	}
	if v := w.lastErrMsg.Load(); v != nil {
		return v.(string)
	}
	return ""
}

// LastWrite is the time of the most recent non-empty Write, or zero if none.
func (w *StatusWriter) LastWrite() time.Time {
	if w == nil {
		return time.Time{}
	}
	if ns := w.lastWriteNs.Load(); ns > 0 {
		return time.Unix(0, ns)
	}
	return time.Time{}
}

func (w *StatusWriter) SetLastError(msg string) {
	if w == nil {
		return
	}
	w.LastErrMsg = msg
	w.lastErrMsg.Store(msg)
}

func (w *StatusWriter) AppendError(msg string) {
	if w == nil || msg == "" {
		return
	}

	if current := w.LastError(); current != "" {
		msg = current + "\n" + msg
	}

	if len(msg) > maxCapturedErrorBytes {
		msg = msg[len(msg)-maxCapturedErrorBytes:]
		if i := strings.IndexByte(msg, '\n'); i >= 0 {
			msg = msg[i+1:]
		}
	}

	w.SetLastError(msg)
}

var errorPrefixes = []string{
	"mlx:",
	"MLX:",
	"panic:",
	"fatal error:",
	"error:",
	"Error:",
	"CUDA error",
	"ROCm error",
	"cudaMalloc failed",
	"\"ERR\"",
	"error loading model",
	"GGML_ASSERT",
	"Deepseek2 does not support K-shift",
	"signal arrived during cgo execution",
	"llama_init_from_model:",
}

var outOfMemorySubstrings = []string{
	"out of memory",
	"out of device memory",
	"cudaMalloc failed",
	"hipMalloc failed",
	"failed to allocate",
	"allocation failed",
	"not enough memory",
	"insufficient memory",
	"vk_error_out_of_device_memory",
	"erroroutofmemory",
}

var recoverableOutOfMemorySubstrings = []string{
	"retrying without pipeline parallelism",
}

func IsOutOfMemory(err error) bool {
	if err == nil {
		return false
	}
	return IsOutOfMemoryMessage(err.Error())
}

func isRecoverableOutOfMemory(err error) bool {
	if err == nil {
		return false
	}
	return isRecoverableOutOfMemoryMessage(err.Error())
}

func IsOutOfMemoryMessage(msg string) bool {
	msg = strings.ToLower(msg)
	for _, needle := range outOfMemorySubstrings {
		if strings.Contains(msg, strings.ToLower(needle)) {
			return true
		}
	}
	return false
}

func isRecoverableOutOfMemoryMessage(msg string) bool {
	lastLine := lastNonEmptyLine(msg)
	if !IsOutOfMemoryMessage(lastLine) {
		return false
	}

	lastLine = strings.ToLower(lastLine)
	for _, needle := range recoverableOutOfMemorySubstrings {
		if strings.Contains(lastLine, strings.ToLower(needle)) {
			return true
		}
	}
	return false
}

func lastNonEmptyLine(msg string) string {
	lines := strings.Split(strings.TrimSpace(msg), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return line
		}
	}
	return ""
}

func (w *StatusWriter) Write(b []byte) (int, error) {
	wrote := false
	for _, raw := range bytes.Split(b, []byte{'\n'}) {
		line := strings.TrimRight(string(raw), " \t\r")
		if line == "" {
			continue
		}
		wrote = true

		if errMsg := statusErrorLine(line); errMsg != "" {
			w.AppendError(errMsg)
		}

		agentstats.MaybeRecordRunnerLine(line)
	}
	if wrote {
		w.lastWriteNs.Store(time.Now().UnixNano())
	}

	if w.out == nil {
		return len(b), nil
	}

	return w.out.Write(b)
}

func isInformationalMLXLine(line string) bool {
	lower := strings.ToLower(line)
	if strings.Contains(lower, "optional symbol") && strings.Contains(lower, "(no-op)") {
		return true
	}
	// uma_glue auto/degraded logs: "uma_mlx: auto — broker not running, MLX ungated".
	// Substring "mlx:" must not make these look like runner failures (UMA is optional).
	if strings.Contains(lower, "uma_mlx:") || strings.Contains(lower, "mlx ungated") {
		return true
	}
	return false
}

func isIdentByte(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') ||
		(b >= '0' && b <= '9') || b == '_'
}

// indexErrorPrefix finds prefix at a token boundary so "uma_mlx:" does not match "mlx:".
func indexErrorPrefix(line, prefix string) int {
	for i := 0; i <= len(line)-len(prefix); {
		j := strings.Index(line[i:], prefix)
		if j < 0 {
			return -1
		}
		j += i
		if j == 0 || !isIdentByte(line[j-1]) {
			return j
		}
		i = j + 1
	}
	return -1
}

func statusErrorLine(line string) string {
	if isInformationalMLXLine(line) {
		return ""
	}

	errStart := -1
	errPrefix := ""
	for _, prefix := range errorPrefixes {
		if i := indexErrorPrefix(line, prefix); i >= 0 && (errStart < 0 || i < errStart) {
			errStart = i
			errPrefix = prefix
		}
	}

	if errStart >= 0 {
		return errPrefix + strings.TrimRight(line[errStart+len(errPrefix):], " \t\r")
	}

	if IsOutOfMemoryMessage(line) {
		return line
	}

	return ""
}
