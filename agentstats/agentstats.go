// Package agentstats writes JSONL diagnostics for MLX Gemma agent loops (Hermes, etc.).
// The log file is truncated on each serve start so operators get one session per restart.
package agentstats

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/version"
)

var (
	mu       sync.Mutex
	enabled  bool
	path     string
	file     *os.File
	enc      *json.Encoder
)

// Init opens the agent stats log for a new serve session. Any non-empty prior file
// is rotated to <path>.prev before writing serve_start (version, pid, …).
func Init(logPath string) error {
	mu.Lock()
	defer mu.Unlock()

	closeLocked()

	logPath = strings.TrimSpace(logPath)
	if logPath == "" || logPath == "0" || strings.EqualFold(logPath, "off") || strings.EqualFold(logPath, "false") {
		enabled = false
		path = ""
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}

	prevLog := rotatePreviousLog(logPath)

	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}

	enabled = true
	path = logPath
	file = f
	enc = json.NewEncoder(f)

	start := map[string]any{
		"ts":         time.Now().UTC().Format(time.RFC3339Nano),
		"event":      "serve_start",
		"path":       logPath,
		"version":    version.Version,
		"edge_build": version.IsEdgeBuild(),
		"pid":        os.Getpid(),
		"goos":       runtime.GOOS,
		"goarch":     runtime.GOARCH,
	}
	if prevLog != "" {
		start["previous_log"] = prevLog
	}
	if err := enc.Encode(start); err != nil {
		closeLocked()
		return err
	}
	return nil
}

// Path returns the active log path, or "" when disabled.
func Path() string {
	mu.Lock()
	defer mu.Unlock()
	if !enabled {
		return ""
	}
	return path
}

// Enabled reports whether agent stats logging is active.
func Enabled() bool {
	mu.Lock()
	defer mu.Unlock()
	return enabled
}

// Record appends one JSON object when the event matches agent/Gemma traffic filters.
func Record(event string, fields map[string]any) {
	if !shouldRecord(event, fields) {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	if !enabled || enc == nil {
		return
	}
	row := map[string]any{
		"ts":    time.Now().UTC().Format(time.RFC3339Nano),
		"event": event,
	}
	for k, v := range fields {
		row[k] = v
	}
	_ = enc.Encode(row)
}

func shouldRecord(event string, fields map[string]any) bool {
	if strings.HasPrefix(event, "mlx_") {
		return true
	}
	if fields == nil {
		return false
	}
	if key, _ := fields["prompt_cache_key"].(string); strings.TrimSpace(key) != "" {
		return true
	}
	if model, _ := fields["model"].(string); isGemmaModel(model) {
		return true
	}
	return false
}

func isGemmaModel(model string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(model)), "gemma")
}

// rotatePreviousLog archives the prior session file so operators can compare runs
// across binary versions. Returns the archive path when a file was rotated.
func rotatePreviousLog(logPath string) string {
	info, err := os.Stat(logPath)
	if err != nil || info.Size() == 0 {
		return ""
	}
	archive := logPath + ".prev"
	_ = os.Remove(archive)
	if err := os.Rename(logPath, archive); err != nil {
		return ""
	}
	return archive
}

func closeLocked() {
	enabled = false
	path = ""
	enc = nil
	if file != nil {
		_ = file.Close()
		file = nil
	}
}
