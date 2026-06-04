// Package runtimeworker embeds the zerollama Python inference runtime in-process (CGO).
// When started, sets an internal base URL so Go handlers proxy to localhost without a separate process.
package runtimeworker

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/runtimeworker/pyembed"
)

var (
	mu      sync.RWMutex
	baseURL string
)

// Client tracks embedded runtime lifecycle (uvicorn runs on a daemon thread).
type Client struct{}

func checkLoopbackPortFree(port int) error {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf(
			"loopback %s already in use (stop stale zerollama serve or zerollama-runtime sidecar before embed): %w",
			addr,
			err,
		)
	}
	_ = ln.Close()
	return nil
}

func newEmbedBootToken() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

func healthEmbedBoot(body []byte) string {
	var h struct {
		EmbedBoot string `json:"embed_boot"`
	}
	if json.Unmarshal(body, &h) != nil {
		return ""
	}
	return strings.TrimSpace(h.EmbedBoot)
}

// Start launches embedded Python runtime HTTP on 127.0.0.1:port.
// Shares CPython with training when training already called Py_Initialize.
func Start(ctx context.Context, repoRoot string) (*Client, error) {
	_ = ctx
	if !envconfig.RuntimeEmbedEnabled() {
		return nil, fmt.Errorf("embedded runtime disabled (ZEROLLAMA_RUNTIME_EMBED=0)")
	}
	if pyembed.IsStarted() {
		return &Client{}, nil
	}
	if repoRoot == "" {
		repoRoot = resolveRepoRoot()
	}
	if repoRoot == "" {
		return nil, fmt.Errorf("zerollama repo root not found (set OLLAMA_TRAINING_PYTHONPATH)")
	}

	port := envconfig.RuntimeEmbedPort()
	if err := checkLoopbackPortFree(port); err != nil {
		return nil, err
	}
	boot := newEmbedBootToken()
	_ = os.Setenv("ZEROLLAMA_RUNTIME_EMBED_BOOT", boot)
	runtimeParent := repoRoot + "/runtime"
	if err := pyembed.EmbedStart(repoRoot, runtimeParent, port); err != nil {
		return nil, err
	}

	url := fmt.Sprintf("http://127.0.0.1:%d", port)
	deadline := time.Now().Add(120 * time.Second)
	for time.Now().Before(deadline) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"/health", nil)
		if err == nil {
			resp, err := http.DefaultClient.Do(req)
			if err == nil {
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				if resp.StatusCode == 200 && healthEmbedBoot(body) == boot {
					mu.Lock()
					baseURL = url
					mu.Unlock()
					return &Client{}, nil
				}
				_ = body
			}
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
	return nil, fmt.Errorf(
		"embedded runtime not healthy at %s within 120s (port conflict or stale listener on :%d? stop other zerollama/runtime processes)",
		url,
		port,
	)
}

// BaseURL returns the loopback URL for the in-process runtime, or "" if not embedded.
func BaseURL() string {
	mu.RLock()
	defer mu.RUnlock()
	return baseURL
}

// SetBaseURLForTest registers a loopback URL as if embed Start succeeded (tests only).
// Why: server/runtime_tokenize tests must cover embed path without CGO uvicorn; production
// uses runtimeworker.BaseURL() when ZEROLLAMA_RUNTIME_URL is unset.
func SetBaseURLForTest(u string) {
	mu.Lock()
	baseURL = strings.TrimSuffix(strings.TrimSpace(u), "/")
	mu.Unlock()
}

// ClearBaseURLForTest clears the test-registered embed URL.
func ClearBaseURLForTest() {
	SetBaseURLForTest("")
}

// Close clears the registered URL (uvicorn thread is daemon; exits with process).
func (c *Client) Close() {
	mu.Lock()
	baseURL = ""
	mu.Unlock()
}

func resolveRepoRoot() string {
	if p := strings.TrimSpace(os.Getenv("OLLAMA_TRAINING_PYTHONPATH")); p != "" {
		c := filepath.Clean(p)
		if hasRuntimePackage(c) {
			return c
		}
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		for _, rel := range []string{"..", "../..", "../../..", "../../../.."} {
			cand := filepath.Clean(filepath.Join(dir, rel))
			if hasRuntimePackage(cand) {
				return cand
			}
		}
	}
	if wd, err := os.Getwd(); err == nil {
		if hasRuntimePackage(wd) {
			return wd
		}
	}
	return ""
}

func hasRuntimePackage(dir string) bool {
	st, err := os.Stat(filepath.Join(dir, "runtime", "runtime", "__init__.py"))
	return err == nil && !st.IsDir()
}
