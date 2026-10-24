package remotestore_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/server/remotestore"
)

// TestFetchThroughWedgedPeer simulates the .39 bug: Range ≤512 is fine; larger
// Ranges return HTTP 206 with Content-Length=rangeLen but close after 512 body
// bytes (unexpected EOF). Resolver must latch preamble-safe 512-byte ranges and
// still materialize the blob.
func TestFetchThroughWedgedPeer(t *testing.T) {
	remotestore.TestingResetPreambleSkip()
	t.Cleanup(remotestore.TestingResetPreambleSkip)

	restore := remotestore.TestingSetParallelSizes(1024, 2048)
	defer restore()

	secret := "wedged-peer-secret"
	auth, err := remotestore.NewAuth(secret)
	if err != nil {
		t.Fatal(err)
	}

	// 8 KiB → parallel path with 2 KiB chunks (would EOF on wedged peer).
	payload := make([]byte, 8192)
	for i := range payload {
		payload[i] = byte(i * 3)
	}
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	srv := &http.Server{Handler: wedgedBlobHandler(auth, digest, payload)}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Close() })

	base := "http://" + ln.Addr().String()
	cache := t.TempDir()
	r := &remotestore.Resolver{
		Servers:  []string{base},
		Auth:     auth,
		CacheDir: cache,
		Chain:    remotestore.PreferRDMAThenTCP(auth),
		HTTP:     &http.Client{Timeout: time.Minute},
	}

	path, err := r.Fetch(context.Background(), digest)
	if err != nil {
		t.Fatalf("Fetch through wedged peer: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("content mismatch: got %d want %d", len(got), len(payload))
	}
	if !remotestore.TestingPreambleSkipEnabled() {
		t.Fatal("expected preamble-safe mode to latch after short-body probe/retry")
	}
}

// wedgedBlobHandler mirrors the broken storage peer:
//   HEAD → correct size
//   GET Range ≤512 → correct 206 body
//   GET Range >512 → 206 with CL=rangeLen but only 512 body bytes then close
func wedgedBlobHandler(auth *remotestore.Auth, digest string, payload []byte) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := auth.VerifyRequest(r, nil); err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		if !strings.Contains(r.URL.Path, digest) {
			http.NotFound(w, r)
			return
		}
		size := int64(len(payload))
		switch r.Method {
		case http.MethodHead:
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
			w.WriteHeader(http.StatusOK)
			return
		case http.MethodGet:
			start, end := int64(0), size-1
			if rg := r.Header.Get("Range"); strings.HasPrefix(rg, "bytes=") {
				_, _ = fmt.Sscanf(rg, "bytes=%d-%d", &start, &end)
			}
			if start < 0 {
				start = 0
			}
			if end >= size {
				end = size - 1
			}
			length := end - start + 1
			if length <= 0 {
				http.Error(w, "bad range", http.StatusRequestedRangeNotSatisfiable)
				return
			}
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
			w.Header().Set("Content-Length", strconv.FormatInt(length, 10))
			w.WriteHeader(http.StatusPartialContent)
			body := payload[start : start+length]
			if length > 512 {
				_, _ = w.Write(body[:512])
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				if hj, ok := w.(http.Hijacker); ok {
					conn, _, err := hj.Hijack()
					if err == nil {
						_ = conn.Close()
					}
				}
				return
			}
			_, _ = io.Copy(w, bytes.NewReader(body))
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})
}
