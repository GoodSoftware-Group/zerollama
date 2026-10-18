package storaged_test

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/storaged"
)

// TestRangeGETFraming ensures Range responses start with HTTP/1.1 (no leading
// blob bytes). A peer that sniffed without rewinding prepended (rangeLen-512)
// bytes before the status line and broke storage run clients.
func TestRangeGETFraming(t *testing.T) {
	secret := "range-cl"
	auth, err := remotestore.NewAuth(secret)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	payload := make([]byte, 8192)
	for i := range payload {
		payload[i] = byte(i)
	}
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	_ = os.MkdirAll(filepath.Join(dir, "blobs"), 0o755)
	_ = os.WriteFile(filepath.Join(dir, "blobs", digest), payload, 0o644)

	srv := storaged.New(dir, auth)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	hs := &http.Server{Handler: srv}
	go hs.Serve(ln)
	t.Cleanup(func() { _ = hs.Close() })
	base := "http://" + ln.Addr().String()

	for _, size := range []int64{64, 512, 513, 1024, 4096} {
		if !rawRangeHTTP(t, auth, base, digest, size) {
			t.Fatalf("size %d: response did not start with HTTP/1.1", size)
		}
	}
}

func rawRangeHTTP(t *testing.T, auth *remotestore.Auth, base, digest string, size int64) bool {
	t.Helper()
	end := size - 1
	req, err := http.NewRequest(http.MethodGet, base+"/v1/blob/"+digest, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", end))
	if err := auth.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	conn, err := net.DialTimeout("tcp", req.URL.Host, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(5 * time.Second))
	fmt.Fprintf(conn, "GET %s HTTP/1.1\r\nHost: %s\r\nRange: bytes=0-%d\r\nConnection: close\r\nAuthorization: %s\r\n\r\n",
		req.URL.RequestURI(), req.URL.Host, end, req.Header.Get("Authorization"))
	buf := make([]byte, 16)
	n, _ := conn.Read(buf)
	_, _ = io.Copy(io.Discard, conn)
	return n >= 5 && string(buf[:5]) == "HTTP/"
}
