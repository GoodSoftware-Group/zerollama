package remotestore

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestFetchChunkSkipsHTTPPreamble(t *testing.T) {
	TestingResetPreambleSkip()
	t.Cleanup(TestingResetPreambleSkip)

	payload := make([]byte, 4096)
	for i := range payload {
		payload[i] = byte(i)
	}
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				buf := make([]byte, 4096)
				_, _ = c.Read(buf)
				junk := bytes.Repeat([]byte{0xAB}, 1024-512)
				_, _ = c.Write(junk)
				body := payload[:1024]
				hdr := "HTTP/1.1 206 Partial Content\r\n" +
					"Content-Length: 1024\r\n" +
					"Content-Range: bytes 0-1023/4096\r\n" +
					"Content-Type: application/octet-stream\r\n" +
					"Connection: close\r\n\r\n"
				_, _ = c.Write([]byte(hdr))
				_, _ = c.Write(body)
			}(conn)
		}
	}()

	auth, err := NewAuth("preamble-test-secret")
	if err != nil {
		t.Fatal(err)
	}
	tcp := NewTCPTransport(auth)
	tcp.Client = &http.Client{Timeout: time.Minute}
	preferPreambleSkip.Store(true)

	base := "http://" + ln.Addr().String()
	rc, n, err := tcp.FetchChunk(context.Background(), base, digest, 0, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()
	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1024 {
		t.Fatalf("n=%d want 1024", n)
	}
	if !bytes.Equal(got, payload[:1024]) {
		t.Fatalf("body mismatch len=%d", len(got))
	}
}

func TestReaderAfterHTTPPreamble(t *testing.T) {
	br := bufio.NewReader(io.MultiReader(
		bytes.NewReader([]byte{0x00, 0xAB, 0xCD}),
		bytes.NewReader([]byte("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")),
	))
	out, skipped, err := readerAfterHTTPPreamble(br, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 3 {
		t.Fatalf("skipped=%d", skipped)
	}
	line, err := out.ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	if line != "HTTP/1.1 200 OK\r\n" {
		t.Fatalf("line=%q", line)
	}

	// "HTTP/1." without version digit+space must not match.
	br2 := bufio.NewReader(bytes.NewReader([]byte("xxxHTTP/1.xFOOHTTP/1.0 204 No Content\r\n\r\n")))
	out2, skipped2, err := readerAfterHTTPPreamble(br2, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if skipped2 < 3 {
		t.Fatalf("skipped2=%d", skipped2)
	}
	line2, _ := out2.ReadString('\n')
	if !strings.HasPrefix(line2, "HTTP/1.0 204") {
		t.Fatalf("line2=%q", line2)
	}
}
