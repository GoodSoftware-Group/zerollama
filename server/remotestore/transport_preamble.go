package remotestore

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// preferPreambleSkip is set after a peer is seen prepending blob bytes before
// "HTTP/1.1" (rangeLen-512 junk). Subsequent GETs use the skip path directly.
var preferPreambleSkip atomic.Bool

// TestingResetPreambleSkip clears the process-wide preamble-skip latch (tests).
func TestingResetPreambleSkip() { preferPreambleSkip.Store(false) }

func (t *TCPTransport) FetchChunk(ctx context.Context, baseURL, digest string, offset, length int64) (io.ReadCloser, int64, error) {
	digest = normalizeDigest(digest)
	if preferPreambleSkip.Load() {
		return t.fetchChunkSkipPreamble(ctx, baseURL, digest, offset, length)
	}
	url := strings.TrimRight(baseURL, "/") + "/v1/blob/" + digest
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, err
	}
	setBlobRange(req, offset, length)
	if t.Auth != nil {
		if err := t.Auth.SignRequest(req, nil); err != nil {
			return nil, 0, err
		}
	}
	resp, err := t.client().Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "malformed HTTP status") {
			preferPreambleSkip.Store(true)
			slog.Warn("remotestore peer prepends bytes before HTTP status; enabling preamble-skip",
				"server", baseURL, "digest", digest)
			return t.fetchChunkSkipPreamble(ctx, baseURL, digest, offset, length)
		}
		return nil, 0, annotateMalformedHTTP(err)
	}
	return finishBlobResponse(resp, offset, length, digest)
}

func (t *TCPTransport) client() *http.Client {
	if t != nil && t.Client != nil {
		return t.Client
	}
	return &http.Client{Timeout: 30 * time.Minute}
}

func setBlobRange(req *http.Request, offset, length int64) {
	if length > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, offset+length-1))
	} else if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
}

func finishBlobResponse(resp *http.Response, offset, length int64, digest string) (io.ReadCloser, int64, error) {
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, 0, fmt.Errorf("tcp fetch %s: %s: %s", digest, resp.Status, strings.TrimSpace(string(b)))
	}
	n := resp.ContentLength
	if n < 0 {
		if cr := resp.Header.Get("Content-Range"); cr != "" {
			if i := strings.LastIndex(cr, "/"); i >= 0 {
				if tot, err := strconv.ParseInt(cr[i+1:], 10, 64); err == nil && length <= 0 {
					n = tot - offset
				}
			}
		}
	}
	return resp.Body, n, nil
}

// fetchChunkSkipPreamble dials the peer, skips any leading non-HTTP bytes, then
// parses the real response. Why: wedged storage peers send (rangeLen-512) blob
// bytes before "HTTP/1.1 206" on Range GET.
func (t *TCPTransport) fetchChunkSkipPreamble(ctx context.Context, baseURL, digest string, offset, length int64) (io.ReadCloser, int64, error) {
	url := strings.TrimRight(baseURL, "/") + "/v1/blob/" + digest
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, err
	}
	setBlobRange(req, offset, length)
	req.Header.Set("Connection", "close")
	if t.Auth != nil {
		if err := t.Auth.SignRequest(req, nil); err != nil {
			return nil, 0, err
		}
	}

	host, err := blobDialHost(req)
	if err != nil {
		return nil, 0, err
	}

	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", host)
	if err != nil {
		return nil, 0, err
	}
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	closeConn := true
	defer func() {
		stop()
		if closeConn {
			_ = conn.Close()
		}
	}()

	if err := req.Write(conn); err != nil {
		return nil, 0, err
	}

	maxSkip := int64(8 << 30)
	if length > 0 {
		maxSkip = length
	}
	br := bufio.NewReaderSize(conn, 256<<10)
	br, skipped, err := readerAfterHTTPPreamble(br, maxSkip)
	if err != nil {
		return nil, 0, fmt.Errorf("tcp fetch preamble skip: %w", err)
	}
	if skipped > 0 {
		slog.Debug("remotestore skipped HTTP preamble", "bytes", skipped, "digest", digest)
	}
	resp, err := http.ReadResponse(br, req)
	if err != nil {
		return nil, 0, fmt.Errorf("tcp fetch after preamble: %w", err)
	}
	closeConn = false
	resp.Body = &preambleBody{ReadCloser: resp.Body, conn: conn}
	return finishBlobResponse(resp, offset, length, digest)
}

func blobDialHost(req *http.Request) (string, error) {
	host := req.URL.Host
	if host == "" {
		return "", fmt.Errorf("tcp fetch: empty host")
	}
	if req.URL.Port() != "" {
		return host, nil
	}
	if req.URL.Scheme == "https" {
		return net.JoinHostPort(req.URL.Hostname(), "443"), nil
	}
	return net.JoinHostPort(req.URL.Hostname(), "80"), nil
}

type preambleBody struct {
	io.ReadCloser
	conn net.Conn
}

func (b *preambleBody) Close() error {
	err := b.ReadCloser.Close()
	_ = b.conn.Close()
	return err
}

// readerAfterHTTPPreamble discards up to maxSkip bytes until "HTTP/1." then
// returns a reader positioned at that status line.
func readerAfterHTTPPreamble(br *bufio.Reader, maxSkip int64) (*bufio.Reader, int64, error) {
	const needle = "HTTP/1."
	peek, err := br.Peek(len(needle))
	if err == nil && bytes.Equal(peek, []byte(needle)) {
		return br, 0, nil
	}

	var window []byte
	var skipped int64
	for skipped <= maxSkip {
		b, err := br.ReadByte()
		if err != nil {
			return nil, skipped, fmt.Errorf("EOF after skipping %d bytes looking for HTTP status: %w", skipped, err)
		}
		skipped++
		window = append(window, b)
		if len(window) > len(needle) {
			window = window[1:]
		}
		if string(window) == needle {
			// Re-prefix the needle we consumed; remainder stays in br.
			merged := io.MultiReader(bytes.NewReader([]byte(needle)), br)
			return bufio.NewReaderSize(merged, 256<<10), skipped - int64(len(needle)), nil
		}
	}
	return nil, skipped, fmt.Errorf("no HTTP status within %d leading bytes", maxSkip)
}
