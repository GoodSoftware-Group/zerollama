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

// TestingPreambleSkipEnabled reports whether preamble-safe mode is latched.
func TestingPreambleSkipEnabled() bool { return preferPreambleSkip.Load() }

func (t *TCPTransport) FetchChunk(ctx context.Context, baseURL, digest string, offset, length int64) (io.ReadCloser, int64, error) {
	digest = normalizeDigest(digest)
	// Ranges ≤512 are correctly framed even on wedged peers — use std Client
	// (keep-alive). Only larger GETs need the raw preamble-skip dial path.
	if preferPreambleSkip.Load() && (length <= 0 || length > preambleSafeRange) {
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
			if preferPreambleSkip.CompareAndSwap(false, true) {
				slog.Warn("remotestore peer prepends bytes before HTTP status; enabling preamble-safe ranges",
					"server", baseURL, "digest", digest)
			}
			if length > 0 && length <= preambleSafeRange {
				// Probe-sized range should not need skip; still try once.
				return t.fetchChunkSkipPreamble(ctx, baseURL, digest, offset, length)
			}
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

// readerAfterHTTPPreamble discards up to maxSkip bytes until a real status
// line ("HTTP/1.0 " / "HTTP/1.1 "), then returns a reader positioned there.
func readerAfterHTTPPreamble(br *bufio.Reader, maxSkip int64) (*bufio.Reader, int64, error) {
	// Require the space after the version so random "HTTP/1." in GGUF junk
	// is less likely to false-match.
	const needle = "HTTP/1."
	peek, err := br.Peek(9)
	if err == nil && len(peek) >= 9 && bytes.HasPrefix(peek, []byte(needle)) && (peek[7] == '0' || peek[7] == '1') && peek[8] == ' ' {
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
		if len(window) > 9 {
			window = window[1:]
		}
		if len(window) == 9 && bytes.HasPrefix(window, []byte(needle)) && (window[7] == '0' || window[7] == '1') && window[8] == ' ' {
			merged := io.MultiReader(bytes.NewReader(append([]byte(nil), window...)), br)
			return bufio.NewReaderSize(merged, 256<<10), skipped - 9, nil
		}
	}
	return nil, skipped, fmt.Errorf("no HTTP status within %d leading bytes", maxSkip)
}
