package remotestore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// parallelWorkers caps concurrent chunk downloads.
	parallelWorkers = 4
	// stallTimeout aborts a chunk read that stops making progress.
	stallTimeout = 60 * time.Second
	// peerCooldown skips a dead peer briefly before retrying.
	peerCooldown = 30 * time.Second
	// chunkMaxRetries per Range window.
	chunkMaxRetries = 4
)

// Tunables (vars so tests can shrink windows without multi-MiB fixtures).
var (
	// parallelChunkSize is the Range-GET window for large blob fetches.
	parallelChunkSize int64 = 64 << 20 // 64 MiB
	// parallelMinSize triggers parallel Range-GET instead of a single stream.
	parallelMinSize int64 = 64 << 20
)

// TestingSetParallelSizes shrinks Range-GET windows for unit tests. Restore with the returned func.
func TestingSetParallelSizes(min, chunk int64) (restore func()) {
	oldMin, oldChunk := parallelMinSize, parallelChunkSize
	parallelMinSize, parallelChunkSize = min, chunk
	return func() {
		parallelMinSize, parallelChunkSize = oldMin, oldChunk
	}
}

// downloadParallel fetches digest into dest using parallel HTTP Range-GET when
// size >= parallelMinSize; otherwise a single stream. On hash mismatch the
// .partial file is left in place for inspection (no silent truncate loop).
func (r *Resolver) downloadParallel(ctx context.Context, base, digest, dest string, size int64, cap Capability) (via string, retries int, err error) {
	partial := dest + ".partial"
	_ = os.Remove(partial)

	start := time.Now()
	defer func() {
		if err == nil {
			slog.Info("remotestore fetched blob",
				"digest", digest, "via", via, "server", base,
				"bytes", size, "duration", time.Since(start), "retries", retries)
		}
	}()

	if size <= 0 || size < parallelMinSize {
		via, err = r.downloadSingle(ctx, base, digest, partial, dest, cap)
		return via, retries, err
	}

	// Prefer TCP for parallel ranges (RDMA path is whole-blob oriented in v1).
	tcp := NewTCPTransport(r.Auth)
	tcp.Client = r.chunkHTTPClient()

	f, err := os.OpenFile(partial, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return "tcp", retries, err
	}
	if err := f.Truncate(size); err != nil {
		f.Close()
		return "tcp", retries, err
	}

	type chunk struct {
		off, len int64
	}
	var chunks []chunk
	for off := int64(0); off < size; off += parallelChunkSize {
		n := parallelChunkSize
		if off+n > size {
			n = size - off
		}
		chunks = append(chunks, chunk{off: off, len: n})
	}

	jobs := make(chan chunk, len(chunks))
	for _, c := range chunks {
		jobs <- c
	}
	close(jobs)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		first    error
		retryCnt atomic.Int64
	)
	workers := parallelWorkers
	if workers > len(chunks) {
		workers = len(chunks)
	}
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range jobs {
				if ctx.Err() != nil {
					return
				}
				var last error
				for attempt := 0; attempt < chunkMaxRetries; attempt++ {
					if attempt > 0 {
						retryCnt.Add(1)
						select {
						case <-ctx.Done():
							return
						case <-time.After(time.Duration(attempt) * 200 * time.Millisecond):
						}
					}
					last = r.fetchChunkToFile(ctx, tcp, base, digest, f, c.off, c.len)
					if last == nil {
						break
					}
					if ctx.Err() != nil {
						return
					}
				}
				if last != nil {
					mu.Lock()
					if first == nil {
						first = last
						cancel() // stop siblings; cancels in-flight HTTP bodies
					}
					mu.Unlock()
					return
				}
			}
		}()
	}
	wg.Wait()
	retries = int(retryCnt.Load())
	if err := f.Sync(); err != nil && first == nil {
		first = err
	}
	_ = f.Close()
	if first != nil {
		return "tcp", retries, first
	}
	via = "tcp"
	err = finalizePartialFile(partial, dest, digest)
	return via, retries, err
}

func (r *Resolver) downloadSingle(ctx context.Context, base, digest, partial, dest string, cap Capability) (string, error) {
	ctx, watch := withStallWatch(ctx, stallTimeout)
	defer watch.Stop()

	rc, n, via, err := r.Chain.FetchChunk(ctx, base, digest, 0, 0, cap)
	if err != nil {
		return "", err
	}
	defer rc.Close()
	err = writeAtomicKeepPartial(partial, dest, io.TeeReader(rc, watch), n, digest)
	return via, err
}

func (r *Resolver) fetchChunkToFile(ctx context.Context, tcp *TCPTransport, base, digest string, f *os.File, offset, length int64) error {
	ctx, watch := withStallWatch(ctx, stallTimeout)
	defer watch.Stop()

	rc, n, err := tcp.FetchChunk(ctx, base, digest, offset, length)
	if err != nil {
		return err
	}
	defer rc.Close()

	buf := make([]byte, 1<<20)
	var written int64
	for written < length {
		if err := ctx.Err(); err != nil {
			return err
		}
		want := length - written
		if want > int64(len(buf)) {
			want = int64(len(buf))
		}
		nr, rerr := rc.Read(buf[:want])
		if nr > 0 {
			watch.Bump()
			nw, werr := f.WriteAt(buf[:nr], offset+written)
			written += int64(nw)
			if werr != nil {
				return werr
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return rerr
		}
	}
	if n > 0 && written != n && written != length {
		return fmt.Errorf("chunk short write: got %d want %d", written, length)
	}
	if written != length {
		return fmt.Errorf("chunk short write: got %d want %d", written, length)
	}
	return nil
}

func (r *Resolver) chunkHTTPClient() *http.Client {
	base := r.HTTP
	if base == nil {
		base = &http.Client{}
	}
	tr := http.DefaultTransport
	if base.Transport != nil {
		tr = base.Transport
	}
	return &http.Client{
		Transport:     tr,
		CheckRedirect: base.CheckRedirect,
		Jar:           base.Jar,
		Timeout:       0, // stall watch + request context enforce progress
	}
}

// stallWatch cancels its context if no Bump() arrives within timeout.
// Canceling aborts HTTP body reads — no orphaned Read goroutines.
type stallWatch struct {
	bump func()
	stop func()
	ctx  context.Context
}

func (w *stallWatch) Bump() {
	if w != nil && w.bump != nil {
		w.bump()
	}
}

func (w *stallWatch) Stop() {
	if w != nil && w.stop != nil {
		w.stop()
	}
}

// Write implements io.Writer so TeeReader can bump progress.
func (w *stallWatch) Write(p []byte) (int, error) {
	if w != nil && w.ctx != nil {
		if err := w.ctx.Err(); err != nil {
			return 0, err
		}
	}
	w.Bump()
	return len(p), nil
}

func withStallWatch(parent context.Context, timeout time.Duration) (context.Context, *stallWatch) {
	if timeout <= 0 {
		timeout = stallTimeout
	}
	ctx, cancel := context.WithCancel(parent)
	var mu sync.Mutex
	last := time.Now()
	done := make(chan struct{})
	var once sync.Once
	stop := func() {
		once.Do(func() {
			close(done)
			cancel()
		})
	}
	go func() {
		t := time.NewTicker(time.Second)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-parent.Done():
				cancel()
				return
			case <-t.C:
				mu.Lock()
				stalled := time.Since(last) > timeout
				mu.Unlock()
				if stalled {
					cancel()
					return
				}
			}
		}
	}()
	return ctx, &stallWatch{
		ctx: ctx,
		bump: func() {
			mu.Lock()
			last = time.Now()
			mu.Unlock()
		},
		stop: stop,
	}
}

// writeAtomicKeepPartial is like writeAtomic but leaves .partial on digest mismatch.
func writeAtomicKeepPartial(partial, final string, rc io.Reader, expect int64, digest string) error {
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	h := sha256.New()
	written, err := io.Copy(io.MultiWriter(f, h), rc)
	if syncErr := f.Sync(); syncErr != nil && err == nil {
		err = syncErr
	}
	cerr := f.Close()
	if err != nil {
		_ = os.Remove(partial)
		return err
	}
	if cerr != nil {
		_ = os.Remove(partial)
		return cerr
	}
	if expect > 0 && written != expect {
		_ = os.Remove(partial)
		return fmt.Errorf("short write: got %d want %d", written, expect)
	}
	if digest != "" {
		want := strings.TrimPrefix(normalizeDigest(digest), "sha256-")
		got := hex.EncodeToString(h.Sum(nil))
		if !strings.EqualFold(got, want) {
			return fmt.Errorf("digest mismatch: got %s want %s (partial kept at %s)", got, want, partial)
		}
	}
	if err := os.Rename(partial, final); err != nil {
		return err
	}
	return fsyncFileAndDir(final)
}

func finalizePartialFile(partial, final, digest string) error {
	sum, err := hashFileSHA256(partial)
	if err != nil {
		return err
	}
	want := strings.TrimPrefix(normalizeDigest(digest), "sha256-")
	if !strings.EqualFold(sum, want) {
		return fmt.Errorf("digest mismatch: got %s want %s (partial kept at %s)", sum, want, partial)
	}
	if err := fsyncFile(partial); err != nil {
		return err
	}
	if err := os.Rename(partial, final); err != nil {
		return err
	}
	return fsyncFileAndDir(final)
}

func hashFileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func fsyncFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	err = f.Sync()
	cerr := f.Close()
	if err != nil {
		return err
	}
	return cerr
}

func fsyncFileAndDir(path string) error {
	if err := fsyncFile(path); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	err = dir.Sync()
	cerr := dir.Close()
	if err != nil {
		return err
	}
	return cerr
}

func isSoftFail(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return true
	}
	s := err.Error()
	return strings.Contains(s, "timeout") ||
		strings.Contains(s, "context canceled") ||
		strings.Contains(s, "connection refused") ||
		strings.Contains(s, "connection reset") ||
		strings.Contains(s, "503") ||
		strings.Contains(s, "502") ||
		strings.Contains(s, "504") ||
		strings.Contains(s, "500 ") ||
		strings.HasSuffix(s, "500")
}

func isDigestMismatch(err error) bool {
	return err != nil && strings.Contains(err.Error(), "digest mismatch")
}
