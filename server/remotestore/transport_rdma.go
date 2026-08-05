//go:build rdma

package remotestore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/server/remotestore/verbs"
)

// RDMATransport fetches blob ranges via one-sided RDMA READ.
//
// Why first-class: IB/RoCE beats TCP for bulk weights. Control plane (session +
// MR lease) stays HMAC HTTP; bytes move with IBV_WR_RDMA_READ. Falls back to
// TCP only when the peer lacks verbs:true or QP setup fails.
type RDMATransport struct {
	Auth *Auth
	TCP  *TCPTransport

	mu       sync.Mutex
	dev      *verbs.Device
	sessions map[string]*clientSession // baseURL → session
}

type clientSession struct {
	id          string
	qp          *verbs.QP
	dev         *verbs.Device
	maxRDAtomic int // peer responder depth; 0/omit → 1
}

// NewRDMATransport returns an RDMA transport when an HCA can be opened.
func NewRDMATransport(auth *Auth) BulkTransport {
	dev, err := verbs.OpenFirst()
	if err != nil {
		// No local HCA — PreferRDMAThenTCP simply skips us (Available false
		// without verbs), but keep a stub that reports unavailable.
		return &RDMATransport{Auth: auth, TCP: NewTCPTransport(auth), sessions: make(map[string]*clientSession)}
	}
	return &RDMATransport{
		Auth:     auth,
		TCP:      NewTCPTransport(auth),
		dev:      dev,
		sessions: make(map[string]*clientSession),
	}
}

func (t *RDMATransport) Name() string { return "rdma" }

func (t *RDMATransport) Available(cap Capability) bool {
	if t == nil || t.dev == nil {
		return false
	}
	// Why require Verbs: older servers advertised GID without a QP data plane.
	if cap.RDMA == nil || !cap.RDMA.Verbs || strings.TrimSpace(cap.RDMA.GID) == "" {
		return false
	}
	for _, n := range cap.Transports {
		if strings.EqualFold(n, "rdma") {
			return true
		}
	}
	return false
}

func (t *RDMATransport) FetchChunk(ctx context.Context, baseURL, digest string, offset, length int64) (io.ReadCloser, int64, error) {
	if t.dev == nil {
		return nil, 0, fmt.Errorf("rdma: no local HCA")
	}
	sess, err := t.ensureSession(ctx, baseURL)
	if err != nil {
		if t.TCP != nil {
			return t.TCP.FetchChunk(ctx, baseURL, digest, offset, length)
		}
		return nil, 0, err
	}

	pr, pw := io.Pipe()
	go func() {
		n, err := t.readAll(ctx, baseURL, sess, digest, offset, length, pw)
		if err != nil {
			_ = pw.CloseWithError(err)
			return
		}
		_ = n
		_ = pw.Close()
	}()
	// Length unknown until first MR if length<=0; resolver accepts 0 as EOF.
	return pr, length, nil
}

func (t *RDMATransport) ensureSession(ctx context.Context, baseURL string) (*clientSession, error) {
	t.mu.Lock()
	if s := t.sessions[baseURL]; s != nil {
		t.mu.Unlock()
		return s, nil
	}
	t.mu.Unlock()

	qp, err := t.dev.CreateQP()
	if err != nil {
		return nil, err
	}
	reqBody, _ := json.Marshal(RDMASessionRequest{Client: qp.Endpoint()})
	url := strings.TrimRight(baseURL, "/") + RDMASessionPath
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		qp.Close()
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if err := t.Auth.SignRequest(httpReq, reqBody); err != nil {
		qp.Close()
		return nil, err
	}
	client := t.TCP.Client
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		qp.Close()
		return nil, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		qp.Close()
		return nil, fmt.Errorf("rdma session: %s: %s", resp.Status, b)
	}
	var sessResp RDMASessionResponse
	if err := json.Unmarshal(b, &sessResp); err != nil {
		qp.Close()
		return nil, err
	}
	if err := qp.Connect(sessResp.Server); err != nil {
		qp.Close()
		return nil, fmt.Errorf("rdma client connect: %w", err)
	}
	cs := &clientSession{id: sessResp.SessionID, qp: qp, dev: t.dev, maxRDAtomic: sessResp.MaxRDAtomic}
	if cs.maxRDAtomic <= 0 {
		cs.maxRDAtomic = 1 // old peers omit the field
	}
	t.mu.Lock()
	t.sessions[baseURL] = cs
	t.mu.Unlock()
	return cs, nil
}

func (t *RDMATransport) readAll(ctx context.Context, baseURL string, sess *clientSession, digest string, offset, length int64, w io.Writer) (int64, error) {
	// Larger windows cut HTTP MR-lease RTTs; pipeline READs + pin prefetch fill the IB link.
	const window = 128 << 20
	const chunk = 2 << 20
	depth := sess.maxRDAtomic
	if depth <= 0 {
		depth = 1
	}
	if depth > 16 {
		depth = 16
	}
	var total int64
	pos := offset
	remaining := length // <=0 means unknown / to EOF

	// Reuse one local MR — ibv_reg_mr of multi-hundred-MiB windows was a major tax.
	localMR, err := sess.dev.AllocMR(window, nil, false)
	if err != nil {
		return 0, err
	}
	defer localMR.Close()

	type pinned struct {
		resp   *RDMAMRResponse
		reqLen int64
		pinMs  int64
		err    error
	}
	pinNext := func(off, rem int64) *pinned {
		reqLen := int64(window)
		if rem > 0 && rem < reqLen {
			reqLen = rem
		}
		t0 := time.Now()
		resp, err := t.pinMR(ctx, baseURL, sess.id, digest, off, reqLen)
		return &pinned{resp: resp, reqLen: reqLen, pinMs: time.Since(t0).Milliseconds(), err: err}
	}

	cur := pinNext(pos, remaining)
	for {
		if err := ctx.Err(); err != nil {
			if cur != nil && cur.resp != nil && cur.resp.MRID != "" {
				go t.releaseMR(context.Background(), baseURL, sess.id, cur.resp.MRID)
			}
			return total, err
		}
		if cur.err != nil {
			return total, cur.err
		}
		if cur.resp == nil || cur.resp.Length <= 0 {
			if cur.resp != nil && cur.resp.MRID != "" {
				t.releaseMR(ctx, baseURL, sess.id, cur.resp.MRID)
			}
			break
		}
		if int(cur.resp.Length) > localMR.Len() {
			t.releaseMR(ctx, baseURL, sess.id, cur.resp.MRID)
			return total, fmt.Errorf("rdma window %d exceeds local mr %d", cur.resp.Length, localMR.Len())
		}

		nextPos := pos + cur.resp.Length
		nextRem := remaining
		if remaining > 0 {
			nextRem = remaining - cur.resp.Length
		}
		var nextCh chan *pinned
		if remaining <= 0 || nextRem > 0 {
			nextCh = make(chan *pinned, 1)
			go func(off, rem int64) { nextCh <- pinNext(off, rem) }(nextPos, nextRem)
		}

		tRead := time.Now()
		err = sess.qp.ReadRemotePipeline(localMR, cur.resp.VAddr, cur.resp.RKey, int(cur.resp.Length), chunk, depth)
		readMs := time.Since(tRead).Milliseconds()
		if err != nil {
			t.releaseMR(ctx, baseURL, sess.id, cur.resp.MRID)
			if nextCh != nil {
				n := <-nextCh
				if n.resp != nil && n.resp.MRID != "" {
					go t.releaseMR(context.Background(), baseURL, sess.id, n.resp.MRID)
				}
			}
			return total, err
		}
		nw, err := w.Write(localMR.Bytes()[:cur.resp.Length])
		go t.releaseMR(context.Background(), baseURL, sess.id, cur.resp.MRID)

		if cur.pinMs+readMs > 50 {
			slog.Debug("remotestore rdma window", "bytes", cur.resp.Length, "pin_ms", cur.pinMs, "read_ms", readMs)
		}

		total += int64(nw)
		if err != nil {
			if nextCh != nil {
				n := <-nextCh
				if n.resp != nil && n.resp.MRID != "" {
					go t.releaseMR(context.Background(), baseURL, sess.id, n.resp.MRID)
				}
			}
			return total, err
		}
		pos = nextPos
		if remaining > 0 {
			remaining = nextRem
			if remaining <= 0 {
				if nextCh != nil {
					n := <-nextCh
					if n.resp != nil && n.resp.MRID != "" {
						go t.releaseMR(context.Background(), baseURL, sess.id, n.resp.MRID)
					}
				}
				break
			}
		} else if cur.resp.Length < cur.reqLen {
			if nextCh != nil {
				n := <-nextCh
				if n.resp != nil && n.resp.MRID != "" {
					go t.releaseMR(context.Background(), baseURL, sess.id, n.resp.MRID)
				}
			}
			break
		}

		if nextCh != nil {
			cur = <-nextCh
		} else {
			break
		}
	}
	return total, nil
}

func (t *RDMATransport) pinMR(ctx context.Context, baseURL, sessionID, digest string, offset, length int64) (*RDMAMRResponse, error) {
	reqBody, _ := json.Marshal(RDMAMRRequest{SessionID: sessionID, Digest: digest, Offset: offset, Length: length})
	url := strings.TrimRight(baseURL, "/") + RDMAMRPath
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if err := t.Auth.SignRequest(httpReq, reqBody); err != nil {
		return nil, err
	}
	client := t.TCP.Client
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute}
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rdma mr: %s: %s", resp.Status, b)
	}
	var out RDMAMRResponse
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (t *RDMATransport) releaseMR(ctx context.Context, baseURL, sessionID, mrID string) {
	reqBody, _ := json.Marshal(map[string]string{"session_id": sessionID, "mr_id": mrID})
	url := strings.TrimRight(baseURL, "/") + RDMAMRPath
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, bytes.NewReader(reqBody))
	if err != nil {
		return
	}
	_ = t.Auth.SignRequest(httpReq, reqBody)
	client := t.TCP.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	resp, err := client.Do(httpReq)
	if err == nil {
		resp.Body.Close()
	}
}
