//go:build rdma

package remotestore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	id  string
	qp  *verbs.QP
	dev *verbs.Device
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
	cs := &clientSession{id: sessResp.SessionID, qp: qp, dev: t.dev}
	t.mu.Lock()
	t.sessions[baseURL] = cs
	t.mu.Unlock()
	return cs, nil
}

func (t *RDMATransport) readAll(ctx context.Context, baseURL string, sess *clientSession, digest string, offset, length int64, w io.Writer) (int64, error) {
	const window = 64 << 20
	var total int64
	pos := offset
	remaining := length // <=0 means unknown / to EOF

	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		reqLen := int64(window)
		if remaining > 0 && remaining < reqLen {
			reqLen = remaining
		}
		mrResp, err := t.pinMR(ctx, baseURL, sess.id, digest, pos, reqLen)
		if err != nil {
			return total, err
		}
		if mrResp.Length <= 0 {
			t.releaseMR(ctx, baseURL, sess.id, mrResp.MRID)
			break
		}

		localMR, err := sess.dev.AllocMR(int(mrResp.Length), nil, false)
		if err != nil {
			t.releaseMR(ctx, baseURL, sess.id, mrResp.MRID)
			return total, err
		}
		const chunk = 4 << 20
		var off int
		for off < localMR.Len() {
			n := chunk
			if n > localMR.Len()-off {
				n = localMR.Len() - off
			}
			if err := sess.qp.ReadRemote(localMR, off, mrResp.VAddr+uint64(off), mrResp.RKey, n); err != nil {
				localMR.Close()
				t.releaseMR(ctx, baseURL, sess.id, mrResp.MRID)
				return total, err
			}
			off += n
		}
		nw, err := w.Write(localMR.Bytes())
		localMR.Close()
		t.releaseMR(ctx, baseURL, sess.id, mrResp.MRID)

		total += int64(nw)
		if err != nil {
			return total, err
		}
		pos += mrResp.Length
		if remaining > 0 {
			remaining -= mrResp.Length
			if remaining <= 0 {
				break
			}
		} else if mrResp.Length < reqLen {
			// Server returned short window → EOF.
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
