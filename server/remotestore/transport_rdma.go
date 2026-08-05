//go:build rdma

package remotestore

import (
	"context"
	"fmt"
	"io"
	"strings"
)

// RDMATransport prefers InfiniBand when the peer advertises RDMA.
//
// Why capability-gated TCP fallback in v1: a full verbs RC/QP data path needs
// more cgo surface; shipping preference + correct bytes today beats claiming
// RDMA while delivering nothing. PreferRDMAThenTCP still selects RDMA-capable
// peers first; FetchChunk may use HTTP Range-GET until QP lands.
type RDMATransport struct {
	Auth *Auth
	TCP  *TCPTransport
}

func NewRDMATransport(auth *Auth) BulkTransport {
	return &RDMATransport{
		Auth: auth,
		TCP:  NewTCPTransport(auth),
	}
}

func (t *RDMATransport) Name() string { return "rdma" }

func (t *RDMATransport) Available(cap Capability) bool {
	if cap.RDMA == nil || strings.TrimSpace(cap.RDMA.GID) == "" {
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
	// Full verbs RC path (cgo libibverbs) is roadmap follow-up; use TCP data plane
	// after RDMA capability selection so PreferRDMAThenTCP still prefers peers that
	// advertise RDMA while delivering correct bytes today.
	if t.TCP == nil {
		return nil, 0, fmt.Errorf("rdma: tcp fallback not configured")
	}
	return t.TCP.FetchChunk(ctx, baseURL, digest, offset, length)
}
