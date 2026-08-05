package remotestore

import (
	"context"
	"io"
)

// BulkTransport fetches a byte range of a content-addressed blob.
//
// Why an interface: v1 ships TCP (+ RDMA preference); L2/UDP are roadmap and
// should plug in without rewriting the resolver. Digest may identify a model
// blob, KV slot, or future buffer — keep the wire payload-agnostic.
type BulkTransport interface {
	Name() string
	// FetchChunk returns a reader for [offset, offset+length). length<=0 means to EOF.
	FetchChunk(ctx context.Context, baseURL, digest string, offset, length int64) (io.ReadCloser, int64, error)
	// Available reports whether this transport can talk to the peer given its capability ad.
	Available(cap Capability) bool
}

// TransportChain tries transports in order, falling back on failure.
// Why ordered fallback: prefer IB when the peer and build support it; never
// fail a fetch solely because RDMA negotiation failed.
type TransportChain struct {
	Transports []BulkTransport
}

// PreferRDMAThenTCP builds the v1 chain: RDMA (if built/available) then TCP.
func PreferRDMAThenTCP(auth *Auth) *TransportChain {
	var ts []BulkTransport
	if t := NewRDMATransport(auth); t != nil {
		ts = append(ts, t)
	}
	ts = append(ts, NewTCPTransport(auth))
	return &TransportChain{Transports: ts}
}

// FetchChunk tries each Available transport until one succeeds.
func (c *TransportChain) FetchChunk(ctx context.Context, baseURL, digest string, offset, length int64, cap Capability) (io.ReadCloser, int64, string, error) {
	var last error
	for _, t := range c.Transports {
		if !t.Available(cap) {
			continue
		}
		rc, n, err := t.FetchChunk(ctx, baseURL, digest, offset, length)
		if err == nil {
			return rc, n, t.Name(), nil
		}
		last = err
	}
	if last == nil {
		last = ErrNoTransport
	}
	return nil, 0, "", last
}

// ErrNoTransport means no transport in the chain was available or all failed.
var ErrNoTransport = errString("no available bulk transport")

type errString string

func (e errString) Error() string { return string(e) }
