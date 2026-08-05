//go:build !rdma

package remotestore

// NewRDMATransport returns nil when built without the rdma tag.
// Why stub: default builds must not require libibverbs; PreferRDMAThenTCP simply
// omits the RDMA hop and uses TCP.
func NewRDMATransport(auth *Auth) BulkTransport {
	_ = auth
	return nil
}
