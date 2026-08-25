package remotestore

import "github.com/ollama/ollama/server/remotestore/verbs"

// RDMA control-plane paths (HMAC HTTP). Data plane is one-sided RDMA READ.
const (
	RDMASessionPath = "/v1/rdma/session"
	RDMAMRPath      = "/v1/rdma/mr"
)

// RDMASessionRequest opens an RC QP session. Client sends its endpoint; server
// replies with its own after Connect(client).
type RDMASessionRequest struct {
	Client verbs.Endpoint `json:"client"`
}

// RDMASessionResponse is the server endpoint + session id.
type RDMASessionResponse struct {
	SessionID string         `json:"session_id"`
	Server    verbs.Endpoint `json:"server"`
	// MaxRDAtomic is the responder outstanding RDMA READ depth (max_dest_rd_atomic).
	// Why advertise: clients must not post deeper than the peer allows (WC status 9).
	// Older servers omit this → clients use depth 1.
	MaxRDAtomic int `json:"max_rd_atomic,omitempty"`
}

// RDMAMRRequest asks the server to mmap+reg_mr a blob range for RDMA READ.
type RDMAMRRequest struct {
	SessionID string `json:"session_id"`
	Digest    string `json:"digest"`
	Offset    int64  `json:"offset"`
	Length    int64  `json:"length"` // <=0 → remainder of file
}

// RDMAMRResponse is the remote MR the client READs from.
type RDMAMRResponse struct {
	MRID   string `json:"mr_id"`
	VAddr  uint64 `json:"vaddr"`
	RKey   uint32 `json:"rkey"`
	Length int64  `json:"length"`
	Digest string `json:"digest"`
}
