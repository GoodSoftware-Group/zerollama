//go:build rdma

package storaged

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/verbs"
)

type rdmaServer struct {
	hub *rdmaHub
}

type rdmaHub struct {
	dev  *verbs.Device
	mu   sync.Mutex
	sess map[string]*rdmaSession
}

type rdmaSession struct {
	qp      *verbs.QP
	created time.Time
	mrs     map[string]*rdmaMR
}

type rdmaMR struct {
	mr     *verbs.MR
	digest string
}

func (s *Server) initRDMA() {
	dev, err := verbs.OpenFirst()
	if err != nil {
		slog.Info("storaged RDMA verbs unavailable", "error", err)
		return
	}
	s.hub = &rdmaHub{dev: dev, sess: make(map[string]*rdmaSession)}
	s.Mux.HandleFunc(remotestore.RDMASessionPath, s.handleRDMASession)
	s.Mux.HandleFunc(remotestore.RDMAMRPath, s.handleRDMAMR)
	device, gid, gidIdx, port, lid := dev.ProbeCap()
	slog.Info("storaged RDMA verbs ready", "device", device, "gid", gid, "gid_index", gidIdx, "port", port, "lid", lid)
}

func (s *Server) rdmaCapability() *remotestore.RDMACap {
	if s.hub == nil || s.hub.dev == nil {
		return probeSysfsRDMACap()
	}
	device, gid, gidIdx, port, lid := s.hub.dev.ProbeCap()
	return &remotestore.RDMACap{
		Device: device, GID: gid, GIDIndex: gidIdx, Port: port, LID: lid, Verbs: true,
	}
}

func (s *Server) handleRDMASession(w http.ResponseWriter, r *http.Request) {
	if s.hub == nil {
		http.Error(w, "rdma verbs not available", http.StatusServiceUnavailable)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.Auth.VerifyRequest(r, body); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req remotestore.RDMASessionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	qp, err := s.hub.dev.CreateQP()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := qp.Connect(req.Client); err != nil {
		qp.Close()
		http.Error(w, "qp connect: "+err.Error(), http.StatusBadRequest)
		return
	}
	id := randomID()
	s.hub.mu.Lock()
	s.hub.sess[id] = &rdmaSession{qp: qp, created: time.Now(), mrs: make(map[string]*rdmaMR)}
	s.hub.mu.Unlock()

	resp := remotestore.RDMASessionResponse{SessionID: id, Server: qp.Endpoint()}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleRDMAMR(w http.ResponseWriter, r *http.Request) {
	if s.hub == nil {
		http.Error(w, "rdma verbs not available", http.StatusServiceUnavailable)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.Auth.VerifyRequest(r, body); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	switch r.Method {
	case http.MethodPost:
		var req remotestore.RDMAMRRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		resp, err := s.pinMR(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)

	case http.MethodDelete:
		var req struct {
			SessionID string `json:"session_id"`
			MRID      string `json:"mr_id"`
		}
		_ = json.Unmarshal(body, &req)
		s.releaseMR(req.SessionID, req.MRID)
		w.WriteHeader(http.StatusNoContent)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *Server) pinMR(req remotestore.RDMAMRRequest) (*remotestore.RDMAMRResponse, error) {
	path, err := s.blobPath(req.Digest)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := fi.Size()
	if req.Offset < 0 || req.Offset > size {
		f.Close()
		return nil, fmt.Errorf("bad offset")
	}
	if req.Offset == size {
		f.Close()
		return &remotestore.RDMAMRResponse{Digest: req.Digest, Length: 0}, nil
	}
	length := req.Length
	if length <= 0 {
		length = size - req.Offset
	}
	if req.Offset+length > size {
		length = size - req.Offset // clamp to EOF
	}
	const maxMR = 512 << 20
	if length > maxMR {
		length = maxMR
	}

	// Why AllocMR (C heap): Go slices move; mlx4 has no ODP for file mmap MRs.
	buf := make([]byte, int(length))
	if _, err := f.ReadAt(buf, req.Offset); err != nil {
		f.Close()
		return nil, err
	}
	_ = f.Close()

	mr, err := s.hub.dev.AllocMR(int(length), buf, true)
	if err != nil {
		return nil, err
	}

	s.hub.mu.Lock()
	sess := s.hub.sess[req.SessionID]
	if sess == nil {
		s.hub.mu.Unlock()
		mr.Close()
		return nil, fmt.Errorf("unknown session")
	}
	id := randomID()
	sess.mrs[id] = &rdmaMR{mr: mr, digest: req.Digest}
	s.hub.mu.Unlock()

	return &remotestore.RDMAMRResponse{
		MRID: id, VAddr: mr.Addr(), RKey: mr.RKey(), Length: length, Digest: req.Digest,
	}, nil
}

func (s *Server) releaseMR(sessionID, mrID string) {
	s.hub.mu.Lock()
	defer s.hub.mu.Unlock()
	sess := s.hub.sess[sessionID]
	if sess == nil {
		return
	}
	m := sess.mrs[mrID]
	if m == nil {
		return
	}
	delete(sess.mrs, mrID)
	m.mr.Close()
}

func randomID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func probeSysfsRDMACap() *remotestore.RDMACap {
	ents, err := os.ReadDir("/sys/class/infiniband")
	if err != nil {
		return nil
	}
	for _, dev := range ents {
		gidPath := fmt.Sprintf("/sys/class/infiniband/%s/ports/1/gids/0", dev.Name())
		b, err := os.ReadFile(gidPath)
		if err != nil {
			continue
		}
		gid := string(b)
		for len(gid) > 0 && (gid[len(gid)-1] == '\n' || gid[len(gid)-1] == ' ') {
			gid = gid[:len(gid)-1]
		}
		if gid == "" || gid == "0000:0000:0000:0000:0000:0000:0000:0000" {
			continue
		}
		var lid uint16
		if lb, err := os.ReadFile(fmt.Sprintf("/sys/class/infiniband/%s/ports/1/lid", dev.Name())); err == nil {
			var v uint64
			fmt.Sscanf(string(lb), "0x%x", &v)
			lid = uint16(v)
		}
		return &remotestore.RDMACap{Device: dev.Name(), GID: gid, GIDIndex: 0, Port: 1, LID: lid, Verbs: false}
	}
	return nil
}
