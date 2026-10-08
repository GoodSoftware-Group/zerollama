package storaged

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/server/remotestore"
)

type replicateJob struct {
	kind   string // "blob" or "manifest"
	digest string // blob digest
	rel    string // manifest rel host/ns/model/tag
}

// StartReplicator begins the async peer replicate worker and background reconciler.
// Safe to call once after Peers are set.
func (s *Server) StartReplicator(ctx context.Context) {
	if s == nil || len(s.Peers) == 0 {
		return
	}
	s.replOnce.Do(func() {
		s.replWake = make(chan struct{}, 1)
		s.peerStatus = make(map[string]string)
		if s.HTTP == nil {
			s.HTTP = &http.Client{Timeout: 60 * time.Minute}
		}
		go s.replicateWorker(ctx)
		go s.reconcileLoop(ctx)
	})
}

func (s *Server) enqueueReplicateBlob(digest string) {
	if s == nil || len(s.Peers) == 0 || s.replWake == nil {
		return
	}
	s.pushJob(replicateJob{kind: "blob", digest: digest})
}

func (s *Server) enqueueReplicateManifest(rel string) {
	if s == nil || len(s.Peers) == 0 || s.replWake == nil {
		return
	}
	s.pushJob(replicateJob{kind: "manifest", rel: rel})
}

func (s *Server) pushJob(job replicateJob) {
	s.replMu.Lock()
	s.replQ = append(s.replQ, job)
	s.replMu.Unlock()
	select {
	case s.replWake <- struct{}{}:
	default:
	}
}

func (s *Server) pendingReplicate() int {
	if s == nil {
		return 0
	}
	s.replMu.Lock()
	n := len(s.replQ)
	s.replMu.Unlock()
	return n
}

func (s *Server) popJobs() []replicateJob {
	s.replMu.Lock()
	defer s.replMu.Unlock()
	if len(s.replQ) == 0 {
		return nil
	}
	out := s.replQ
	s.replQ = nil
	return out
}

func (s *Server) peerHealthSnapshot() (ok, total int, status map[string]string) {
	if s == nil {
		return 0, 0, nil
	}
	s.peerMu.Lock()
	defer s.peerMu.Unlock()
	total = len(s.Peers)
	status = make(map[string]string, len(s.peerStatus))
	for _, p := range s.Peers {
		st := s.peerStatus[p]
		if st == "" {
			st = "unknown"
		}
		status[p] = st
		if st == "ok" {
			ok++
		}
	}
	return ok, total, status
}

func (s *Server) setPeerStatus(peer, st string) {
	s.peerMu.Lock()
	if s.peerStatus == nil {
		s.peerStatus = make(map[string]string)
	}
	s.peerStatus[peer] = st
	s.peerMu.Unlock()
}

func (s *Server) replicateWorker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.replWake:
			for {
				jobs := s.popJobs()
				if len(jobs) == 0 {
					break
				}
				// Blobs before manifests so peers never advertise missing layers.
				for _, job := range jobs {
					if job.kind == "blob" {
						s.doReplicateJob(ctx, job)
					}
				}
				for _, job := range jobs {
					if job.kind == "manifest" {
						s.doReplicateJob(ctx, job)
					}
				}
			}
		}
	}
}

func (s *Server) doReplicateJob(ctx context.Context, job replicateJob) {
	for _, peer := range s.Peers {
		var err error
		switch job.kind {
		case "blob":
			err = s.replicateBlobTo(ctx, peer, job.digest)
		case "manifest":
			// Ensure referenced local blobs land on the peer before the manifest.
			err = s.replicateManifestWithBlobs(ctx, peer, job.rel)
		}
		if err != nil {
			s.metrics.replicateErrs.Add(1)
			s.setPeerStatus(peer, "error")
			slog.Warn("storaged replicate failed", "peer", peer, "kind", job.kind, "error", err)
			continue
		}
		s.metrics.replicates.Add(1)
		s.setPeerStatus(peer, "ok")
	}
}

func (s *Server) replicateManifestWithBlobs(ctx context.Context, peer, rel string) error {
	path := filepath.Join(s.ModelsDir, "manifests", filepath.FromSlash(rel))
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	for _, dig := range extractDigestsFromJSON(string(data)) {
		bp, err := s.blobPath(dig)
		if err != nil {
			continue
		}
		if _, err := os.Stat(bp); err != nil {
			continue
		}
		if err := s.replicateBlobTo(ctx, peer, dig); err != nil {
			return err
		}
	}
	return s.replicateManifestTo(ctx, peer, rel)
}

func (s *Server) replicateBlobTo(ctx context.Context, peer, digest string) error {
	digest = strings.ReplaceAll(digest, ":", "-")
	if !strings.HasPrefix(digest, "sha256-") {
		digest = "sha256-" + digest
	}
	tcp := remotestore.NewTCPTransport(s.Auth)
	tcp.Client = s.HTTP
	_, ok, err := tcp.HeadBlob(ctx, peer, digest)
	if err == nil && ok {
		return nil
	}
	path, err := s.blobPath(digest)
	if err != nil {
		return err
	}
	if _, err := os.Stat(path); err != nil {
		return err
	}
	return remotestore.PushBlob(ctx, s.Auth, s.HTTP, peer, digest, path)
}

func (s *Server) replicateManifestTo(ctx context.Context, peer, rel string) error {
	path := filepath.Join(s.ModelsDir, "manifests", filepath.FromSlash(rel))
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	parts := strings.Split(rel, "/")
	if len(parts) != 4 {
		return nil
	}
	return remotestore.PushManifest(ctx, s.Auth, s.HTTP, peer, parts[0], parts[1], parts[2], parts[3], data)
}

func (s *Server) reconcileLoop(ctx context.Context) {
	t := time.NewTicker(5 * time.Minute)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return
	case <-time.After(15 * time.Second):
		if err := s.Reconcile(ctx); err != nil {
			slog.Warn("storaged reconcile", "error", err)
		}
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := s.Reconcile(ctx); err != nil {
				slog.Warn("storaged reconcile", "error", err)
			}
		}
	}
}

// Reconcile pushes local blobs then manifests to peers (direct, not via the
// bounded event queue), and pulls digests referenced by local manifests that
// are missing locally.
func (s *Server) Reconcile(ctx context.Context) error {
	if s == nil || len(s.Peers) == 0 {
		return nil
	}
	blobs, _, _, err := s.scanBlobs()
	if err != nil {
		return err
	}
	for _, d := range blobs {
		if err := ctx.Err(); err != nil {
			return err
		}
		for _, peer := range s.Peers {
			if err := s.replicateBlobTo(ctx, peer, d); err != nil {
				s.setPeerStatus(peer, "error")
				slog.Warn("storaged reconcile blob", "peer", peer, "digest", d, "error", err)
				continue
			}
			s.setPeerStatus(peer, "ok")
		}
	}
	manifests, err := s.listManifestRels("")
	if err != nil {
		return err
	}
	for _, rel := range manifests {
		if err := ctx.Err(); err != nil {
			return err
		}
		for _, peer := range s.Peers {
			if err := s.replicateManifestWithBlobs(ctx, peer, rel); err != nil {
				s.setPeerStatus(peer, "error")
				slog.Warn("storaged reconcile manifest", "peer", peer, "manifest", rel, "error", err)
				continue
			}
			s.setPeerStatus(peer, "ok")
		}
		b, err := os.ReadFile(filepath.Join(s.ModelsDir, "manifests", filepath.FromSlash(rel)))
		if err != nil {
			continue
		}
		for _, dig := range extractDigestsFromJSON(string(b)) {
			path, err := s.blobPath(dig)
			if err != nil {
				continue
			}
			if _, err := os.Stat(path); err == nil {
				continue
			}
			_ = s.pullBlobFromPeers(ctx, dig, path)
		}
	}
	return nil
}

func (s *Server) pullBlobFromPeers(ctx context.Context, digest, dest string) error {
	partial := dest + ".partial"
	for _, peer := range s.Peers {
		tcp := remotestore.NewTCPTransport(s.Auth)
		tcp.Client = s.HTTP
		size, ok, err := tcp.HeadBlob(ctx, peer, digest)
		if err != nil || !ok {
			continue
		}
		rc, n, err := tcp.FetchChunk(ctx, peer, digest, 0, 0)
		if err != nil {
			continue
		}
		if n <= 0 {
			n = size
		}
		err = writeAtomicLocal(partial, dest, rc, n, digest)
		rc.Close()
		if err == nil {
			s.setPeerStatus(peer, "ok")
			return nil
		}
		s.setPeerStatus(peer, "error")
	}
	return os.ErrNotExist
}

// SyncToPeer performs a one-shot push of all local blobs and manifests to peer.
func (s *Server) SyncToPeer(ctx context.Context, peer string) error {
	peer = strings.TrimRight(peer, "/")
	blobs, _, _, err := s.scanBlobs()
	if err != nil {
		return err
	}
	for _, d := range blobs {
		if err := s.replicateBlobTo(ctx, peer, d); err != nil {
			return err
		}
	}
	manifests, err := s.listManifestRels("")
	if err != nil {
		return err
	}
	for _, rel := range manifests {
		if err := s.replicateManifestWithBlobs(ctx, peer, rel); err != nil {
			return err
		}
	}
	return nil
}
