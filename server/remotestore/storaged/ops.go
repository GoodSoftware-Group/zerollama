package storaged

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// HealthResponse is returned by GET /v1/health.
// Live = process can serve; Ready = live and peer replication is healthy when peers configured.
// OK mirrors Live for simple liveness probes.
type HealthResponse struct {
	OK               bool              `json:"ok"`
	Live             bool              `json:"live"`
	Ready            bool              `json:"ready"`
	ModelsDir        string            `json:"models_dir"`
	BlobCount        int               `json:"blob_count"`
	BlobBytes        int64             `json:"blob_bytes"`
	ManifestCount    int               `json:"manifest_count"`
	PartialUploads   int               `json:"partial_uploads"`
	RDMASessions     int               `json:"rdma_sessions"`
	PeersOK          int               `json:"peers_ok"`
	PeersTotal       int               `json:"peers_total"`
	PendingReplicate int               `json:"pending_replicate"`
	PeerStatus       map[string]string `json:"peer_status,omitempty"`
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := s.Auth.VerifyRequest(r, nil); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	blobs, blobBytes, partials, err := s.scanBlobs()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	manifests, _ := s.listManifestRels("")
	peersOK, peersTotal, peerStatus := s.peerHealthSnapshot()
	live := true
	// Ready requires every configured peer to have succeeded at least once recently
	// (status "ok"). Unknown/error peers make ready=false while live stays true.
	ready := live && (peersTotal == 0 || peersOK == peersTotal)
	resp := HealthResponse{
		OK:               live,
		Live:             live,
		Ready:            ready,
		ModelsDir:        s.ModelsDir,
		BlobCount:        len(blobs),
		BlobBytes:        blobBytes,
		ManifestCount:    len(manifests),
		PartialUploads:   partials,
		RDMASessions:     s.rdmaSessionCount(),
		PeersOK:          peersOK,
		PeersTotal:       peersTotal,
		PendingReplicate: s.pendingReplicate(),
		PeerStatus:       peerStatus,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := s.Auth.VerifyRequest(r, nil); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	blobs, blobBytes, partials, _ := s.scanBlobs()
	var b strings.Builder
	writeGauge := func(name, help string, v float64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", name, help, name, name, v)
	}
	writeCounter := func(name, help string, v int64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n%s %d\n", name, help, name, name, v)
	}
	writeGauge("zerollama_storage_blobs", "Number of complete blobs on disk", float64(len(blobs)))
	writeGauge("zerollama_storage_blob_bytes", "Total bytes of complete blobs", float64(blobBytes))
	writeGauge("zerollama_storage_partial_uploads", "Incomplete .partial uploads", float64(partials))
	writeGauge("zerollama_storage_rdma_sessions", "Active RDMA sessions", float64(s.rdmaSessionCount()))
	writeGauge("zerollama_storage_inflight", "In-flight blob transfers", float64(s.metrics.inFlight.Load()))
	writeGauge("zerollama_storage_pending_replicate", "Queued peer replicate jobs", float64(s.pendingReplicate()))
	writeCounter("zerollama_storage_gets_total", "Completed blob GET/HEAD hits", s.metrics.gets.Load())
	writeCounter("zerollama_storage_puts_total", "Completed blob PUTs", s.metrics.puts.Load())
	writeCounter("zerollama_storage_get_bytes_total", "Bytes served on GET", s.metrics.getBytes.Load())
	writeCounter("zerollama_storage_put_bytes_total", "Bytes accepted on PUT", s.metrics.putBytes.Load())
	writeCounter("zerollama_storage_errors_total", "Handler errors", s.metrics.errors.Load())
	writeCounter("zerollama_storage_replicates_total", "Successful peer replicates", s.metrics.replicates.Load())
	writeCounter("zerollama_storage_replicate_errors_total", "Failed peer replicates", s.metrics.replicateErrs.Load())
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(b.String()))
}

type listResponse struct {
	Items      []string `json:"items"`
	NextCursor string   `json:"next_cursor,omitempty"`
}

func (s *Server) handleBlobsList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := s.Auth.VerifyRequest(r, nil); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	limit := parseLimit(r, 100)
	cursor := r.URL.Query().Get("cursor")
	blobs, _, _, err := s.scanBlobs()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	sort.Strings(blobs)
	start := 0
	if cursor != "" {
		i := sort.SearchStrings(blobs, cursor)
		if i < len(blobs) && blobs[i] == cursor {
			start = i + 1
		} else {
			start = i
		}
	}
	end := start + limit
	if end > len(blobs) {
		end = len(blobs)
	}
	items := blobs[start:end]
	resp := listResponse{Items: items}
	if end < len(blobs) && len(items) > 0 {
		resp.NextCursor = items[len(items)-1]
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleManifestsList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := s.Auth.VerifyRequest(r, nil); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	limit := parseLimit(r, 100)
	cursor := r.URL.Query().Get("cursor")
	all, err := s.listManifestRels("")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	sort.Strings(all)
	start := 0
	if cursor != "" {
		i := sort.SearchStrings(all, cursor)
		if i < len(all) && all[i] == cursor {
			start = i + 1
		} else {
			start = i
		}
	}
	end := start + limit
	if end > len(all) {
		end = len(all)
	}
	items := all[start:end]
	resp := listResponse{Items: items}
	if end < len(all) && len(items) > 0 {
		resp.NextCursor = items[len(items)-1]
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

type verifyRequest struct {
	Digests []string `json:"digests"`
}

type verifyResult struct {
	Digest string `json:"digest"`
	OK     bool   `json:"ok"`
	Error  string `json:"error,omitempty"`
}

type verifyResponse struct {
	Results []verifyResult `json:"results"`
}

func (s *Server) handleVerify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.Auth.VerifyRequest(r, body); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	var req verifyRequest
	if len(body) > 0 {
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
	}
	digests := req.Digests
	if len(digests) == 0 {
		digests, _, _, err = s.scanBlobs()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	out := verifyResponse{Results: make([]verifyResult, 0, len(digests))}
	for _, d := range digests {
		d = strings.ReplaceAll(d, ":", "-")
		if !strings.HasPrefix(d, "sha256-") {
			d = "sha256-" + d
		}
		res := verifyResult{Digest: d}
		path, err := s.blobPath(d)
		if err != nil {
			res.Error = err.Error()
			out.Results = append(out.Results, res)
			continue
		}
		sum, err := hashFileHex(path)
		if err != nil {
			res.Error = err.Error()
			out.Results = append(out.Results, res)
			continue
		}
		want := strings.TrimPrefix(d, "sha256-")
		if !strings.EqualFold(sum, want) {
			res.Error = "digest mismatch"
			out.Results = append(out.Results, res)
			continue
		}
		res.OK = true
		out.Results = append(out.Results, res)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

type gcRequest struct {
	Apply bool `json:"apply"`
}

type gcResponse struct {
	Orphans []string `json:"orphans"`
	Deleted []string `json:"deleted,omitempty"`
	DryRun  bool     `json:"dry_run"`
}

func (s *Server) handleGC(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
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
	var req gcRequest
	if len(body) > 0 {
		_ = json.Unmarshal(body, &req)
	}
	orphans, err := s.findOrphanBlobs()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp := gcResponse{Orphans: orphans, DryRun: !req.Apply}
	if req.Apply {
		for _, d := range orphans {
			path, err := s.blobPath(d)
			if err != nil {
				continue
			}
			if err := os.Remove(path); err == nil {
				resp.Deleted = append(resp.Deleted, d)
			}
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func parseLimit(r *http.Request, def int) int {
	s := r.URL.Query().Get("limit")
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return def
	}
	if n > 1000 {
		return 1000
	}
	return n
}

func (s *Server) scanBlobs() (digests []string, totalBytes int64, partials int, err error) {
	dir := filepath.Join(s.ModelsDir, "blobs")
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, 0, 0, nil
		}
		return nil, 0, 0, err
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".partial") {
			partials++
			continue
		}
		if !blobDigestRe.MatchString(name) {
			continue
		}
		fi, err := e.Info()
		if err != nil {
			continue
		}
		digests = append(digests, name)
		totalBytes += fi.Size()
	}
	return digests, totalBytes, partials, nil
}

func (s *Server) listManifestRels(prefix string) ([]string, error) {
	root := filepath.Join(s.ModelsDir, "manifests")
	var out []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		slash := filepath.ToSlash(rel)
		if len(strings.Split(slash, "/")) != 4 {
			return nil
		}
		if prefix != "" && !strings.HasPrefix(slash, prefix) {
			return nil
		}
		out = append(out, slash)
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	return out, nil
}

func (s *Server) findOrphanBlobs() ([]string, error) {
	blobs, _, _, err := s.scanBlobs()
	if err != nil {
		return nil, err
	}
	refs := map[string]struct{}{}
	manifests, err := s.listManifestRels("")
	if err != nil {
		return nil, err
	}
	for _, rel := range manifests {
		b, err := os.ReadFile(filepath.Join(s.ModelsDir, "manifests", filepath.FromSlash(rel)))
		if err != nil {
			continue
		}
		for _, dig := range extractDigestsFromJSON(string(b)) {
			refs[dig] = struct{}{}
		}
	}
	var orphans []string
	for _, d := range blobs {
		if _, ok := refs[d]; !ok {
			orphans = append(orphans, d)
		}
	}
	sort.Strings(orphans)
	return orphans, nil
}

func extractDigestsFromJSON(s string) []string {
	var out []string
	seen := map[string]struct{}{}
	const needle = "sha256:"
	for {
		i := strings.Index(s, needle)
		if i < 0 {
			break
		}
		rest := s[i:]
		if len(rest) < 7+64 {
			break
		}
		d := rest[:7+64]
		ok := true
		for _, c := range d[7:] {
			if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
				ok = false
				break
			}
		}
		if ok {
			norm := "sha256-" + strings.ToLower(d[7:])
			if _, exists := seen[norm]; !exists {
				seen[norm] = struct{}{}
				out = append(out, norm)
			}
		}
		s = rest[7:]
	}
	return out
}

func hashFileHex(path string) (string, error) {
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
