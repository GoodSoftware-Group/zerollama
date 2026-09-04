// Package storaged implements the zerollama remote model storage HTTP server.
//
// Why a dedicated serve path (not the inference daemon): storage binds a lab
// port (:18090 by default), serves a possibly huge OLLAMA_MODELS tree, and must
// never compete with production :11434 / :8081 listeners.
package storaged

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/catalog"
)

// blobDigestRe matches only sha256-<64 lowercase hex chars>.
// Why hex-only: a mere "sha256-" prefix allowed path traversal via
// sha256-aa/../../../etc/passwd after filepath.Join cleaned the segments.
var blobDigestRe = regexp.MustCompile(`^sha256-[0-9a-f]{64}$`)

// Server serves an OLLAMA_MODELS-shaped tree over HTTP.
type Server struct {
	ModelsDir string
	Auth      *remotestore.Auth
	Mux       *http.ServeMux
	rdmaServer // build-tagged: real verbs hub or empty stub
}

// New constructs a Server and registers routes.
func New(modelsDir string, auth *remotestore.Auth) *Server {
	s := &Server{
		ModelsDir: modelsDir,
		Auth:      auth,
		Mux:       http.NewServeMux(),
	}
	s.Mux.HandleFunc(remotestore.CapabilityPath, s.handleCapability)
	s.Mux.HandleFunc("/v1/blob/", s.handleBlob)
	s.Mux.HandleFunc("/v1/manifest/", s.handleManifest)
	s.Mux.HandleFunc("/v1/tensor/", s.handleTensor)
	s.initRDMA()
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.Mux.ServeHTTP(w, r)
}

func (s *Server) handleCapability(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := s.Auth.VerifyRequest(r, nil); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	cap := remotestore.Capability{
		Transports: []string{"tcp"},
	}
	if rc := s.rdmaCapability(); rc != nil && rc.GID != "" {
		cap.Transports = append([]string{"rdma"}, cap.Transports...)
		cap.RDMA = rc
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(cap)
}

func (s *Server) blobPath(digest string) (string, error) {
	digest = strings.ReplaceAll(digest, ":", "-")
	if !blobDigestRe.MatchString(digest) {
		return "", fmt.Errorf("invalid digest %q: must be sha256-<64 hex chars>", digest)
	}
	// filepath.Base is redundant after the regex but is kept as a defense-in-depth guard.
	p := filepath.Join(s.ModelsDir, "blobs", filepath.Base(digest))
	if !strings.HasPrefix(filepath.Clean(p), filepath.Clean(filepath.Join(s.ModelsDir, "blobs"))) {
		return "", fmt.Errorf("invalid digest: path escape")
	}
	return p, nil
}

const (
	headerPartial     = "X-Zerollama-Partial"
	headerPartialSize = "X-Zerollama-Partial-Size"
)

func (s *Server) handleBlob(w http.ResponseWriter, r *http.Request) {
	digest := strings.TrimPrefix(r.URL.Path, "/v1/blob/")
	digest = strings.Trim(digest, "/")
	path, err := s.blobPath(digest)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodHead, http.MethodGet:
		if err := s.Auth.VerifyRequest(r, nil); err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		fi, err := os.Stat(path)
		if err == nil {
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Content-Length", strconv.FormatInt(fi.Size(), 10))
			if r.Method == http.MethodHead {
				w.WriteHeader(http.StatusOK)
				return
			}
			http.ServeFile(w, r, path)
			return
		}
		if !os.IsNotExist(err) {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Incomplete range PUT (LA14): HEAD reports partial size so push can resume.
		pfi, perr := os.Stat(path + ".partial")
		if perr != nil {
			if os.IsNotExist(perr) {
				http.NotFound(w, r)
				return
			}
			http.Error(w, perr.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Length", strconv.FormatInt(pfi.Size(), 10))
		w.Header().Set(headerPartial, "1")
		w.Header().Set(headerPartialSize, strconv.FormatInt(pfi.Size(), 10))
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.ServeFile(w, r, path+".partial")

	case http.MethodPut:
		if err := s.Auth.VerifyRequest(r, nil); err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		cr, err := parseContentRange(r.Header.Get("Content-Range"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if cr != nil {
			s.putBlobRange(w, r, path, digest, cr)
			return
		}
		s.putBlobFull(w, r, path, digest)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *Server) putBlobFull(w http.ResponseWriter, r *http.Request, path, digest string) {
	tmp := path + ".partial"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	h := sha256.New()
	_, err = io.Copy(io.MultiWriter(f, h), r.Body)
	cerr := f.Close()
	if err != nil {
		_ = os.Remove(tmp)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if cerr != nil {
		_ = os.Remove(tmp)
		http.Error(w, cerr.Error(), http.StatusInternalServerError)
		return
	}
	if err := finalizeBlobPartial(tmp, path, digest, h); err != nil {
		_ = os.Remove(tmp)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusCreated)
}

func (s *Server) putBlobRange(w http.ResponseWriter, r *http.Request, path, digest string, cr *contentRange) {
	tmp := path + ".partial"
	var current int64
	if fi, err := os.Stat(tmp); err == nil {
		current = fi.Size()
	} else if !os.IsNotExist(err) {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := os.Stat(path); err == nil && cr.start == 0 {
		// Replacing a complete blob: drop it so resume metadata is on .partial.
		_ = os.Remove(path)
	}

	if cr.start == 0 && current > 0 {
		_ = os.Remove(tmp)
		current = 0
	}
	if cr.start != current {
		w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", cr.total))
		w.Header().Set(headerPartialSize, strconv.FormatInt(current, 10))
		http.Error(w, fmt.Sprintf("Content-Range start %d does not match current size %d", cr.start, current), http.StatusRequestedRangeNotSatisfiable)
		return
	}

	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := f.Seek(cr.start, io.SeekStart); err != nil {
		_ = f.Close()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	expect := cr.end - cr.start + 1
	n, err := io.Copy(f, io.LimitReader(r.Body, expect))
	cerr := f.Close()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if cerr != nil {
		http.Error(w, cerr.Error(), http.StatusInternalServerError)
		return
	}
	if n != expect {
		http.Error(w, fmt.Sprintf("short write: got %d want %d", n, expect), http.StatusBadRequest)
		return
	}
	newSize := cr.start + n
	if newSize < cr.total {
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set(headerPartial, "1")
		w.Header().Set(headerPartialSize, strconv.FormatInt(newSize, 10))
		w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", cr.total))
		w.WriteHeader(http.StatusAccepted) // incomplete; not 308 (clients would follow as redirect)
		return
	}

	sum, err := hashFile(tmp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := finalizeBlobPartial(tmp, path, digest, sum); err != nil {
		_ = os.Remove(tmp)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusCreated)
}

func hashFile(path string) (hash hash.Hash, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}
	return h, nil
}

func finalizeBlobPartial(tmp, path, digest string, h hash.Hash) error {
	got := hex.EncodeToString(h.Sum(nil))
	want := strings.TrimPrefix(strings.ReplaceAll(digest, ":", "-"), "sha256-")
	if !strings.EqualFold(got, want) {
		return fmt.Errorf("digest mismatch")
	}
	return os.Rename(tmp, path)
}


func (s *Server) handleManifest(w http.ResponseWriter, r *http.Request) {
	rel := strings.TrimPrefix(r.URL.Path, "/v1/manifest/")
	rel = filepath.Clean("/" + rel)
	rel = strings.TrimPrefix(rel, "/")
	if rel == "" || strings.Contains(rel, "..") {
		http.Error(w, "bad path", http.StatusBadRequest)
		return
	}
	path := filepath.Join(s.ModelsDir, "manifests", filepath.FromSlash(rel))

	switch r.Method {
	case http.MethodGet:
		if err := s.Auth.VerifyRequest(r, nil); err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		b, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				http.NotFound(w, r)
				return
			}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(b)

	case http.MethodPut:
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := s.Auth.VerifyRequest(r, body); err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := os.WriteFile(path, body, 0o644); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusCreated)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleTensor resolves GET /v1/tensor/{host}/{ns}/{model}/{tag}/{tensor_ref}
// where tensor_ref is a module-role or raw GGUF tensor name.
//
// Why server-side catalog resolve: clients should ask for layer.3.attn without
// re-implementing GGUF offset math; byte-range /v1/blob remains the low-level primitive.
func (s *Server) handleTensor(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := s.Auth.VerifyRequest(r, nil); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	rest := strings.TrimPrefix(r.URL.Path, "/v1/tensor/")
	parts := strings.Split(rest, "/")
	if len(parts) < 5 {
		http.Error(w, "path must be /v1/tensor/{host}/{ns}/{model}/{tag}/{tensor_ref}", http.StatusBadRequest)
		return
	}
	host, ns, model, tag := parts[0], parts[1], parts[2], parts[3]
	tensorRef := strings.Join(parts[4:], "/")
	for _, seg := range []string{host, ns, model, tag} {
		if seg == "" || strings.Contains(seg, "..") || strings.ContainsAny(seg, "/\\") {
			http.Error(w, "bad path component", http.StatusBadRequest)
			return
		}
	}
	manifestPath := filepath.Join(s.ModelsDir, "manifests", host, ns, model, tag)
	// Ensure the resolved path stays within the manifests tree.
	if !strings.HasPrefix(filepath.Clean(manifestPath), filepath.Clean(filepath.Join(s.ModelsDir, "manifests"))) {
		http.Error(w, "bad path", http.StatusBadRequest)
		return
	}
	mfBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	var mf struct {
		Layers []struct {
			Digest    string `json:"digest"`
			MediaType string `json:"mediaType"`
		} `json:"layers"`
	}
	if err := json.Unmarshal(mfBytes, &mf); err != nil {
		http.Error(w, "bad manifest", http.StatusInternalServerError)
		return
	}
	var modelDigest string
	for _, l := range mf.Layers {
		if l.MediaType == "application/vnd.ollama.image.model" {
			modelDigest = l.Digest
			break
		}
	}
	if modelDigest == "" {
		http.Error(w, "no model layer", http.StatusNotFound)
		return
	}
	blobPath, err := s.blobPath(modelDigest)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	entry, err := catalog.LookupGGUF(blobPath, normalizeDigestLocal(modelDigest), tensorRef)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	f, err := os.Open(blobPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()
	if _, err := f.Seek(int64(entry.Offset), io.SeekStart); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.FormatInt(entry.Length, 10))
	w.Header().Set("X-Zerollama-Tensor-Name", entry.Name)
	w.Header().Set("X-Zerollama-Module-Role", string(entry.Role))
	_, _ = io.CopyN(w, f, entry.Length)
}

func normalizeDigestLocal(d string) string {
	d = strings.ReplaceAll(d, ":", "-")
	if !strings.HasPrefix(d, "sha256-") {
		d = "sha256-" + d
	}
	return d
}
