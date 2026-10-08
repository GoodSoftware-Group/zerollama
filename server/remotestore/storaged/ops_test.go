package storaged

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/server/remotestore"
)

func signed(t *testing.T, auth *remotestore.Auth, req *http.Request, body []byte) {
	t.Helper()
	if err := auth.SignRequest(req, body); err != nil {
		t.Fatal(err)
	}
}

func TestHealthAndMetrics(t *testing.T) {
	dir := t.TempDir()
	auth := mustAuth(t)
	s := New(dir, auth)

	payload := []byte("health-blob-payload")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	_ = os.MkdirAll(filepath.Join(dir, "blobs"), 0o755)
	_ = os.WriteFile(filepath.Join(dir, "blobs", digest), payload, 0o644)

	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	signed(t, auth, req, nil)
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("health %d: %s", rr.Code, rr.Body.String())
	}
	var h HealthResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &h); err != nil {
		t.Fatal(err)
	}
	if !h.OK || !h.Live || !h.Ready || h.BlobCount != 1 {
		t.Fatalf("health %+v", h)
	}

	// With peers configured but never contacted → live but not ready.
	s.Peers = []string{"http://127.0.0.1:1"}
	req = httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	signed(t, auth, req, nil)
	rr = httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	_ = json.Unmarshal(rr.Body.Bytes(), &h)
	if !h.Live || h.Ready {
		t.Fatalf("want live&&!ready with unknown peers, got %+v", h)
	}

	req = httptest.NewRequest(http.MethodGet, "/v1/metrics", nil)
	signed(t, auth, req, nil)
	rr = httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("metrics %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "zerollama_storage_blobs") {
		t.Fatalf("metrics body: %s", rr.Body.String())
	}
}

func TestListBlobsPagination(t *testing.T) {
	dir := t.TempDir()
	auth := mustAuth(t)
	s := New(dir, auth)
	_ = os.MkdirAll(filepath.Join(dir, "blobs"), 0o755)
	var digests []string
	for i := 0; i < 3; i++ {
		payload := []byte{byte(i)}
		sum := sha256.Sum256(payload)
		d := "sha256-" + hex.EncodeToString(sum[:])
		digests = append(digests, d)
		_ = os.WriteFile(filepath.Join(dir, "blobs", d), payload, 0o644)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/blobs?limit=2", nil)
	signed(t, auth, req, nil)
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatal(rr.Body.String())
	}
	var page1 listResponse
	_ = json.Unmarshal(rr.Body.Bytes(), &page1)
	if len(page1.Items) != 2 || page1.NextCursor == "" {
		t.Fatalf("page1 %+v", page1)
	}
	req = httptest.NewRequest(http.MethodGet, "/v1/blobs?limit=2&cursor="+page1.NextCursor, nil)
	signed(t, auth, req, nil)
	rr = httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	var page2 listResponse
	_ = json.Unmarshal(rr.Body.Bytes(), &page2)
	if len(page2.Items) != 1 {
		t.Fatalf("page2 %+v", page2)
	}
}

func TestGCDryRunAndApply(t *testing.T) {
	dir := t.TempDir()
	auth := mustAuth(t)
	s := New(dir, auth)
	_ = os.MkdirAll(filepath.Join(dir, "blobs"), 0o755)

	keepPayload := []byte("keep-me")
	keepSum := sha256.Sum256(keepPayload)
	keepDig := "sha256-" + hex.EncodeToString(keepSum[:])
	_ = os.WriteFile(filepath.Join(dir, "blobs", keepDig), keepPayload, 0o644)

	orphanPayload := []byte("orphan")
	orphanSum := sha256.Sum256(orphanPayload)
	orphanDig := "sha256-" + hex.EncodeToString(orphanSum[:])
	_ = os.WriteFile(filepath.Join(dir, "blobs", orphanDig), orphanPayload, 0o644)

	mfDir := filepath.Join(dir, "manifests", "h", "n", "m")
	_ = os.MkdirAll(mfDir, 0o755)
	mf := `{"layers":[{"digest":"sha256:` + hex.EncodeToString(keepSum[:]) + `"}]}`
	_ = os.WriteFile(filepath.Join(mfDir, "t"), []byte(mf), 0o644)

	body, _ := json.Marshal(map[string]any{"apply": false})
	req := httptest.NewRequest(http.MethodPost, "/v1/gc", bytes.NewReader(body))
	signed(t, auth, req, body)
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	var dry gcResponse
	_ = json.Unmarshal(rr.Body.Bytes(), &dry)
	if !dry.DryRun || len(dry.Orphans) != 1 || dry.Orphans[0] != orphanDig {
		t.Fatalf("dry %+v", dry)
	}
	if _, err := os.Stat(filepath.Join(dir, "blobs", orphanDig)); err != nil {
		t.Fatal("dry-run should not delete")
	}

	body, _ = json.Marshal(map[string]any{"apply": true})
	req = httptest.NewRequest(http.MethodPost, "/v1/gc", bytes.NewReader(body))
	signed(t, auth, req, body)
	rr = httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if _, err := os.Stat(filepath.Join(dir, "blobs", orphanDig)); !os.IsNotExist(err) {
		t.Fatal("apply should delete orphan")
	}
	if _, err := os.Stat(filepath.Join(dir, "blobs", keepDig)); err != nil {
		t.Fatal("keep blob deleted")
	}
}

func TestVerifyDigest(t *testing.T) {
	dir := t.TempDir()
	auth := mustAuth(t)
	s := New(dir, auth)
	_ = os.MkdirAll(filepath.Join(dir, "blobs"), 0o755)
	payload := []byte("verify-ok")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	_ = os.WriteFile(filepath.Join(dir, "blobs", digest), payload, 0o644)

	body, _ := json.Marshal(verifyRequest{Digests: []string{digest}})
	req := httptest.NewRequest(http.MethodPost, "/v1/verify", bytes.NewReader(body))
	signed(t, auth, req, body)
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	var resp verifyResponse
	_ = json.Unmarshal(rr.Body.Bytes(), &resp)
	if len(resp.Results) != 1 || !resp.Results[0].OK {
		t.Fatalf("%+v", resp)
	}
}
