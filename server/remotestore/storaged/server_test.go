package storaged

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/server/remotestore"
)

func TestBlobPathRejectsTraversal(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	s := New(dir, mustAuth(t))
	for _, bad := range []string{
		"sha256-aa/../../../etc/passwd",
		"sha256-../../etc/passwd",
		"sha256-",
		"notadigest",
		"sha256-" + strings.Repeat("g", 64),
		"sha256-" + strings.Repeat("a", 63),
	} {
		if _, err := s.blobPath(bad); err == nil {
			t.Fatalf("blobPath(%q) should reject", bad)
		}
	}
	good := "sha256-" + strings.Repeat("ab", 32)
	p, err := s.blobPath(good)
	if err != nil {
		t.Fatal(err)
	}
	wantPrefix := filepath.Clean(filepath.Join(dir, "blobs"))
	if !strings.HasPrefix(filepath.Clean(p), wantPrefix) {
		t.Fatalf("path %q escapes %q", p, wantPrefix)
	}
}

func TestTensorPathRejectsDotDot(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	auth := mustAuth(t)
	s := New(dir, auth)
	// Bypass ServeMux path cleaning: exercise the handler with a raw ".." segment.
	req := httptest.NewRequest(http.MethodGet, "http://example/v1/tensor/h/n/m/../tok", nil)
	req.URL.Path = "/v1/tensor/h/n/m/../tok"
	if err := auth.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.handleTensor(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400, got %d body=%s", rr.Code, rr.Body.String())
	}
}

func TestPutBlobDigestMismatch(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	auth := mustAuth(t)
	s := New(dir, auth)
	payload := []byte("hello")
	wrong := "sha256-" + strings.Repeat("0", 64)
	req := httptest.NewRequest(http.MethodPut, "/v1/blob/"+wrong, bytes.NewReader(payload))
	if err := auth.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400 digest mismatch, got %d", rr.Code)
	}
}

func TestPutBlobRoundTrip(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	auth := mustAuth(t)
	s := New(dir, auth)
	payload := []byte("hello storaged")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	req := httptest.NewRequest(http.MethodPut, "/v1/blob/"+digest, bytes.NewReader(payload))
	req.ContentLength = int64(len(payload))
	if err := auth.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusCreated {
		t.Fatalf("put: %d %s", rr.Code, rr.Body.String())
	}
	got, err := os.ReadFile(filepath.Join(dir, "blobs", digest))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("content mismatch")
	}

	get := httptest.NewRequest(http.MethodGet, "/v1/blob/"+digest, nil)
	if err := auth.SignRequest(get, nil); err != nil {
		t.Fatal(err)
	}
	rr2 := httptest.NewRecorder()
	s.ServeHTTP(rr2, get)
	if rr2.Code != http.StatusOK {
		t.Fatalf("get: %d", rr2.Code)
	}
	body, _ := io.ReadAll(rr2.Body)
	if !bytes.Equal(body, payload) {
		t.Fatalf("get body mismatch")
	}
}

func mustAuth(t *testing.T) *remotestore.Auth {
	t.Helper()
	a, err := remotestore.NewAuth("storaged-test-secret")
	if err != nil {
		t.Fatal(err)
	}
	return a
}
