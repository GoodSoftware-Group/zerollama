package remotestore_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/storaged"
)

func TestParallelRangeFetch(t *testing.T) {
	restore := remotestore.TestingSetParallelSizes(1024, 512)
	defer restore()

	secret := "parallel-test-secret"
	auth, err := remotestore.NewAuth(secret)
	if err != nil {
		t.Fatal(err)
	}
	modelsDir := t.TempDir()
	payload := bytes.Repeat([]byte("abcdefghijklmnop"), 256) // 4 KiB → several chunks
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	_ = os.MkdirAll(filepath.Join(modelsDir, "blobs"), 0o755)
	_ = os.WriteFile(filepath.Join(modelsDir, "blobs", digest), payload, 0o644)

	srv := storaged.New(modelsDir, auth)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	cache := t.TempDir()
	r := &remotestore.Resolver{
		Servers:  []string{ts.URL},
		Auth:     auth,
		CacheDir: cache,
		Chain:    remotestore.PreferRDMAThenTCP(auth),
		HTTP:     ts.Client(),
	}
	path, err := r.Fetch(context.Background(), digest)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("content mismatch")
	}
}

func TestPeerSoftFailover(t *testing.T) {
	secret := "failover-secret"
	auth, _ := remotestore.NewAuth(secret)
	modelsDir := t.TempDir()
	payload := []byte("failover-blob-contents-xx")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	_ = os.MkdirAll(filepath.Join(modelsDir, "blobs"), 0o755)
	_ = os.WriteFile(filepath.Join(modelsDir, "blobs", digest), payload, 0o644)

	good := storaged.New(modelsDir, auth)
	goodTS := httptest.NewServer(good)
	defer goodTS.Close()

	deadURL := "http://127.0.0.1:1"
	cache := t.TempDir()
	r := &remotestore.Resolver{
		Servers:  []string{deadURL, goodTS.URL},
		Auth:     auth,
		CacheDir: cache,
		Chain:    remotestore.PreferRDMAThenTCP(auth),
		HTTP:     goodTS.Client(),
	}
	path, err := r.Fetch(context.Background(), digest)
	if err != nil {
		t.Fatalf("failover fetch: %v", err)
	}
	got, _ := os.ReadFile(path)
	if !bytes.Equal(got, payload) {
		t.Fatal("content mismatch after failover")
	}
}

func TestDigestMismatchTriesNextPeer(t *testing.T) {
	secret := "mismatch-failover"
	auth, _ := remotestore.NewAuth(secret)

	goodDir := t.TempDir()
	badDir := t.TempDir()
	payload := []byte("correct-payload-bytes!!")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])

	_ = os.MkdirAll(filepath.Join(goodDir, "blobs"), 0o755)
	_ = os.WriteFile(filepath.Join(goodDir, "blobs", digest), payload, 0o644)

	_ = os.MkdirAll(filepath.Join(badDir, "blobs"), 0o755)
	// Wrong bytes under the same digest name (simulates corruption).
	_ = os.WriteFile(filepath.Join(badDir, "blobs", digest), []byte("WRONG-BYTES-NOT-MATCHING"), 0o644)

	badSrv := storaged.New(badDir, auth)
	badTS := httptest.NewServer(badSrv)
	defer badTS.Close()
	goodSrv := storaged.New(goodDir, auth)
	goodTS := httptest.NewServer(goodSrv)
	defer goodTS.Close()

	cache := t.TempDir()
	r := &remotestore.Resolver{
		Servers:  []string{badTS.URL, goodTS.URL},
		Auth:     auth,
		CacheDir: cache,
		Chain:    remotestore.PreferRDMAThenTCP(auth),
		HTTP:     &http.Client{},
	}
	path, err := r.Fetch(context.Background(), digest)
	if err != nil {
		t.Fatalf("expected failover past corrupt peer: %v", err)
	}
	got, _ := os.ReadFile(path)
	if !bytes.Equal(got, payload) {
		t.Fatalf("got %q", got)
	}
}
