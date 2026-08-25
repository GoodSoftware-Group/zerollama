package remotestore_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/storaged"
	"github.com/ollama/ollama/server/remotestore/tensorproto"
)

func TestIntegrationFetchRoundTrip(t *testing.T) {
	secret := "integration-test-secret"
	auth, err := remotestore.NewAuth(secret)
	if err != nil {
		t.Fatal(err)
	}

	modelsDir := t.TempDir()
	payload := []byte("hello remote storage blob contents")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	blobDir := filepath.Join(modelsDir, "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blobDir, digest), payload, 0o644); err != nil {
		t.Fatal(err)
	}
	mfDir := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "toy", "latest")
	if err := os.MkdirAll(mfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	mf := map[string]any{
		"schemaVersion": 2,
		"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
		"layers": []map[string]any{
			{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + hex.EncodeToString(sum[:]), "size": len(payload)},
		},
	}
	mfBytes, _ := json.Marshal(mf)
	if err := os.WriteFile(filepath.Join(mfDir, "latest"), mfBytes, 0o644); err != nil {
		// path is manifests/.../toy/latest as file
	}
	// Fix: walk expects file at .../model/tag
	_ = os.RemoveAll(mfDir)
	parent := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "toy")
	_ = os.MkdirAll(parent, 0o755)
	_ = os.WriteFile(filepath.Join(parent, "latest"), mfBytes, 0o644)

	srv := storaged.New(modelsDir, auth)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	cacheDir := t.TempDir()
	r := &remotestore.Resolver{
		Servers:  []string{ts.URL},
		Auth:     auth,
		CacheDir: cacheDir,
		MaxBytes: 1 << 20,
		Mode:     remotestore.CachePersist,
		Chain:    remotestore.PreferRDMAThenTCP(auth),
		HTTP:     ts.Client(),
	}

	ctx := context.Background()
	path, err := r.Fetch(ctx, digest)
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("content mismatch")
	}

	// Second fetch should be cache hit (file already present).
	path2, err := r.Fetch(ctx, digest)
	if err != nil {
		t.Fatal(err)
	}
	if path2 != path {
		t.Fatalf("cache path changed")
	}

	// Capability
	cap, err := remotestore.GetCapability(ctx, auth, ts.Client(), ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(cap.Transports) == 0 {
		t.Fatal("expected transports")
	}

	// Manifest fetch
	b, err := r.FetchManifest(ctx, "registry.ollama.ai", "library", "toy", "latest")
	if err != nil {
		t.Fatalf("manifest: %v", err)
	}
	if len(b) == 0 {
		t.Fatal("empty manifest")
	}

	_ = io.Discard
}

func TestTensorProtoValidate(t *testing.T) {
	t.Parallel()
	var r tensorproto.Request
	if err := r.Validate(); err == nil {
		t.Fatal("expected error")
	}
	r.Digest = "sha256-abc"
	r.TensorRef = "layer.0.attn"
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestEphemeralMode(t *testing.T) {
	secret := "eph-secret"
	auth, err := remotestore.NewAuth(secret)
	if err != nil {
		t.Fatal(err)
	}
	modelsDir := t.TempDir()
	payload := []byte("ephemeral-blob-data-xxxxxx")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	_ = os.MkdirAll(filepath.Join(modelsDir, "blobs"), 0o755)
	_ = os.WriteFile(filepath.Join(modelsDir, "blobs", digest), payload, 0o644)

	srv := storaged.New(modelsDir, auth)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	scratch := t.TempDir()
	r := &remotestore.Resolver{
		Servers:    []string{ts.URL},
		Auth:       auth,
		CacheDir:   t.TempDir(),
		Mode:       remotestore.CacheEphemeral,
		ScratchDir: scratch,
		Chain:      remotestore.PreferRDMAThenTCP(auth),
		HTTP:       ts.Client(),
	}
	path, err := r.Fetch(context.Background(), digest)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Dir(path) != scratch {
		t.Fatalf("expected scratch dir, got %s", path)
	}
	r.ReleaseEphemeral(digest)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected scratch file removed")
	}
}

func TestUnauthorizedRejected(t *testing.T) {
	auth, _ := remotestore.NewAuth("right-secret")
	srv := storaged.New(t.TempDir(), auth)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	bad, _ := remotestore.NewAuth("wrong-secret")
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/v1/capability", nil)
	_ = bad.SignRequest(req, nil)
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("got %d", resp.StatusCode)
	}
}
