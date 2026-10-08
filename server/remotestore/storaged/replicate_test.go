package storaged

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
	"time"
)

func TestPeerReplicateOnPut(t *testing.T) {
	auth := mustAuth(t)
	dirA := t.TempDir()
	dirB := t.TempDir()
	_ = os.MkdirAll(filepath.Join(dirA, "blobs"), 0o755)
	_ = os.MkdirAll(filepath.Join(dirB, "blobs"), 0o755)

	srvB := New(dirB, auth)
	tsB := httptest.NewServer(srvB)
	defer tsB.Close()

	srvA := New(dirA, auth)
	srvA.Peers = []string{tsB.URL}
	srvA.HTTP = tsB.Client()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	srvA.StartReplicator(ctx)

	tsA := httptest.NewServer(srvA)
	defer tsA.Close()

	payload := []byte("replicate-me-please-xx")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])

	req, err := http.NewRequest(http.MethodPut, tsA.URL+"/v1/blob/"+digest, bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	if err := auth.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	resp, err := tsA.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("put status %d", resp.StatusCode)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(filepath.Join(dirB, "blobs", digest)); err == nil {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("blob not replicated to peer B")
}

func TestSyncToPeer(t *testing.T) {
	auth := mustAuth(t)
	dirA := t.TempDir()
	dirB := t.TempDir()
	payload := []byte("sync-blob")
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	_ = os.MkdirAll(filepath.Join(dirA, "blobs"), 0o755)
	_ = os.WriteFile(filepath.Join(dirA, "blobs", digest), payload, 0o644)

	srvB := New(dirB, auth)
	tsB := httptest.NewServer(srvB)
	defer tsB.Close()

	srvA := New(dirA, auth)
	srvA.HTTP = tsB.Client()
	if err := srvA.SyncToPeer(context.Background(), tsB.URL); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(dirB, "blobs", digest))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatal("sync content mismatch")
	}
}
