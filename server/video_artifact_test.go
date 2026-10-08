package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/openai"
)

func TestVideoFromDiskArtifactRoundTrip(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("ZEROLLAMA_VIDEO_ARTIFACT_TTL", "3600")

	id := "abcd1234"
	mp4 := videoArtifactPath(id)
	if err := os.MkdirAll(filepath.Dir(mp4), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mp4, []byte("fake-mp4-bytes-padding-ok!!!!!!!!!!"), 0o644); err != nil {
		t.Fatal(err)
	}
	v := openai.Video{
		ID:        id,
		Object:    "video",
		CreatedAt: time.Now().Unix(),
		Status:    "completed",
		Model:     "ltxv-2b-distilled:lab",
		Progress:  100,
		Size:      "1280x704",
	}
	if err := writeVideoArtifactMeta(v, mp4); err != nil {
		t.Fatal(err)
	}

	got, path, ok := videoFromDiskArtifact(id)
	if !ok {
		t.Fatal("expected disk artifact")
	}
	if path != mp4 {
		t.Fatalf("path=%q want %q", path, mp4)
	}
	if got.Status != "completed" || got.Model != v.Model || got.Size != v.Size {
		t.Fatalf("got %+v", got)
	}
}

func TestVideoFromDiskArtifactExpired(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("ZEROLLAMA_VIDEO_ARTIFACT_TTL", "1")

	id := "deadbeef"
	mp4 := videoArtifactPath(id)
	if err := os.MkdirAll(filepath.Dir(mp4), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mp4, []byte("fake-mp4-bytes-padding-ok!!!!!!!!!!"), 0o644); err != nil {
		t.Fatal(err)
	}
	v := openai.Video{ID: id, CreatedAt: time.Now().Add(-2 * time.Hour).Unix(), Status: "completed"}
	if err := writeVideoArtifactMeta(v, mp4); err != nil {
		t.Fatal(err)
	}
	// Force expires_at into the past.
	metaPath := videoArtifactMetaPath(id)
	_ = os.WriteFile(metaPath, []byte(`{"id":"deadbeef","status":"completed","expires_at":1,"output_path":"`+mp4+`"}`), 0o644)

	if _, _, ok := videoFromDiskArtifact(id); ok {
		t.Fatal("expired artifact must not serve")
	}
}

func TestDeleteVideoArtifact(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	id := "delme001"
	mp4 := videoArtifactPath(id)
	if err := os.MkdirAll(filepath.Dir(mp4), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mp4, []byte("fake-mp4-bytes-padding-ok!!!!!!!!!!"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeVideoArtifactMeta(openai.Video{ID: id, Status: "completed"}, mp4); err != nil {
		t.Fatal(err)
	}
	if err := deleteVideoArtifact(id); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(mp4); !os.IsNotExist(err) {
		t.Fatalf("mp4 should be gone: %v", err)
	}
	if _, err := os.Stat(videoArtifactMetaPath(id)); !os.IsNotExist(err) {
		t.Fatalf("json should be gone: %v", err)
	}
	if err := deleteVideoArtifact(id); err != nil {
		t.Fatalf("idempotent delete: %v", err)
	}
}

func TestPruneExpiredVideoArtifacts(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("ZEROLLAMA_VIDEO_ARTIFACT_TTL", "3600")

	keep := "keep0001"
	drop := "drop0001"
	for _, id := range []string{keep, drop} {
		mp4 := videoArtifactPath(id)
		if err := os.MkdirAll(filepath.Dir(mp4), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(mp4, []byte("fake-mp4-bytes-padding-ok!!!!!!!!!!"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := writeVideoArtifactMeta(openai.Video{ID: id, Status: "completed"}, mp4); err != nil {
			t.Fatal(err)
		}
	}
	// Expire only drop.
	_ = os.WriteFile(videoArtifactMetaPath(drop), []byte(`{"id":"drop0001","status":"completed","expires_at":1}`), 0o644)

	n := pruneExpiredVideoArtifacts()
	if n != 1 {
		t.Fatalf("removed=%d want 1", n)
	}
	if _, err := os.Stat(videoArtifactPath(keep)); err != nil {
		t.Fatal("keep should remain")
	}
	if _, err := os.Stat(videoArtifactPath(drop)); !os.IsNotExist(err) {
		t.Fatal("drop should be pruned")
	}
}
