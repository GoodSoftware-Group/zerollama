package remotestore

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestInvariantRemoteMatchIsSizeNotStatus(t *testing.T) {
	if !RemoteBlobMatchesLocal(100, 100) {
		t.Fatal("equal sizes match")
	}
	if RemoteBlobMatchesLocal(-1, 100) || RemoteBlobMatchesLocal(0, 100) || RemoteBlobMatchesLocal(99, 100) {
		t.Fatal("HEAD without an equal positive length is not a match")
	}
}

func TestInvariantBlobSizeMustMatchTheManifest(t *testing.T) {
	if !BlobSizeMatches(10, 10) || BlobSizeMatches(10, 11) || BlobSizeMatches(0, 0) {
		t.Fatal("size check")
	}
	if !BlobSizeMatches(10, 0) {
		t.Fatal("unknown manifest size still accepts a non-empty file")
	}
}

func TestInvariantSameSizeCorruptBlobIsNotTrusted(t *testing.T) {
	dir := t.TempDir()
	body := []byte("hello blob")
	sum := sha256.Sum256(body)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	path := filepath.Join(dir, digest)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	ok, err := BlobContentTrusted(path, digest, fi)
	if err != nil || !ok {
		t.Fatalf("matching bytes: %v %v", ok, err)
	}
	fi, err = os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	ok, err = BlobContentTrusted(path, digest, fi)
	if err != nil || !ok {
		t.Fatalf("stamp: %v %v", ok, err)
	}
	corrupt := append([]byte(nil), body...)
	corrupt[0] = 'H'
	if err := os.WriteFile(path, corrupt, 0o644); err != nil {
		t.Fatal(err)
	}
	fi, err = os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if ok, err := BlobContentTrusted(path, digest, fi); err != nil || ok {
		t.Fatalf("same size, wrong bytes: %v %v", ok, err)
	}
}

func TestInvariantBlobDirSymlinkIsTheModelsDir(t *testing.T) {
	root := t.TempDir()
	blobs := filepath.Join(root, "blobs")
	if err := os.Mkdir(blobs, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "link")
	if err := os.Symlink(blobs, link); err != nil {
		t.Fatal(err)
	}
	if !SameBlobDir(blobs, blobs) || !SameBlobDir(link, blobs) {
		t.Fatal("symlink cache dir is the models blob dir")
	}
	other := t.TempDir()
	if SameBlobDir(other, blobs) {
		t.Fatal("different directories")
	}
}

func TestInvariantFailedManifestCheckExpiresSooner(t *testing.T) {
	if !ManifestCheckFresh(true, time.Second) || ManifestCheckFresh(true, ManifestRefreshInterval) {
		t.Fatal("success window")
	}
	if !ManifestCheckFresh(false, time.Second) || ManifestCheckFresh(false, ManifestRefreshRetry) {
		t.Fatal("failure window")
	}
	if ManifestCheckFresh(false, -time.Second) {
		t.Fatal("negative age")
	}
}

func TestInvariantManifestRefreshReplacesDifferentBytes(t *testing.T) {
	if ManifestBytesDiffer([]byte("a"), []byte("a")) || !ManifestBytesDiffer([]byte("a"), []byte("b")) {
		t.Fatal("equal manifests stay")
	}
	if ManifestBytesDiffer([]byte("a"), nil) {
		t.Fatal("empty remote must not wipe a local tag")
	}
}

func TestEmptyContentDigestIsARealBlob(t *testing.T) {
	const hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if !IsEmptyContentDigest("sha256:"+hex) || !IsEmptyContentDigest("sha256-"+strings.ToUpper(hex)) {
		t.Fatal("empty digest")
	}
	if IsEmptyContentDigest("sha256-" + strings.Repeat("ab", 32)) {
		t.Fatal("non-empty digest")
	}
}

func TestInvariantZeroByteBlobIsNotComplete(t *testing.T) {
	if LocalBlobComplete(0) {
		t.Fatal("empty file")
	}
	if !LocalBlobComplete(1) {
		t.Fatal("non-empty file")
	}
}

func TestInvariantYoungBlobSurvivesGC(t *testing.T) {
	now := time.Now()
	if !BlobTooYoungToGC(now.Add(-time.Minute), now) {
		t.Fatal("fresh blob")
	}
	if BlobTooYoungToGC(now.Add(-2*BlobPublishGrace), now) {
		t.Fatal("old orphan")
	}
}

func TestInvariantSharedPinBlocksEphemeralDelete(t *testing.T) {
	if EphemeralMayDelete(1) || EphemeralMayDelete(2) {
		t.Fatal("remaining pins")
	}
	if !EphemeralMayDelete(0) {
		t.Fatal("last unpin")
	}
}

func TestInvariantPullWritesTheServedTree(t *testing.T) {
	if err := PullWritesServedTree("/models", "/models"); err != nil {
		t.Fatal(err)
	}
	if err := PullWritesServedTree("/models/", "/models"); err != nil {
		t.Fatal(err)
	}
	if err := PullWritesServedTree("/served", "/elsewhere"); err == nil {
		t.Fatal("expected mismatch")
	}
}

func TestInvariantPullNeedsSuccessEvent(t *testing.T) {
	if err := PullStreamComplete(true); err != nil {
		t.Fatal(err)
	}
	err := PullStreamComplete(false)
	if err == nil || !strings.Contains(err.Error(), "success") {
		t.Fatalf("got %v", err)
	}
}

func TestInvariantManifestDigestsAreNotEvictableNames(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "manifests", "registry.ollama.ai", "library", "m", "latest")
	if err := os.MkdirAll(filepath.Dir(manifest), 0o755); err != nil {
		t.Fatal(err)
	}
	digest := "sha256:" + strings.Repeat("ab", 32)
	body := `{"layers":[{"digest":"` + digest + `"}]}`
	if err := os.WriteFile(manifest, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got := ManifestReferencedDigests(dir)
	want := "sha256-" + strings.Repeat("ab", 32)
	if _, ok := got[want]; !ok {
		t.Fatalf("missing %s in %v", want, got)
	}
}

func TestInvariantClientDigestIsHexOnly(t *testing.T) {
	good := "sha256-" + strings.Repeat("ab", 32)
	if !safeBlobDigest(good) {
		t.Fatal(good)
	}
	if safeBlobDigest("sha256-../../etc/passwd") || safeBlobDigest("sha256-"+strings.Repeat("AB", 32)) {
		t.Fatal("path or uppercase must not pass the client check")
	}
}
