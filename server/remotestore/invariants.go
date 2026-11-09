package remotestore

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// BlobPublishGrace is how long a blob may sit without a manifest before GC.
// A push writes layers first and the manifest second. Deleting inside this
// window removes the layers the manifest is about to name.
const BlobPublishGrace = time.Hour

// RemoteBlobMatchesLocal is the reclaim rule. HEAD 200 is not enough.
// A missing Content-Length (-1 or 0) does not match.
func RemoteBlobMatchesLocal(remoteSize, localSize int64) bool {
	return remoteSize > 0 && localSize > 0 && remoteSize == localSize
}

// emptyBlobSHA256 is sha256 of zero bytes. TEMPLATE "" and an empty LICENSE
// are real layers with this digest, so a zero-byte file of this name is complete.
const emptyBlobSHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

// IsEmptyContentDigest reports the empty-blob digest (sha256: or sha256-).
func IsEmptyContentDigest(digest string) bool {
	d := strings.ToLower(strings.TrimSpace(digest))
	d = strings.TrimPrefix(d, "sha256:")
	d = strings.TrimPrefix(d, "sha256-")
	return d == emptyBlobSHA256
}

// LocalBlobComplete is the hit rule for a digest path. Zero bytes is a
// truncated write, not a cached blob. The empty-content digest is the exception
// and is checked with IsEmptyContentDigest, because its size is legitimately 0.
func LocalBlobComplete(size int64) bool {
	return size > 0
}

// BlobSizeMatches is the size half of a digest hit. A positive expect must
// equal the file. An unknown expect (0) still rejects a zero-byte file.
// BlobContentTrusted then checks the bytes.
func BlobSizeMatches(size, expect int64) bool {
	if !LocalBlobComplete(size) {
		return false
	}
	if expect <= 0 {
		return true
	}
	return size == expect
}

// ManifestRefreshInterval is how long a successful storage check can stand.
// ManifestRefreshRetry is how long a failed check can stand. A failure is not
// proof that the local tag is current, so the next look is sooner.
const (
	ManifestRefreshInterval = 5 * time.Second
	ManifestRefreshRetry    = 2 * time.Second
)

// ManifestCheckFresh reports whether a previous check is still in its window.
func ManifestCheckFresh(succeeded bool, age time.Duration) bool {
	if age < 0 {
		return false
	}
	if succeeded {
		return age < ManifestRefreshInterval
	}
	return age < ManifestRefreshRetry
}

// SameBlobDir reports that the cache blob directory is the models blob directory.
// Equal cleaned paths match. A symlink to that directory matches by file identity.
func SameBlobDir(a, b string) bool {
	if filepath.Clean(a) == filepath.Clean(b) {
		return true
	}
	ia, err1 := os.Stat(a)
	ib, err2 := os.Stat(b)
	if err1 != nil || err2 != nil {
		return false
	}
	return os.SameFile(ia, ib)
}

// BlobContentTrusted reports that path's bytes are digest. A sidecar records a
// passing hash for this size and mtime so a later load does not read the file
// again. A same-size file with a new mtime is hashed.
func BlobContentTrusted(path, digest string, fi os.FileInfo) (bool, error) {
	if fi == nil {
		return false, nil
	}
	want := digestHex(digest)
	if len(want) != 64 {
		return false, nil
	}
	if stampTrusted(path, want, fi) {
		return true, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	if hex.EncodeToString(h.Sum(nil)) != want {
		return false, nil
	}
	_ = writeBlobStamp(path, want, fi)
	return true, nil
}

func digestHex(digest string) string {
	d := strings.ToLower(strings.TrimSpace(digest))
	d = strings.TrimPrefix(d, "sha256:")
	d = strings.TrimPrefix(d, "sha256-")
	return d
}

func blobStampPath(path string) string {
	return path + ".verified"
}

func stampTrusted(path, want string, fi os.FileInfo) bool {
	b, err := os.ReadFile(blobStampPath(path))
	if err != nil {
		return false
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 3 || lines[0] != want {
		return false
	}
	size, err1 := strconv.ParseInt(lines[1], 10, 64)
	mod, err2 := strconv.ParseInt(lines[2], 10, 64)
	if err1 != nil || err2 != nil {
		return false
	}
	return size == fi.Size() && mod == fi.ModTime().UnixNano()
}

func writeBlobStamp(path, want string, fi os.FileInfo) error {
	body := fmt.Sprintf("%s\n%d\n%d\n", want, fi.Size(), fi.ModTime().UnixNano())
	return os.WriteFile(blobStampPath(path), []byte(body), 0o644)
}

// RemoveBlobStamp deletes the verification sidecar for a blob path.
func RemoveBlobStamp(path string) {
	_ = os.Remove(blobStampPath(path))
}

// ManifestBytesDiffer reports that the storage manifest should replace the
// local file. Equal bytes stay put so a refresh does not rewrite a warm tag.
func ManifestBytesDiffer(local, remote []byte) bool {
	return len(remote) > 0 && !bytes.Equal(local, remote)
}

// BlobTooYoungToGC reports a blob still inside BlobPublishGrace.
func BlobTooYoungToGC(modTime, now time.Time) bool {
	if modTime.IsZero() || now.IsZero() {
		return false
	}
	age := now.Sub(modTime)
	return age >= 0 && age < BlobPublishGrace
}

// EphemeralMayDelete is the shared-layer rule. A pin count above zero means
// another loaded model still maps the scratch file.
func EphemeralMayDelete(pins int) bool {
	return pins <= 0
}

// ManifestReferencedDigests returns blob filenames named by manifests under modelsDir.
// Cache eviction uses this so a cap cannot delete a layer a local tag still names.
func ManifestReferencedDigests(modelsDir string) map[string]struct{} {
	out := map[string]struct{}{}
	root := filepath.Join(modelsDir, "manifests")
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d == nil || d.IsDir() {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		for _, name := range blobNamesInManifest(string(b)) {
			out[name] = struct{}{}
		}
		return nil
	})
	return out
}

func blobNamesInManifest(s string) []string {
	var out []string
	seen := map[string]struct{}{}
	for _, needle := range []string{"sha256:", "sha256-"} {
		rest := s
		for {
			i := strings.Index(rest, needle)
			if i < 0 {
				break
			}
			rest = rest[i:]
			if len(rest) < 7+64 {
				break
			}
			hexPart := rest[7 : 7+64]
			ok := true
			for _, c := range hexPart {
				if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
					ok = false
					break
				}
			}
			if ok {
				name := "sha256-" + strings.ToLower(hexPart)
				if _, exists := seen[name]; !exists {
					seen[name] = struct{}{}
					out = append(out, name)
				}
			}
			rest = rest[7:]
		}
	}
	return out
}

// PullWritesServedTree is the storage-serve rule. POST /v1/pull uses
// envconfig.Models(). That path has to be the tree this process serves.
func PullWritesServedTree(served, envModels string) error {
	if served == "" || envModels == "" {
		return fmt.Errorf("storage pull requires a models directory")
	}
	if filepath.Clean(served) != filepath.Clean(envModels) {
		return fmt.Errorf("storage pull would write %s; this server is serving %s", envModels, served)
	}
	return nil
}

// PullStreamComplete fails when a storage pull stream ends without success.
// HTTP 200 is written before the registry pull runs, so status alone is not done.
func PullStreamComplete(sawSuccess bool) error {
	if sawSuccess {
		return nil
	}
	return fmt.Errorf("storage pull: stream ended before success")
}
