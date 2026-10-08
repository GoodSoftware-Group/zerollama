package storaged

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// fsyncPath syncs a file's data to stable storage.
func fsyncPath(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	err = f.Sync()
	cerr := f.Close()
	if err != nil {
		return err
	}
	return cerr
}

// fsyncDir syncs the directory so a prior rename/create is durable.
func fsyncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	err = d.Sync()
	cerr := d.Close()
	if err != nil {
		return err
	}
	return cerr
}

func finalizeBlobPartial(tmp, path, digest string, h hash.Hash) error {
	got := hex.EncodeToString(h.Sum(nil))
	want := strings.TrimPrefix(strings.ReplaceAll(digest, ":", "-"), "sha256-")
	if !strings.EqualFold(got, want) {
		return fmt.Errorf("digest mismatch")
	}
	if err := fsyncPath(tmp); err != nil {
		return fmt.Errorf("fsync partial: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	if err := fsyncPath(path); err != nil {
		return fmt.Errorf("fsync final: %w", err)
	}
	if err := fsyncDir(filepath.Dir(path)); err != nil {
		return fmt.Errorf("fsync dir: %w", err)
	}
	return nil
}

// writeAtomicLocal mirrors client writeAtomic for peer pulls into the models tree.
func writeAtomicLocal(partial, final string, rc io.Reader, expect int64, digest string) error {
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	h := sha256.New()
	written, err := io.Copy(io.MultiWriter(f, h), rc)
	if syncErr := f.Sync(); syncErr != nil && err == nil {
		err = syncErr
	}
	cerr := f.Close()
	if err != nil {
		_ = os.Remove(partial)
		return err
	}
	if cerr != nil {
		_ = os.Remove(partial)
		return cerr
	}
	if expect > 0 && written != expect {
		_ = os.Remove(partial)
		return fmt.Errorf("short write: got %d want %d", written, expect)
	}
	return finalizeBlobPartial(partial, final, digest, h)
}
