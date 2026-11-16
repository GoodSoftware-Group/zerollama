package describe

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// HashBytes returns the hex SHA-256 of data.
func HashBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// LoadCatalog reads a catalog JSON file. Missing file yields an empty catalog.
func LoadCatalog(path string) (*CatalogFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &CatalogFile{Images: []Entry{}}, nil
		}
		return nil, err
	}
	var cat CatalogFile
	if err := json.Unmarshal(data, &cat); err != nil {
		return nil, fmt.Errorf("parse catalog %s: %w", path, err)
	}
	if cat.Images == nil {
		cat.Images = []Entry{}
	}
	return &cat, nil
}

// SaveCatalog writes cat atomically (temp file in the same directory, then rename).
func SaveCatalog(path string, cat *CatalogFile) error {
	if cat == nil {
		return fmt.Errorf("nil catalog")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cat, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(dir, ".zerollama-describe-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	ok := false
	defer func() {
		if !ok {
			_ = os.Remove(tmpName)
		}
	}()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}
	ok = true
	return nil
}

// IndexByID builds a map from entry id to entry.
func IndexByID(entries []Entry) map[string]*Entry {
	out := make(map[string]*Entry, len(entries))
	for i := range entries {
		e := &entries[i]
		out[e.ID] = e
	}
	return out
}

// Upsert replaces or appends entry by id, preserving order of existing ids.
func Upsert(images []Entry, entry Entry) []Entry {
	for i := range images {
		if images[i].ID == entry.ID {
			images[i] = entry
			return images
		}
	}
	return append(images, entry)
}

// DropMissing keeps only entries whose id is in keep.
func DropMissing(images []Entry, keep map[string]struct{}) []Entry {
	out := make([]Entry, 0, len(images))
	for _, e := range images {
		if _, ok := keep[e.ID]; ok {
			out = append(out, e)
		}
	}
	return out
}
