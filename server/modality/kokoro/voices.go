package kokoro

import (
	"archive/zip"
	"path/filepath"
	"sort"
	"strings"
)

// VoiceIDsFromBin lists voice ids from a voices-v1.0.bin numpy archive.
// The file is a zip of *.npy members. An unreadable file returns nil.
func VoiceIDsFromBin(path string) []string {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil
	}
	defer r.Close()
	seen := map[string]struct{}{}
	var out []string
	for _, f := range r.File {
		name := filepath.Base(f.Name)
		if !strings.HasSuffix(strings.ToLower(name), ".npy") {
			continue
		}
		id := strings.TrimSuffix(name, filepath.Ext(name))
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}
