package discover

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// ANEModelEntry is a local pulled GGUF tag for ANE prefill / geometry probes.
type ANEModelEntry struct {
	Name            string `json:"name"`
	Tag             string `json:"tag"`
	GGUFPath        string `json:"gguf_path"`
	Architecture    string `json:"architecture,omitempty"`
	EmbeddingLength int    `json:"embedding_length,omitempty"`
	BlockCount      int    `json:"block_count,omitempty"`
	NumCtx          int    `json:"num_ctx,omitempty"`
	SpecType        string `json:"spec_type,omitempty"`
	SizeBytes       int64  `json:"size_bytes,omitempty"`
}

// ListANEModelInventory scans all local manifests with a model GGUF blob.
func ListANEModelInventory() ([]ANEModelEntry, error) {
	ms, err := manifest.Manifests(true)
	if err != nil {
		return nil, err
	}

	var out []ANEModelEntry
	for name, mf := range ms {
		entry, ok, err := aneModelEntryFromManifest(name, mf)
		if err != nil || !ok {
			continue
		}
		out = append(out, entry)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Tag < out[j].Tag })
	return out, nil
}

// SelectANEModel picks an entry by preferred tag/name (case-insensitive).
func SelectANEModel(entries []ANEModelEntry, preferred string) (ANEModelEntry, bool) {
	if len(entries) == 0 {
		return ANEModelEntry{}, false
	}
	preferred = strings.TrimSpace(preferred)
	if preferred != "" {
		want := strings.ToLower(preferred)
		for _, e := range entries {
			tagLower := strings.ToLower(e.Tag)
			shortLower := strings.ToLower(strings.SplitN(e.Tag, ":", 2)[0])
			if strings.EqualFold(e.Name, preferred) ||
				strings.EqualFold(e.Tag, preferred) ||
				strings.EqualFold(shortLower, want) ||
				strings.Contains(tagLower, want) {
				return e, true
			}
		}
		return ANEModelEntry{}, false
	}
	return entries[0], true
}

// ANEModelInventorySummary counts local GGUF tags for ANE prefill probes.
type ANEModelInventorySummary struct {
	Total         int `json:"total"`
	WithEmbedding int `json:"with_embedding"`
}

// ProbeANEModelInventorySummary returns model inventory counts.
func ProbeANEModelInventorySummary() (ANEModelInventorySummary, error) {
	entries, err := ListANEModelInventory()
	if err != nil {
		return ANEModelInventorySummary{}, err
	}
	out := ANEModelInventorySummary{Total: len(entries)}
	for _, e := range entries {
		if e.EmbeddingLength > 0 {
			out.WithEmbedding++
		}
	}
	return out, nil
}
func RunANEModelResolveJSON(w io.Writer, preferred string) error {
	entries, err := ListANEModelInventory()
	if err != nil {
		return err
	}
	if preferred != "" {
		if e, ok := SelectANEModel(entries, preferred); ok {
			return json.NewEncoder(w).Encode(e)
		}
		return fmt.Errorf("no local GGUF model matching %q", preferred)
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(entries)
}

func aneModelEntryFromManifest(name model.Name, mf *manifest.Manifest) (ANEModelEntry, bool, error) {
	if mf == nil {
		return ANEModelEntry{}, false, nil
	}

	tag := name.DisplayShortest()
	if tag == "" {
		tag = name.String()
	}

	var (
		ggufPath string
		params   map[string]any
	)

	for _, layer := range mf.Layers {
		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return ANEModelEntry{}, false, err
			}
			if ggufPath == "" {
				ggufPath = path
			}
		case "application/vnd.ollama.image.params":
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return ANEModelEntry{}, false, err
			}
			f, err := os.Open(path)
			if err != nil {
				return ANEModelEntry{}, false, err
			}
			_ = json.NewDecoder(f).Decode(&params)
			f.Close()
		}
	}

	if ggufPath == "" {
		return ANEModelEntry{}, false, nil
	}

	entry := ANEModelEntry{
		Name:     name.String(),
		Tag:      tag,
		GGUFPath: ggufPath,
	}
	if st, err := os.Stat(ggufPath); err == nil {
		entry.SizeBytes = st.Size()
	}
	if params != nil {
		if v, ok := params["spec_type"].(string); ok {
			entry.SpecType = strings.TrimSpace(v)
		}
		if v, ok := params["num_ctx"].(float64); ok {
			entry.NumCtx = int(v)
		}
	}
	if info, err := ProbeANEDraftGGUF(ggufPath, ""); err == nil {
		entry.Architecture = info.Architecture
		entry.EmbeddingLength = info.EmbeddingLength
		entry.BlockCount = info.BlockCount
	}
	return entry, true, nil
}
