package discover

import (
	"encoding/json"
	"os"
	"sort"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// ANEDraftEntry is a local eliza DFlash / speculative draft target for ANE research.
type ANEDraftEntry struct {
	Name                string `json:"name"`
	Tag                 string `json:"tag"`
	BaseGGUF            string `json:"base_gguf,omitempty"`
	DraftGGUF           string `json:"draft_gguf,omitempty"`
	DraftSidecarPresent bool   `json:"draft_sidecar_present,omitempty"`
	SpecType            string `json:"spec_type,omitempty"`
	NumCtx              int    `json:"num_ctx,omitempty"`
	Architecture        string `json:"architecture,omitempty"`
	EmbeddingLength     int    `json:"embedding_length,omitempty"`
	ProxyChannels       int    `json:"proxy_channels,omitempty"`
	ProxySpatial        int    `json:"proxy_spatial,omitempty"`
}

// ListANEDraftInventory scans pulled eliza-*-dflash (and spec_type draft-eagle3) tags.
func ListANEDraftInventory() ([]ANEDraftEntry, error) {
	ms, err := manifest.Manifests(true)
	if err != nil {
		return nil, err
	}

	var out []ANEDraftEntry
	for name, mf := range ms {
		entry, ok, err := aneDraftEntryFromManifest(name, mf)
		if err != nil || !ok {
			continue
		}
		out = append(out, entry)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Tag < out[j].Tag })
	return out, nil
}

// SelectANEDraftModel picks an entry by preferred tag/name (case-insensitive).
func SelectANEDraftModel(entries []ANEDraftEntry, preferred string) (ANEDraftEntry, bool) {
	if len(entries) == 0 {
		return ANEDraftEntry{}, false
	}
	preferred = strings.TrimSpace(preferred)
	if preferred != "" {
		want := strings.ToLower(preferred)
		for _, e := range entries {
			tagLower := strings.ToLower(e.Tag)
			shortLower := strings.ToLower(strings.TrimSuffix(strings.SplitN(e.Tag, ":", 2)[0], ""))
			if strings.EqualFold(e.Name, preferred) ||
				strings.EqualFold(e.Tag, preferred) ||
				strings.EqualFold(shortLower, want) ||
				strings.Contains(tagLower, want) {
				return e, true
			}
		}
	}
	return entries[0], true
}

func aneDraftEntryFromManifest(name model.Name, mf *manifest.Manifest) (ANEDraftEntry, bool, error) {
	if mf == nil {
		return ANEDraftEntry{}, false, nil
	}

	tag := name.DisplayShortest()
	if tag == "" {
		tag = name.String()
	}
	short := strings.SplitN(tag, ":", 2)[0]
	if !strings.Contains(strings.ToLower(short), "dflash") &&
		!strings.Contains(strings.ToLower(short), "eliza") {
		return ANEDraftEntry{}, false, nil
	}

	var (
		basePath  string
		draftPath string
		params    map[string]any
		specType  string
		numCtx    int
	)

	for _, layer := range mf.Layers {
		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return ANEDraftEntry{}, false, err
			}
			if basePath == "" {
				basePath = path
			} else {
				draftPath = path
			}
		case "application/vnd.ollama.image.params":
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return ANEDraftEntry{}, false, err
			}
			f, err := os.Open(path)
			if err != nil {
				return ANEDraftEntry{}, false, err
			}
			_ = json.NewDecoder(f).Decode(&params)
			f.Close()
		}
	}

	if params != nil {
		if v, ok := params["spec_type"].(string); ok {
			specType = strings.TrimSpace(v)
		}
		if v, ok := params["num_ctx"].(float64); ok {
			numCtx = int(v)
		}
	}

	isDraft := strings.Contains(strings.ToLower(short), "dflash") ||
		specType == "draft-eagle3" || specType == "draft-mtp"
	if !isDraft || basePath == "" {
		return ANEDraftEntry{}, false, nil
	}

	if params != nil {
		if v, ok := params["draft_model_path"].(string); ok && strings.TrimSpace(v) != "" {
			draftPath = strings.TrimSpace(v)
		}
	}
	if draftPath == "" {
		draftPath = FindANEDraftSidecarPath(short)
	}
	if draftPath == "" {
		candidates := ANEDraftSidecarCandidates(short)
		if len(candidates) > 0 {
			draftPath = candidates[0]
		}
	}

	entry := ANEDraftEntry{
		Name:      name.String(),
		Tag:       tag,
		BaseGGUF:  basePath,
		DraftGGUF: draftPath,
		SpecType:  specType,
		NumCtx:    numCtx,
	}
	if draftPath != "" {
		if st, err := os.Stat(draftPath); err == nil && !st.IsDir() {
			entry.DraftSidecarPresent = true
		}
	}
	if info, err := ProbeANEDraftGGUF(basePath, draftPath); err == nil {
		entry.Architecture = info.Architecture
		entry.EmbeddingLength = info.EmbeddingLength
		entry.ProxyChannels = info.ProxyChannels
		entry.ProxySpatial = info.ProxySpatial
	}
	return entry, true, nil
}

// FindANEDraftSidecarPath returns the first existing drafter GGUF for a tag.
func FindANEDraftSidecarPath(shortName string) string {
	for _, c := range ANEDraftSidecarCandidates(shortName) {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c
		}
	}
	return ""
}
