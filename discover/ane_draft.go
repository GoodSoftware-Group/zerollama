package discover

import (
	"encoding/json"
	"fmt"
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
	DraftArchitecture   string `json:"draft_architecture,omitempty"`
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
		entry, ok, err := aneDraftEntryFromManifest(name, mf, ms)
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

// dflashParentShort strips a -dflash suffix so eliza-1-2b-dflash → eliza-1-2b.
func dflashParentShort(short string) string {
	s := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(short)), "-dflash")
	if s == "" || s == strings.ToLower(short) {
		return ""
	}
	return s
}

func modelGGUFPathFromManifest(mf *manifest.Manifest) (string, error) {
	if mf == nil {
		return "", fmt.Errorf("nil manifest")
	}
	for _, layer := range mf.Layers {
		if layer.MediaType != "application/vnd.ollama.image.model" {
			continue
		}
		return manifest.BlobsPath(layer.Digest)
	}
	return "", fmt.Errorf("no model layer")
}

func draftGGUFPathFromManifest(mf *manifest.Manifest) (string, error) {
	if mf == nil {
		return "", fmt.Errorf("nil manifest")
	}
	var secondModel string
	for _, layer := range mf.Layers {
		switch layer.MediaType {
		case "application/vnd.ollama.image.draft":
			return manifest.BlobsPath(layer.Digest)
		case "application/vnd.ollama.image.model":
			if secondModel == "" {
				if p, err := manifest.BlobsPath(layer.Digest); err == nil {
					secondModel = p
				}
			} else if p, err := manifest.BlobsPath(layer.Digest); err == nil {
				return p, nil
			}
		}
	}
	return "", fmt.Errorf("no draft layer")
}

func parentManifestForDflash(short string, all map[model.Name]*manifest.Manifest) (*manifest.Manifest, string) {
	parentShort := dflashParentShort(short)
	if parentShort == "" || len(all) == 0 {
		return nil, ""
	}
	want := strings.ToLower(parentShort)
	for n, mf := range all {
		tag := n.DisplayShortest()
		if tag == "" {
			tag = n.String()
		}
		base := strings.SplitN(tag, ":", 2)[0]
		if strings.EqualFold(base, want) {
			return mf, tag
		}
	}
	return nil, ""
}

func aneDraftEntryFromManifest(name model.Name, mf *manifest.Manifest, all map[model.Name]*manifest.Manifest) (ANEDraftEntry, bool, error) {
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
			} else if draftPath == "" {
				draftPath = path
			}
		case "application/vnd.ollama.image.draft":
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return ANEDraftEntry{}, false, err
			}
			draftPath = path
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
		specType == "draft-eagle3" || specType == "draft-mtp" || specType == "dflash"
	if !isDraft || basePath == "" {
		return ANEDraftEntry{}, false, nil
	}

	// zerollama create may embed a 128k base blob; dflash e2e needs the parent tag base
	// (eliza-1-2b) so llama-server keeps spec decoding enabled with the qwen35 sidecar.
	if strings.Contains(strings.ToLower(short), "dflash") {
		if parentMF, _ := parentManifestForDflash(short, all); parentMF != nil {
			if parentBase, err := modelGGUFPathFromManifest(parentMF); err == nil && parentBase != "" {
				basePath = parentBase
			}
		}
		if draftPath == "" {
			if manifestDraft, err := draftGGUFPathFromManifest(mf); err == nil {
				draftPath = manifestDraft
			}
		}
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
	} else if st, err := os.Stat(draftPath); err != nil || st.IsDir() {
		if alt := FindANEDraftSidecarPath(short); alt != "" {
			draftPath = alt
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
	if entry.DraftSidecarPresent && draftPath != "" {
		if arch, err := ProbeSidecarArchitecture(draftPath); err == nil {
			entry.DraftArchitecture = arch
		}
		if dinfo, err := ProbeANEDraftGGUF(draftPath, ""); err == nil && dinfo.EmbeddingLength > 0 {
			entry.EmbeddingLength = dinfo.EmbeddingLength
			if entry.Architecture == "" {
				entry.Architecture = dinfo.Architecture
			}
		}
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
