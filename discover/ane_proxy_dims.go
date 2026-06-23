package discover

import (
	"fmt"
	"strings"
)

// ANEModelProxyDims is draft conv proxy geometry derived from a local GGUF tag.
type ANEModelProxyDims struct {
	Tag                 string `json:"tag"`
	Name                string `json:"name,omitempty"`
	EmbeddingLength     int    `json:"embedding_length"`
	ProxyChannels       int    `json:"proxy_channels"`
	ProxySpatial        int    `json:"proxy_spatial"`
	SpecType            string `json:"spec_type,omitempty"`
	DraftSidecarPresent bool   `json:"draft_sidecar_present,omitempty"`
	Source              string `json:"source"`
}

func proxyFromDraftEntry(entry ANEDraftEntry) ANEModelProxyDims {
	ch, sp := entry.ProxyChannels, entry.ProxySpatial
	if ch <= 0 || sp <= 0 {
		ch, sp = DraftANEProxyDims(entry.EmbeddingLength)
	}
	return ANEModelProxyDims{
		Tag:                 entry.Tag,
		Name:                entry.Name,
		EmbeddingLength:     entry.EmbeddingLength,
		ProxyChannels:       ch,
		ProxySpatial:        sp,
		SpecType:            entry.SpecType,
		DraftSidecarPresent: entry.DraftSidecarPresent,
		Source:              "draft_inventory",
	}
}

func proxyFromModelEntry(entry ANEModelEntry) ANEModelProxyDims {
	embed := entry.EmbeddingLength
	if embed <= 0 {
		if info, err := ProbeANEDraftGGUF(entry.GGUFPath, ""); err == nil {
			embed = info.EmbeddingLength
		}
	}
	ch, sp := DraftANEProxyDims(embed)
	return ANEModelProxyDims{
		Tag:             entry.Tag,
		Name:            entry.Name,
		EmbeddingLength: embed,
		ProxyChannels:   ch,
		ProxySpatial:    sp,
		SpecType:        entry.SpecType,
		Source:          "model_inventory",
	}
}

func prefersDraftInventory(preferred string) bool {
	if preferred == "" {
		return true
	}
	low := strings.ToLower(preferred)
	return strings.Contains(low, "dflash") ||
		strings.Contains(low, "eagle") ||
		strings.Contains(low, "mtp") ||
		strings.Contains(low, "draft")
}

// ResolveANEModelProxyDims picks draft conv proxy dims from model or dflash inventory.
func ResolveANEModelProxyDims(preferred string) (ANEModelProxyDims, error) {
	preferred = strings.TrimSpace(preferred)

	if prefersDraftInventory(preferred) {
		if entries, err := ListANEDraftInventory(); err == nil {
			if entry, ok := SelectANEDraftModel(entries, preferred); ok {
				return proxyFromDraftEntry(entry), nil
			}
		}
	}

	entries, err := ListANEModelInventory()
	if err != nil {
		return ANEModelProxyDims{}, err
	}
	entry, ok := SelectANEModel(entries, preferred)
	if !ok {
		if preferred != "" {
			return ANEModelProxyDims{}, fmt.Errorf("no local model matching %q — run ane-model-resolve", preferred)
		}
		return ANEModelProxyDims{}, fmt.Errorf("no local GGUF models")
	}
	return proxyFromModelEntry(entry), nil
}
