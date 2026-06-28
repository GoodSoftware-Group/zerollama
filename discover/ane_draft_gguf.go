package discover

import (
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// ANEDraftGGUFInfo is metadata from a base or draft GGUF for ANE hybrid research.
type ANEDraftGGUFInfo struct {
	Path                string `json:"path"`
	Architecture        string `json:"architecture"`
	EmbeddingLength     int    `json:"embedding_length"`
	BlockCount          int    `json:"block_count"`
	DraftSidecarPath    string `json:"draft_sidecar_path,omitempty"`
	DraftSidecarPresent bool   `json:"draft_sidecar_present"`
	ProxyChannels       int    `json:"proxy_channels"`
	ProxySpatial        int    `json:"proxy_spatial"`
	Note                string `json:"note,omitempty"`
}

// DraftANEProxyDims picks conv proxy sizes for ANE draft-step bench from embedding width.
// spatial=16 keeps MIL conv compile stable on Apple Silicon (spatial=1 fails ANE eval).
func DraftANEProxyDims(embedding int) (channels, spatial int) {
	spatial = 16
	if embedding <= 0 {
		return 64, spatial
	}
	ch := embedding / 8
	if ch < 64 {
		ch = 64
	}
	if ch > 512 {
		ch = 512
	}
	return ch, spatial
}

// ProbeANEDraftGGUF reads GGUF metadata and sidecar presence for ANE draft wiring.
func ProbeANEDraftGGUF(path, sidecarPath string) (ANEDraftGGUFInfo, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return ANEDraftGGUFInfo{}, fmt.Errorf("empty gguf path")
	}
	st, err := os.Stat(path)
	if err != nil {
		return ANEDraftGGUFInfo{}, err
	}
	if st.IsDir() {
		return ANEDraftGGUFInfo{}, fmt.Errorf("%s is a directory", path)
	}

	f, err := os.Open(path)
	if err != nil {
		return ANEDraftGGUFInfo{}, err
	}
	defer f.Close()

	m, err := ggml.DecodeMetadata(f)
	if err != nil {
		return ANEDraftGGUFInfo{}, err
	}

	kv := m.KV()
	arch := kv.Architecture()
	embed := int(kv.Uint("embedding_length"))
	if embed == 0 {
		embed = int(kv.Uint(arch + ".embedding_length"))
	}
	blocks := int(kv.Uint("block_count"))
	if blocks == 0 {
		blocks = int(kv.Uint(arch + ".block_count"))
	}

	ch, sp := DraftANEProxyDims(embed)
	info := ANEDraftGGUFInfo{
		Path:            path,
		Architecture:    arch,
		EmbeddingLength: embed,
		BlockCount:      blocks,
		ProxyChannels:   ch,
		ProxySpatial:    sp,
	}

	if sidecarPath != "" {
		info.DraftSidecarPath = sidecarPath
		if st, err := os.Stat(sidecarPath); err == nil && !st.IsDir() {
			info.DraftSidecarPresent = true
		}
	}

	if info.DraftSidecarPresent {
		info.Note = "draft sidecar GGUF present — weight extract + MIL compile is follow-on"
	} else {
		info.Note = "draft-eagle3 uses separate drafter GGUF (--spec-draft-model); eliza tags embed spec_type only"
	}
	return info, nil
}
