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

// DraftANEMatmulDims picks matmul ic×oc for blk.0 ffn_gate.
// ic uses full draft n_embd; static BLOBFILE MIL may fail at ic>256 — session falls back to dynamic MIL.
func DraftANEMatmulDims(entry ANEDraftEntry) (ic, oc, seq int) {
	seq = entry.ProxySpatial
	if seq < 16 {
		seq = 16
	}
	oc = entry.ProxyChannels
	if oc <= 0 {
		oc, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	ic = entry.EmbeddingLength
	if ic <= 0 {
		ic = oc
	}
	return ic, oc, seq
}

// DraftANEHandoffStride picks IOSurface handoff frequency per decode step for e2e A/B.
// P3 matmul runs 3 ANE evals per handoff; async eval overlaps with Metal decode, so stride 8
// cuts handoff rate without blocking the draft loop. P1/P2 stay at stride 4.
func DraftANEHandoffStride(kernel string, matmulChain int) int {
	if strings.EqualFold(strings.TrimSpace(kernel), "matmul") {
		if matmulChain >= 5 {
			return 12
		}
		if matmulChain >= 3 {
			return 8
		}
		return 4
	}
	return 2
}

// DraftANEMatmulChain2Dims returns ic×oc for silu(gate) @ ffn_up after gate matmul (P2 legacy).
func DraftANEMatmulChain2Dims(gateIC, gateOC int) (ic2, oc2 int) {
	return gateOC, gateIC
}

// DraftANEMatmulChain3UpDims returns ic×oc for h @ ffn_up in SwiGLU chain (same ff width as gate).
func DraftANEMatmulChain3UpDims(gateIC, gateOC int) (ic2, oc2 int) {
	return gateIC, gateOC
}

// DraftANEMatmulChain3DownDims returns ic×oc for swiglu @ ffn_down.
func DraftANEMatmulChain3DownDims(gateOC, gateIC int) (ic3, oc3 int) {
	return gateOC, gateIC
}

// DraftANEMatmulChain4AttnGateDims returns ic×oc for ffn_down @ attn_gate (P4 B8 matmul).
func DraftANEMatmulChain4AttnGateDims(ffnEmbd, ffWidth int) (ic4, oc4 int) {
	return ffnEmbd, ffWidth
}

// DraftANEMatmulChain5SSMOutDims returns ic×oc for ffn_down @ ssm_out (P5 hybrid block).
func DraftANEMatmulChain5SSMOutDims(ffnEmbd, ffWidth int) (ic5, oc5 int) {
	return ffnEmbd, ffWidth
}

// ANEDraftNeedsDriveHead is true when B7 must mmap tied-embed for token argmax (force, or conv shadow).
func ANEDraftNeedsDriveHead(kernel, driveMode string) bool {
	return ANEDraftNeedsDriveHeadWithMetrics(kernel, driveMode, "")
}

// ANEDraftNeedsDriveHeadWithMetrics extends drive-head detection for matmul shadow token parity.
func ANEDraftNeedsDriveHeadWithMetrics(kernel, driveMode, driveMetrics string) bool {
	driveMode = strings.TrimSpace(driveMode)
	driveMetrics = strings.TrimSpace(strings.ToLower(driveMetrics))
	if driveMode == "" {
		return false
	}
	if driveMode == "force" {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(kernel), "matmul") {
		return driveMetrics == "tokens" || driveMetrics == "both"
	}
	return true
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
