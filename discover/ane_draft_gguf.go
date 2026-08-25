package discover

import (
	"fmt"
	"os"
	"strconv"
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

// DraftANEMatmulChain11AttnQDims returns ic×oc for dflash_fc out @ blk.0.attn_q (P10 dflash subgraph).
func DraftANEMatmulChain11AttnQDims(entry ANEDraftEntry, fcOut int) (ic, oc int) {
	ic = fcOut
	if ic <= 0 {
		ic = entry.EmbeddingLength
	}
	oc, _ = DraftANEProxyDims(entry.EmbeddingLength)
	return ic, oc
}

// ResolveChain11AttnQTensor picks attn_q or fused attn_qkv for P10 matmul on qwen35 sidecars.
func ResolveChain11AttnQTensor(entry ANEDraftEntry) string {
	if IsNativeDflashDraftSidecar(entry) && DraftSidecarHasTensor(entry, "blk.0.attn_q.weight") {
		return "blk.0.attn_q.weight"
	}
	if DraftSidecarHasTensor(entry, "blk.0.attn_q.weight") {
		return "blk.0.attn_q.weight"
	}
	if DraftSidecarHasTensor(entry, "blk.0.attn_qkv.weight") {
		return "blk.0.attn_qkv.weight"
	}
	return "blk.0.attn_q.weight"
}

// ResolveChain11HiddenNormTensor picks RMS gamma for P10 host hidden_norm after dflash_fc.
func ResolveChain11HiddenNormTensor(entry ANEDraftEntry) string {
	if IsNativeDflashDraftSidecar(entry) && DraftSidecarHasTensor(entry, "dflash_hidden_norm.weight") {
		return "dflash_hidden_norm.weight"
	}
	for _, t := range []string{"blk.0.ffn_norm.weight", "blk.0.attn_norm.weight"} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return "blk.0.attn_norm.weight"
}

// ResolveChain15AttnPostNormTensor picks blk.0 post-attention RMS gamma before FFN (P14 dflash).
func ResolveChain15AttnPostNormTensor(entry ANEDraftEntry) string {
	for _, t := range []string{"blk.0.post_attention_norm.weight", "blk.0.attn_post_norm.weight"} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return "blk.0.post_attention_norm.weight"
}

// DraftANEDflashAttnPostNormDim returns RMS gamma width after attn residual (n_embd / fcOut).
func DraftANEDflashAttnPostNormDim(entry ANEDraftEntry, fcOut int) int {
	if fcOut > 0 {
		return fcOut
	}
	_, ocWo := DraftANEMatmulChain14AttnWoDims(entry, fcOut)
	if ocWo > 0 {
		return ocWo
	}
	return entry.EmbeddingLength
}

// ResolveChain13AttnNormTensor picks blk.0 input RMS norm applied to tok_embd before attn Q/K/V.
func ResolveChain13AttnNormTensor(entry ANEDraftEntry) string {
	for _, t := range []string{"blk.0.attn_norm.weight", "blk.0.attn_norm"} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return "blk.0.attn_norm.weight"
}

// ResolveChain13AttnQNormTensor picks per-head RMS gamma before RoPE on Q (dflash-draft blk.0).
func ResolveChain13AttnQNormTensor(entry ANEDraftEntry) string {
	for _, t := range []string{"blk.0.attn_q_norm.weight", "blk.0.attn_q_norm"} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return "blk.0.attn_q_norm.weight"
}

// ResolveChain13AttnKNormTensor picks per-head RMS gamma before RoPE on K (dflash-draft blk.0).
func ResolveChain13AttnKNormTensor(entry ANEDraftEntry) string {
	for _, t := range []string{"blk.0.attn_k_norm.weight", "blk.0.attn_k_norm"} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return "blk.0.attn_k_norm.weight"
}

// DraftANEDraftAttnUseFullDims enables full n_head/n_head_kv projections for chain >= 13 host cross-attn.
func DraftANEDraftAttnUseFullDims(matmulChain int, entry ANEDraftEntry) bool {
	return matmulChain >= 13 && IsNativeDflashDraftSidecar(entry)
}

// DraftANEDraftMatmulTensorOutDim reads the output width (last dim) of a sidecar weight tensor.
func DraftANEDraftMatmulTensorOutDim(entry ANEDraftEntry, tensorName string) (int, bool) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || strings.TrimSpace(tensorName) == "" {
		return 0, false
	}
	_, tensor, err := ggml.ReadTensorBytes(draftPath, tensorName)
	if err != nil || len(tensor.Shape) < 2 {
		return 0, false
	}
	oc := int(tensor.Shape[len(tensor.Shape)-1])
	return oc, oc > 0
}

// DraftANEDraftAttnQueryHeadCount is GQA Q head count from attn_q output width.
func DraftANEDraftAttnQueryHeadCount(entry ANEDraftEntry) int {
	meta, err := DraftANEDraftAttnHeadMeta(entry)
	if err != nil || meta.HeadDim <= 0 {
		return 0
	}
	ocQ := DraftANEDraftAttnQOutDim(entry)
	if ocQ > 0 && ocQ%meta.HeadDim == 0 {
		return ocQ / meta.HeadDim
	}
	if meta.NHead > 0 {
		return meta.NHead
	}
	return 0
}

// DraftANEDraftAttnQOutDim is Q / mha output width from sidecar attn_q (e.g. 4096 = 32×128 on eliza-27b).
func DraftANEDraftAttnQOutDim(entry ANEDraftEntry) int {
	if oc, ok := DraftANEDraftMatmulTensorOutDim(entry, ResolveChain11AttnQTensor(entry)); ok {
		return oc
	}
	meta, err := DraftANEDraftAttnHeadMeta(entry)
	if err == nil && meta.NHead > 0 && meta.HeadDim > 0 {
		return meta.NHead * meta.HeadDim
	}
	if entry.EmbeddingLength > 0 {
		return entry.EmbeddingLength
	}
	oc, _ := DraftANEProxyDims(0)
	return oc
}

// DraftANEDraftAttnKVOutDim is K/V projection width (n_head_kv × head_dim).
func DraftANEDraftAttnKVOutDim(entry ANEDraftEntry) int {
	meta, err := DraftANEDraftAttnHeadMeta(entry)
	if err != nil || meta.NHeadKV <= 0 || meta.HeadDim <= 0 {
		oc, _ := DraftANEProxyDims(entry.EmbeddingLength)
		return oc
	}
	return meta.NHeadKV * meta.HeadDim
}

// DraftANEMatmulChain11AttnQDimsForChain returns attn_q ic×oc; chain >= 13 uses full Q head width.
func DraftANEMatmulChain11AttnQDimsForChain(entry ANEDraftEntry, fcOut, matmulChain int) (ic, oc int) {
	ic = fcOut
	if ic <= 0 {
		ic = entry.EmbeddingLength
	}
	if DraftANEDraftAttnUseFullDims(matmulChain, entry) {
		oc = DraftANEDraftAttnQOutDim(entry)
		return ic, oc
	}
	return DraftANEMatmulChain11AttnQDims(entry, fcOut)
}

// DraftANEMatmulChain12AttnKVDimsForChain returns attn_k/v ic×oc; chain >= 13 uses full KV head width.
func DraftANEMatmulChain12AttnKVDimsForChain(entry ANEDraftEntry, fcOut, matmulChain int) (ic, oc int) {
	ic = fcOut
	if ic <= 0 {
		ic = entry.EmbeddingLength
	}
	if DraftANEDraftAttnUseFullDims(matmulChain, entry) {
		oc = DraftANEDraftAttnKVOutDim(entry)
		return ic, oc
	}
	return DraftANEMatmulChain12AttnKVDims(entry, fcOut)
}

// DraftANEMatmulChain14AttnWoDimsForChain returns attn_wo ic×oc; chain >= 13 uses full mha output width.
func DraftANEMatmulChain14AttnWoDimsForChain(entry ANEDraftEntry, fcOut, matmulChain int) (ic, oc int) {
	if DraftANEDraftAttnUseFullDims(matmulChain, entry) {
		ic = DraftANEDraftAttnQOutDim(entry)
		oc = fcOut
		if oc <= 0 {
			oc = entry.EmbeddingLength
		}
		return ic, oc
	}
	return DraftANEMatmulChain14AttnWoDims(entry, fcOut)
}

// ANEDraftAttnHeadMeta is RoPE/head layout from draft sidecar GGUF (host cross-attn).
type ANEDraftAttnHeadMeta struct {
	NHead      int     `json:"n_head"`
	NHeadKV    int     `json:"n_head_kv"`
	HeadDim    int     `json:"head_dim"`
	RopeNDims  int     `json:"rope_n_dims"`
	FreqBase   float64 `json:"freq_base"`
	FreqScale  float64 `json:"freq_scale"`
	NormRmsEps float64 `json:"norm_rms_eps"`
	NeoX       bool    `json:"neox"`
}

// DraftANEDraftAttnHeadKVForOC returns KV head count for a proxied attn oc slice (e.g. 512 = 4×128).
func DraftANEDraftAttnHeadKVForOC(entry ANEDraftEntry, ocAttn int) (nHeadKV, headDim int, ok bool) {
	meta, err := DraftANEDraftAttnHeadMeta(entry)
	if err != nil || meta.HeadDim <= 0 || ocAttn <= 0 || ocAttn%meta.HeadDim != 0 {
		return 0, 0, false
	}
	return ocAttn / meta.HeadDim, meta.HeadDim, true
}

// DraftANEDraftAttnHeadMeta reads attention head + RoPE metadata for host cross-attn.
func DraftANEDraftAttnHeadMeta(entry ANEDraftEntry) (ANEDraftAttnHeadMeta, error) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return ANEDraftAttnHeadMeta{}, fmt.Errorf("draft sidecar GGUF missing for %s", entry.Tag)
	}
	f, err := os.Open(draftPath)
	if err != nil {
		return ANEDraftAttnHeadMeta{}, err
	}
	defer f.Close()
	m, err := ggml.DecodeMetadata(f)
	if err != nil {
		return ANEDraftAttnHeadMeta{}, err
	}
	kv := m.KV()
	arch := kv.Architecture()
	nHead := int(kv.HeadCountMax())
	if heads := kv.HeadCount(); len(heads) > 0 && heads[0] > 0 {
		nHead = int(heads[0])
	}
	nHeadKV := int(kv.HeadCountKVMax())
	if heads := kv.HeadCountKV(); len(heads) > 0 && heads[0] > 0 {
		nHeadKV = int(heads[0])
	}
	headDim := int(kv.EmbeddingHeadCountK())
	if headDim <= 0 {
		headDim = int(kv.Uint("attention.key_length", 0))
	}
	if headDim <= 0 && entry.EmbeddingLength > 0 && nHeadKV > 0 {
		headDim = entry.EmbeddingLength / nHeadKV
	}
	freqBase := float64(kv.Float("rope.freq_base"))
	if freqBase <= 0 {
		freqBase = float64(kv.Float(arch + ".rope.freq_base"))
	}
	if freqBase <= 0 {
		freqBase = 1_000_000
	}
	freqScale := float64(kv.Float("rope.freq_scale"))
	if freqScale <= 0 {
		freqScale = float64(kv.Float(arch + ".rope.freq_scale"))
	}
	if freqScale <= 0 {
		freqScale = 1
	}
	nRot := int(kv.Uint("rope.dimension_count"))
	if nRot <= 0 {
		nRot = int(kv.Uint(arch + ".rope.dimension_count"))
	}
	if nRot <= 0 {
		nRot = headDim
	}
	normRmsEps := float64(kv.Float("attention.layer_norm_rms_epsilon"))
	if normRmsEps <= 0 {
		normRmsEps = float64(kv.Float(arch + ".attention.layer_norm_rms_epsilon"))
	}
	if normRmsEps <= 0 {
		normRmsEps = 1e-6
	}
	neox := kv.Uint("rope.scaling.type") != 0 // fallback; qwen/dflash use NeoX ordering
	_ = neox
	return ANEDraftAttnHeadMeta{
		NHead:      nHead,
		NHeadKV:    nHeadKV,
		HeadDim:    headDim,
		RopeNDims:  nRot,
		FreqBase:   freqBase,
		FreqScale:  freqScale,
		NormRmsEps: normRmsEps,
		NeoX:       true,
	}, nil
}

// DraftANEDflashHiddenNormDim returns RMS gamma width after dflash_fc (fcOut, typically n_embd).
func DraftANEDflashHiddenNormDim(entry ANEDraftEntry) int {
	gammaDim := entry.EmbeddingLength
	if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > gammaDim {
		gammaDim = ocFc
	}
	return gammaDim
}

// DraftANEMatmulChain12AttnKVDims returns ic×oc for dflash_fc out @ blk.0 attn_k/v (P11 dflash subgraph).
func DraftANEMatmulChain12AttnKVDims(entry ANEDraftEntry, fcOut int) (ic, oc int) {
	return DraftANEMatmulChain11AttnQDims(entry, fcOut)
}

// ResolveChain12AttnKTensor picks attn_k for P11 matmul on qwen35 sidecars.
func ResolveChain12AttnKTensor(entry ANEDraftEntry) string {
	if IsNativeDflashDraftSidecar(entry) && DraftSidecarHasTensor(entry, "blk.0.attn_k.weight") {
		return "blk.0.attn_k.weight"
	}
	if DraftSidecarHasTensor(entry, "blk.0.attn_k.weight") {
		return "blk.0.attn_k.weight"
	}
	return "blk.0.attn_k.weight"
}

// ResolveChain12AttnVTensor picks attn_v for P11 matmul on qwen35 sidecars.
func ResolveChain12AttnVTensor(entry ANEDraftEntry) string {
	if IsNativeDflashDraftSidecar(entry) && DraftSidecarHasTensor(entry, "blk.0.attn_v.weight") {
		return "blk.0.attn_v.weight"
	}
	if DraftSidecarHasTensor(entry, "blk.0.attn_v.weight") {
		return "blk.0.attn_v.weight"
	}
	return "blk.0.attn_v.weight"
}

// DraftANEMatmulChain14AttnWoDims returns ic×oc for host attn_out @ blk.0 attn_output (P13).
func DraftANEMatmulChain14AttnWoDims(entry ANEDraftEntry, fcOut int) (ic, oc int) {
	_, ic = DraftANEMatmulChain11AttnQDims(entry, fcOut)
	oc = fcOut
	if ic <= 0 {
		ic, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	if oc <= 0 {
		oc = entry.EmbeddingLength
	}
	return ic, oc
}

// ResolveChain14AttnWoTensor picks attn_output/wo for P13 matmul on qwen35 sidecars.
func ResolveChain14AttnWoTensor(entry ANEDraftEntry) string {
	for _, t := range []string{"blk.0.attn_output.weight", "blk.0.attn_out.weight"} {
		if IsNativeDflashDraftSidecar(entry) && DraftSidecarHasTensor(entry, t) {
			return t
		}
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return "blk.0.attn_output.weight"
}

// DraftANEDraftFFNUseFullDims enables full n_ff projections for chain >= 16 on native dflash sidecars.
func DraftANEDraftFFNUseFullDims(matmulChain int, entry ANEDraftEntry) bool {
	return matmulChain >= 16 && IsNativeDflashDraftSidecar(entry)
}

// DraftANEDraftQKVHostFP32 runs Q/K/V noise matmuls on host fp32 for chain >= 13 native dflash (P24).
func DraftANEDraftQKVHostFP32(matmulChain int, entry ANEDraftEntry) bool {
	return matmulChain >= 13 && IsNativeDflashDraftSidecar(entry)
}

// DraftANEDraftFFNOutDim is blk.0 ffn_gate output width (n_ff) from sidecar GGUF.
func DraftANEDraftFFNOutDim(entry ANEDraftEntry) int {
	if oc, ok := DraftANEDraftMatmulTensorOutDim(entry, ResolveChain15FFNGateTensor(entry)); ok && oc > 0 {
		return oc
	}
	if entry.ProxyChannels > 0 {
		return entry.ProxyChannels
	}
	oc, _ := DraftANEProxyDims(entry.EmbeddingLength)
	return oc
}

// DraftANEMatmulChain15FFNGateDims returns ic×oc for wo out @ blk.0 ffn_gate (P14 dflash subgraph).
func DraftANEMatmulChain15FFNGateDims(entry ANEDraftEntry, fcOut int) (ic, oc int) {
	_, ocWo := DraftANEMatmulChain14AttnWoDims(entry, fcOut)
	ic = ocWo
	oc = entry.ProxyChannels
	if oc <= 0 {
		oc, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	return ic, oc
}

// DraftANEMatmulChain15FFNGateDimsForChain returns ffn_gate ic×oc; chain >= 16 uses full n_ff on native dflash.
func DraftANEMatmulChain15FFNGateDimsForChain(entry ANEDraftEntry, fcOut, matmulChain int) (ic, oc int) {
	_, icWo := DraftANEMatmulChain14AttnWoDimsForChain(entry, fcOut, matmulChain)
	ic = icWo
	if ic <= 0 {
		ic = fcOut
	}
	if DraftANEDraftFFNUseFullDims(matmulChain, entry) {
		oc = DraftANEDraftFFNOutDim(entry)
		return ic, oc
	}
	return DraftANEMatmulChain15FFNGateDims(entry, fcOut)
}

// ResolveChain15FFNGateTensor picks blk.0 ffn_gate for P14 matmul on qwen35 sidecars.
func ResolveChain15FFNGateTensor(entry ANEDraftEntry) string {
	if IsNativeDflashDraftSidecar(entry) && DraftSidecarHasTensor(entry, "blk.0.ffn_gate.weight") {
		return "blk.0.ffn_gate.weight"
	}
	if DraftSidecarHasTensor(entry, "blk.0.ffn_gate.weight") {
		return "blk.0.ffn_gate.weight"
	}
	return "blk.0.ffn_gate.weight"
}

// DraftANEMatmulChain16FFNUpDims returns ic×oc for wo out @ blk.0 ffn_up (P15 dflash subgraph).
func DraftANEMatmulChain16FFNUpDims(entry ANEDraftEntry, fcOut int) (ic, oc int) {
	return DraftANEMatmulChain15FFNGateDims(entry, fcOut)
}

// DraftANEMatmulChain16FFNUpDimsForChain returns ffn_up ic×oc; chain >= 16 uses full n_ff on native dflash.
func DraftANEMatmulChain16FFNUpDimsForChain(entry ANEDraftEntry, fcOut, matmulChain int) (ic, oc int) {
	return DraftANEMatmulChain15FFNGateDimsForChain(entry, fcOut, matmulChain)
}

// DraftANEMatmulChain16FFNDownDims returns ic×oc for SwiGLU @ blk.0 ffn_down (P15 dflash subgraph).
func DraftANEMatmulChain16FFNDownDims(entry ANEDraftEntry, fcOut int) (ic, oc int) {
	_, icFF := DraftANEMatmulChain15FFNGateDims(entry, fcOut)
	oc = fcOut
	if oc <= 0 {
		oc = entry.EmbeddingLength
	}
	return icFF, oc
}

// DraftANEMatmulChain16FFNDownDimsForChain returns ffn_down ic×oc; chain >= 16 uses full n_ff on native dflash.
func DraftANEMatmulChain16FFNDownDimsForChain(entry ANEDraftEntry, fcOut, matmulChain int) (ic, oc int) {
	_, icFF := DraftANEMatmulChain15FFNGateDimsForChain(entry, fcOut, matmulChain)
	oc = fcOut
	if oc <= 0 {
		oc = entry.EmbeddingLength
	}
	return icFF, oc
}

// ResolveChain16FFNUpTensor picks blk.0 ffn_up for P15 matmul on qwen35 sidecars.
func ResolveChain16FFNUpTensor(entry ANEDraftEntry) string {
	if DraftSidecarHasTensor(entry, "blk.0.ffn_up.weight") {
		return "blk.0.ffn_up.weight"
	}
	return "blk.0.ffn_up.weight"
}

// ResolveChain16FFNDownTensor picks blk.0 ffn_down for P15 host matmul on qwen35 sidecars.
func ResolveChain16FFNDownTensor(entry ANEDraftEntry) string {
	if DraftSidecarHasTensor(entry, "blk.0.ffn_down.weight") {
		return "blk.0.ffn_down.weight"
	}
	return "blk.0.ffn_down.weight"
}

// DraftANEHandoffStride picks IOSurface handoff frequency per decode step for e2e A/B.
// P3 matmul runs 3 ANE evals per handoff; async eval overlaps with Metal decode, so stride 8
// cuts handoff rate without blocking the draft loop. P1/P2 stay at stride 4.
func DraftANEHandoffStride(kernel string, matmulChain int) int {
	if strings.EqualFold(strings.TrimSpace(kernel), "matmul") {
		if matmulChain >= 17 {
			return 32
		}
		if matmulChain >= 12 {
			return 20
		}
		if matmulChain >= 10 {
			return 28
		}
		if matmulChain >= 9 {
			return 24
		}
		if matmulChain >= 8 {
			return 16
		}
		if matmulChain >= 7 {
			return 20
		}
		if matmulChain >= 6 {
			return 16
		}
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

// DraftANEMatmulChain6QKVDims returns ic×oc for h @ attn_qkv prefix (top-left slice).
func DraftANEMatmulChain6QKVDims(gateIC, gateOC int) (ic6, oc6 int) {
	return gateIC, gateOC
}

// DraftANEMatmulChain7Blk1GateDims returns ic×oc for blk.1 ffn_gate on post-blk.0 hidden (qwen35 proxy).
func DraftANEMatmulChain7Blk1GateDims(ffnEmbd, ffWidth int) (ic7, oc7 int) {
	return ffnEmbd, ffWidth
}

// DraftANEMatmulChain9Blk1UpDims returns ic×oc for blk.1 ffn_up on post-blk.0 hidden (qwen35 proxy).
func DraftANEMatmulChain9Blk1UpDims(ffnEmbd, ffWidth int) (ic9, oc9 int) {
	return ffnEmbd, ffWidth
}

// DraftANEMatmulChain10Blk1DownDims returns ic×oc for blk.1 ffn_down after blk.1 SwiGLU.
func DraftANEMatmulChain10Blk1DownDims(ffWidth, ffnEmbd int) (ic10, oc10 int) {
	return ffWidth, ffnEmbd
}

// DraftANEMatmulChain7DflashFcDims returns ic×oc for target_hidden @ dflash_fc (top-left lab slice).
// Full tensor is [n_target_features × n_embd]; lab uses proxyChannels × embeddingLength when smaller.
func DraftANEMatmulChain7DflashFcDims(entry ANEDraftEntry) (ic7, oc7 int, ok bool) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return 0, 0, false
	}
	_, tensor, err := ggml.ReadTensorBytes(draftPath, "dflash_fc.weight")
	if err != nil || tensor == nil || len(tensor.Shape) < 2 {
		return 0, 0, false
	}
	ic7 = int(tensor.Shape[0])
	oc7 = int(tensor.Shape[1])
	if ic7 <= 0 || oc7 <= 0 {
		return 0, 0, false
	}
	proxyOC := entry.ProxyChannels
	if proxyOC <= 0 {
		proxyOC, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	proxyIC := entry.EmbeddingLength
	if proxyIC <= 0 {
		proxyIC = oc7
	}
	if proxyOC > 0 && proxyOC < ic7 {
		ic7 = proxyOC
	}
	if proxyIC > 0 && proxyIC < oc7 {
		oc7 = proxyIC
	}
	return ic7, oc7, true
}

// DraftANEMatmulChain7DflashFcNativeDims returns full dflash_fc ic×oc from GGUF (no proxy slice).
func DraftANEMatmulChain7DflashFcNativeDims(entry ANEDraftEntry) (ic, oc int, ok bool) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return 0, 0, false
	}
	_, tensor, err := ggml.ReadTensorBytes(draftPath, "dflash_fc.weight")
	if err != nil || tensor == nil || len(tensor.Shape) < 2 {
		return 0, 0, false
	}
	ic = int(tensor.Shape[0])
	oc = int(tensor.Shape[1])
	if ic <= 0 || oc <= 0 {
		return 0, 0, false
	}
	return ic, oc, true
}

// DraftDflashTargetMeta reads dflash-draft sidecar KV for cross.v_embd wiring (B8 lab metadata).
func DraftDflashTargetMeta(entry ANEDraftEntry) (nFeatures int, layerIDs []uint32, ok bool) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return 0, nil, false
	}
	f, err := os.Open(draftPath)
	if err != nil {
		return 0, nil, false
	}
	defer f.Close()
	meta, err := ggml.DecodeMetadataArrays(f, 64)
	if err != nil {
		return 0, nil, false
	}
	kv := meta.KV()
	if !strings.EqualFold(strings.TrimSpace(kv.Architecture()), "dflash-draft") {
		return 0, nil, false
	}
	if v := kv.Uint("dflash.n_target_features"); v > 0 {
		nFeatures = int(v)
	}
	layerIDs = kv.UintOrArrayValueAsArray("dflash.target_layer_ids", 0)
	// Drop placeholder default when array decode failed or KV missing.
	if len(layerIDs) == 1 && layerIDs[0] == 0 && nFeatures > 0 {
		layerIDs = nil
	}
	return nFeatures, layerIDs, nFeatures > 0 || len(layerIDs) > 0
}

// ExportDflashTargetMetaEnv maps sidecar dflash-draft KV into hook env vars (B8 cross pack).
func ExportDflashTargetMetaEnv(entry ANEDraftEntry) map[string]string {
	nFeat, layers, ok := DraftDflashTargetMeta(entry)
	if !ok {
		return nil
	}
	out := make(map[string]string)
	if nFeat > 0 {
		out["ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_FEATURES"] = strconv.Itoa(nFeat)
	}
	if len(layers) > 0 {
		out["ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_LAYERS"] = strconv.Itoa(len(layers))
	}
	return out
}

// DraftSidecarHasTensor reports whether the draft sidecar GGUF contains tensorName.
func DraftSidecarHasTensor(entry ANEDraftEntry, tensorName string) bool {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || strings.TrimSpace(tensorName) == "" {
		return false
	}
	_, _, err := ggml.ReadTensorBytes(draftPath, tensorName)
	return err == nil
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
