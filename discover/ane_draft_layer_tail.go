package discover

import (
	"fmt"
	"strconv"
)

// DraftSidecarBlockCount reads block_count from the draft sidecar GGUF.
func DraftSidecarBlockCount(entry ANEDraftEntry) (int, error) {
	draftPath, ok := resolveDraftGGUFPath(entry)
	if !ok {
		return 0, fmt.Errorf("draft sidecar missing")
	}
	info, err := ProbeANEDraftGGUF(draftPath, "")
	if err != nil {
		return 0, err
	}
	if info.BlockCount <= 0 {
		return 0, fmt.Errorf("block_count missing in %s", draftPath)
	}
	return info.BlockCount, nil
}

func resolveLayerAttnWoTensorFor(entry ANEDraftEntry, il int) string {
	for _, t := range []string{
		fmt.Sprintf("blk.%d.attn_output.weight", il),
		fmt.Sprintf("blk.%d.attn_out.weight", il),
	} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return fmt.Sprintf("blk.%d.attn_output.weight", il)
}

func resolveLayerAttnPostNormTensorFor(entry ANEDraftEntry, il int) string {
	for _, t := range []string{
		fmt.Sprintf("blk.%d.post_attention_norm.weight", il),
		fmt.Sprintf("blk.%d.attn_post_norm.weight", il),
	} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return fmt.Sprintf("blk.%d.post_attention_norm.weight", il)
}

func resolveLayerAttnNormTensorFor(entry ANEDraftEntry, il int) string {
	for _, t := range []string{
		fmt.Sprintf("blk.%d.attn_norm.weight", il),
		fmt.Sprintf("blk.%d.attn_norm", il),
	} {
		if DraftSidecarHasTensor(entry, t) {
			return t
		}
	}
	return fmt.Sprintf("blk.%d.attn_norm.weight", il)
}

// ExportDflashLayerTailEnv materializes blk.1..blk.(n-1) weights for host fp32 layer-tail replay (P38).
func ExportDflashLayerTailEnv(entry ANEDraftEntry, fcOut, matmulChain int) map[string]string {
	out := map[string]string{}
	blocks, err := DraftSidecarBlockCount(entry)
	if err != nil || blocks <= 1 {
		return out
	}
	_, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, matmulChain)
	_, ocK := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, matmulChain)
	ocV := ocK
	icWo, ocWo := DraftANEMatmulChain14AttnWoDimsForChain(entry, fcOut, matmulChain)
	icGate, ocGate := DraftANEMatmulChain15FFNGateDimsForChain(entry, fcOut, matmulChain)
	icUp, ocUp := DraftANEMatmulChain16FFNUpDimsForChain(entry, fcOut, matmulChain)
	icDown, ocDown := DraftANEMatmulChain16FFNDownDimsForChain(entry, fcOut, matmulChain)

	meta, _ := DraftANEDraftAttnHeadMeta(entry)
	headDim := 128
	if meta.HeadDim > 0 {
		headDim = meta.HeadDim
	}

	exported := 0
	for il := 1; il < blocks; il++ {
		prefix := fmt.Sprintf("ZEROLLAMA_ANE_DRAFT_LAYER_%d_", il)
		type pair struct {
			key    string
			tensor string
			ic     int
			oc     int
		}
		pairs := []pair{
			{"WQ_FILE", fmt.Sprintf("blk.%d.attn_q.weight", il), fcOut, ocQ},
			{"WK_FILE", fmt.Sprintf("blk.%d.attn_k.weight", il), fcOut, ocK},
			{"WV_FILE", fmt.Sprintf("blk.%d.attn_v.weight", il), fcOut, ocV},
			{"WO_FILE", resolveLayerAttnWoTensorFor(entry, il), icWo, ocWo},
			{"W_GATE_FILE", fmt.Sprintf("blk.%d.ffn_gate.weight", il), icGate, ocGate},
			{"W_UP_FILE", fmt.Sprintf("blk.%d.ffn_up.weight", il), icUp, ocUp},
			{"W_DOWN_FILE", fmt.Sprintf("blk.%d.ffn_down.weight", il), icDown, ocDown},
		}
		okLayer := true
		for _, p := range pairs {
			if !DraftSidecarHasTensor(entry, p.tensor) {
				okLayer = false
				break
			}
			wpath, _, err := MaterializeANEDraftMatmulWeightFile(entry, p.tensor, p.ic, p.oc)
			if err != nil || wpath == "" {
				okLayer = false
				break
			}
			out[prefix+p.key] = wpath
		}
		if !okLayer {
			continue
		}
		if ap, _, err := MaterializeANEDraftNormGammaFile(entry, resolveLayerAttnNormTensorFor(entry, il), fcOut); err == nil && ap != "" {
			out[prefix+"ATTN_NORM_FILE"] = ap
		}
		if pp, _, err := MaterializeANEDraftNormGammaFile(entry, resolveLayerAttnPostNormTensorFor(entry, il), fcOut); err == nil && pp != "" {
			out[prefix+"POST_ATTN_NORM_FILE"] = pp
		}
		qNormTensor := fmt.Sprintf("blk.%d.attn_q_norm.weight", il)
		if DraftSidecarHasTensor(entry, qNormTensor) {
			if qp, _, err := MaterializeANEDraftNormGammaFile(entry, qNormTensor, headDim); err == nil && qp != "" {
				out[prefix+"Q_NORM_FILE"] = qp
			}
		}
		kNormTensor := fmt.Sprintf("blk.%d.attn_k_norm.weight", il)
		if DraftSidecarHasTensor(entry, kNormTensor) {
			if kp, _, err := MaterializeANEDraftNormGammaFile(entry, kNormTensor, headDim); err == nil && kp != "" {
				out[prefix+"K_NORM_FILE"] = kp
			}
		}
		exported++
	}
	if exported == 0 {
		return map[string]string{}
	}
	out["ZEROLLAMA_ANE_DRAFT_HOST_LAYER_TAIL"] = "1"
	out["ZEROLLAMA_ANE_DRAFT_N_LAYER"] = strconv.Itoa(blocks)
	return out
}
