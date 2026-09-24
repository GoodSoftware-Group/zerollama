package freetokenlab

// Expert-bank byte estimates rematched from FlashML FreeToken
// (python/freetoken/moe/offload_cache.py _BANK_BYTES_PER_EXPERT +
// expert_banks.bank_bytes_estimate). Used when GGUF *_exps tensor sizes are
// missing so pin-budget / slot-bank advice still has a number.

// ExpertBankDims is MoE geometry from model config / GGUF KV (no weight load).
type ExpertBankDims struct {
	Layers  int    // MoE / transformer layers that carry routed experts
	Experts int    // expert_count
	Hidden  int    // embedding_length
	Inter   int    // expert_feed_forward_length (moe_intermediate_size)
	Format  string // bf16 | q4_0 | fp8_block | nvfp4 | mxfp4 | ds_fp4
}

// BytesPerExpertLayer is host/GPU bytes for one expert on one MoE layer
// (gate_up + down for that format). Unknown format or dims → 0.
func BytesPerExpertLayer(format string, hidden, inter int) int64 {
	if hidden < 1 || inter < 1 {
		return 0
	}
	H, I := int64(hidden), int64(inter)
	switch normalizeBankFormat(format) {
	case "bf16":
		// 3 × I × H × 2 (gate, up, down)
		return 3 * I * H * 2
	case "fp8_block":
		// FreeToken: weights + block scales (simplified pad = exact ceil blocks)
		scale := (2*I/128)*(H/128) + (H/128)*(I/128)
		return 3*I*H + scale*2
	case "q4_0":
		// 2×I×(H/32)×18 + H×(I/32)×18 (gate_up + down, Q4_0 blocks)
		return 2*I*(H/32)*18 + H*(I/32)*18
	case "nvfp4":
		return 2*I*(H/2+H/16+2) + H*(I/2+I/16+2)
	case "mxfp4":
		return 2*I*(H/2+H/32+2) + H*(I/2+I/32+2)
	case "ds_fp4":
		return 2*I*(H/2+H/32) + H*(I/2+I/32)
	default:
		return 0
	}
}

// BankBytesEstimate is total expert-bank bytes for a raw checkpoint:
// layers × experts × BytesPerExpertLayer (FreeToken bank_bytes_estimate).
func BankBytesEstimate(d ExpertBankDims) int64 {
	if d.Layers < 1 || d.Experts < 1 {
		return 0
	}
	per := BytesPerExpertLayer(d.Format, d.Hidden, d.Inter)
	if per < 1 {
		return 0
	}
	return int64(d.Layers) * int64(d.Experts) * per
}

// BytesPerExpertSlot is bytes one slot-bank entry occupies across all MoE
// layers (FreeToken expert_bytes_per_slot for a full-layer bank set):
// layers × BytesPerExpertLayer == BankBytesEstimate / experts.
func BytesPerExpertSlot(d ExpertBankDims) int64 {
	if d.Experts < 1 {
		return 0
	}
	total := BankBytesEstimate(d)
	if total < 1 {
		return 0
	}
	return total / int64(d.Experts)
}

// ResolveExpertBankBytes prefers measured GGUF *_exps sum; else FreeToken
// config estimate. source is "measured", "estimate", or "".
func ResolveExpertBankBytes(measured int64, d ExpertBankDims) (bytes int64, source string) {
	if measured > 0 {
		return measured, "measured"
	}
	if est := BankBytesEstimate(d); est > 0 {
		return est, "estimate"
	}
	return 0, ""
}

func normalizeBankFormat(format string) string {
	switch format {
	case "", "f16", "float16", "F16":
		// GGUF F16 experts ≈ bf16 row width for sizing
		if format == "" {
			return "q4_0" // Flash-MoE / anemll default lab assumption
		}
		return "bf16"
	case "BF16", "bfloat16":
		return "bf16"
	case "Q4_0", "q4_0":
		return "q4_0"
	case "fp8", "FP8":
		return "fp8_block"
	case "NVFP4":
		return "nvfp4"
	case "MXFP4":
		return "mxfp4"
	case "DS_FP4", "dsfp4":
		return "ds_fp4"
	default:
		return format
	}
}
