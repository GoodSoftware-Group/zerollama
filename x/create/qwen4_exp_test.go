package create

import "testing"

func TestQwen4ExpQuantizationPolicy(t *testing.T) {
	tr := qwen4ExpImportTransform{}
	ple := "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
	if got := tr.quantizationType(ple, []int32{128, 256}, "int4"); got != "int4" {
		t.Fatalf("PLE shard quant = %q, want int4", got)
	}
	if got := tr.quantizationType(ple, []int32{128, 100}, "int4"); got != "" {
		t.Fatalf("unaligned PLE shard quant = %q, want empty", got)
	}
	low := "model.language_model.layers.0.linear_attn.in_proj_a.weight"
	if got := tr.quantizationType(low, []int32{4096, 256}, "int8"); got != "" {
		t.Fatalf("low-rank projection quant = %q, want empty", got)
	}
	embed := "model.language_model.embed_tokens.weight"
	if got := tr.quantizationType(embed, []int32{1024, 256}, "int4"); got != "" {
		t.Fatalf("token embedding quant = %q, want empty", got)
	}
}
