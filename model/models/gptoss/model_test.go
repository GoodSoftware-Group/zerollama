package gptoss

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

func TestTransformerBlockNormGGUFTag(t *testing.T) {
	field, ok := reflect.TypeOf(TransformerBlock{}).FieldByName("PostAttentionNorm")
	if !ok {
		t.Fatal("PostAttentionNorm field missing")
	}
	tag := field.Tag.Get("gguf")
	if tag != "post_attention_norm,alt:ffn_norm" {
		t.Fatalf("gguf tag=%q want post_attention_norm prioritized", tag)
	}
}

// stubTensor is only used as a non-nil Weight marker for PostLoad completeness checks.
type stubTensor struct{ ml.Tensor }

func TestBlockHasAcceptsFusedLayouts(t *testing.T) {
	w := stubTensor{}
	fused := TransformerBlock{
		Attention: &AttentionBlock{
			QKV:    &nn.Linear{Weight: w},
			Output: &nn.Linear{Weight: w},
		},
		MLP: &MLPBlock{
			Router: &nn.Linear{Weight: w},
			GateUp: &nn.LinearBatch{Weight: w},
			Down:   &nn.LinearBatch{Weight: w},
		},
	}
	if !blockHasAttention(fused) || !blockHasMoE(fused) {
		t.Fatal("fused attn_qkv + ffn_gate_up_exps must count as complete")
	}

	split := TransformerBlock{
		Attention: &AttentionBlock{
			Query:  &nn.Linear{Weight: w},
			Key:    &nn.Linear{Weight: w},
			Value:  &nn.Linear{Weight: w},
			Output: &nn.Linear{Weight: w},
		},
		MLP: &MLPBlock{
			Router: &nn.Linear{Weight: w},
			Gate:   &nn.LinearBatch{Weight: w},
			Up:     &nn.LinearBatch{Weight: w},
			Down:   &nn.LinearBatch{Weight: w},
		},
	}
	if !blockHasAttention(split) || !blockHasMoE(split) {
		t.Fatal("split q/k/v + gate/up must still count as complete")
	}

	empty := TransformerBlock{Attention: &AttentionBlock{}, MLP: &MLPBlock{}}
	if blockHasAttention(empty) || blockHasMoE(empty) {
		t.Fatal("empty blocks must not count as complete")
	}
}
