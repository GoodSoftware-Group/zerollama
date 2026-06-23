package gptoss

import (
	"reflect"
	"testing"
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
