package envconfig

import "testing"

func TestGgmlClampNumCtxEnabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_CLAMP_NUM_CTX", "")
	if GgmlClampNumCtxEnabled() {
		t.Fatal("expected default off")
	}
	t.Setenv("ZEROLLAMA_GGML_CLAMP_NUM_CTX", "1")
	if !GgmlClampNumCtxEnabled() {
		t.Fatal("expected on for 1")
	}
	t.Setenv("ZEROLLAMA_GGML_CLAMP_NUM_CTX", "off")
	if GgmlClampNumCtxEnabled() {
		t.Fatal("expected off")
	}
	t.Setenv("ZEROLLAMA_GGML_CLAMP_NUM_CTX", "auto")
	if !GgmlClampNumCtxEnabled() {
		t.Fatal("expected on for auto")
	}
}

func TestGgmlSuggestCtxMaxCap(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_SUGGEST_CTX_MAX", "")
	if got := GgmlSuggestCtxMaxCap(); got != 131072 {
		t.Fatalf("default cap = %d", got)
	}
	t.Setenv("ZEROLLAMA_GGML_SUGGEST_CTX_MAX", "8192")
	if got := GgmlSuggestCtxMaxCap(); got != 8192 {
		t.Fatalf("cap = %d", got)
	}
}
