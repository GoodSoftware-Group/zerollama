package envconfig

import "testing"

func TestGgmlMTPObserveDefaultOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_MTP", "")
	if GgmlMTPObserveEnabled() {
		t.Fatal("default on")
	}
	t.Setenv("ZEROLLAMA_GGML_MTP", "1")
	if !GgmlMTPObserveEnabled() {
		t.Fatal("want on")
	}
}
