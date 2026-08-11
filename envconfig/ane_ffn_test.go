package envconfig

import "testing"

func TestANEFFNDefaultOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_ANE_FFN", "")
	if ANEFFNEnabled() {
		t.Fatal("expected disabled")
	}
	if ANEFFNMode() != "off" {
		t.Fatalf("mode=%s", ANEFFNMode())
	}
}

func TestANEFFNShadowLab(t *testing.T) {
	t.Setenv("ZEROLLAMA_ANE_FFN", "1")
	t.Setenv("ZEROLLAMA_ANE_FFN_MODE", "shadow")
	t.Setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435")
	if !ANEFFNEnabled() || ANEFFNMode() != "shadow" {
		t.Fatalf("enabled=%v mode=%s", ANEFFNEnabled(), ANEFFNMode())
	}
	if !ANEFFNAllowServe(11435) {
		t.Fatal("lab port should allow")
	}
	if ANEFFNAllowServe(11434) {
		t.Fatal("production must refuse")
	}
	if ANEFFNAllowServe(8081) {
		t.Fatal("sidecar production must refuse")
	}
}

func TestANEFFNForceStillRefusesProd(t *testing.T) {
	t.Setenv("ZEROLLAMA_ANE_FFN", "1")
	t.Setenv("ZEROLLAMA_ANE_FFN_MODE", "force")
	t.Setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11434")
	if ANEFFNAllowServe(11434) {
		t.Fatal("force must still refuse 11434")
	}
}
