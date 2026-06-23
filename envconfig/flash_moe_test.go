package envconfig

import "testing"

func TestFlashMoEEnabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE", "")
	if FlashMoEEnabled() {
		t.Fatal("expected disabled")
	}
	t.Setenv("ZEROLLAMA_FLASH_MOE", "1")
	if !FlashMoEEnabled() {
		t.Fatal("expected enabled")
	}
}

func TestFlashMoEModeDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE_MODE", "")
	if FlashMoEMode() != "slot-bank" {
		t.Fatalf("mode = %q", FlashMoEMode())
	}
}
