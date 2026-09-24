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

func TestFlashMoEPinBudgetGiB(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE_PIN_BUDGET_GB", "")
	if FlashMoEPinBudgetGiB() != 0 {
		t.Fatal("empty")
	}
	t.Setenv("ZEROLLAMA_FLASH_MOE_PIN_BUDGET_GB", "12.5")
	if FlashMoEPinBudgetGiB() != 12.5 {
		t.Fatalf("got %v", FlashMoEPinBudgetGiB())
	}
}
