package envconfig

import "testing"

func TestParseVRAMBudget(t *testing.T) {
	b, err := ParseVRAMBudget("80%")
	if err != nil || !b.IsSet() || b.fraction != 0.8 {
		t.Fatalf("80%%: %+v %v", b, err)
	}
	b, err = ParseVRAMBudget("0.8")
	if err != nil || b.fraction != 0.8 {
		t.Fatalf("0.8: %+v %v", b, err)
	}
	b, err = ParseVRAMBudget("12GiB")
	if err != nil || b.absolute != 12<<30 {
		t.Fatalf("12GiB: %+v %v", b, err)
	}
	b, err = ParseVRAMBudget("12GB")
	if err != nil || b.absolute != 12_000_000_000 {
		t.Fatalf("12GB: %+v %v", b, err)
	}
	if _, err := ParseVRAMBudget("120%"); err == nil {
		t.Fatal("120% should error")
	}
	b, err = ParseVRAMBudget("")
	if err != nil || b.IsSet() {
		t.Fatalf("empty: %+v %v", b, err)
	}
}

func TestVRAMBudgetApply(t *testing.T) {
	b, _ := ParseVRAMBudget("80%")
	phys := uint64(16) << 30
	total, free := b.Apply(phys, phys)
	want := uint64(float64(phys) * 0.8)
	if total != want || free != want {
		t.Fatalf("empty gpu: total=%d free=%d want=%d", total, free, want)
	}
	total, free = b.Apply(phys, 4<<30)
	if total != want || free != 4<<30 {
		t.Fatalf("partial: total=%d free=%d", total, free)
	}
	abs, _ := ParseVRAMBudget("64GiB")
	total, free = abs.Apply(phys, phys)
	if total != phys || free != phys {
		t.Fatalf("over-physical must clamp: %d %d", total, free)
	}
	unset := VRAMBudget{}
	total, free = unset.Apply(phys, 8<<30)
	if total != phys || free != 8<<30 {
		t.Fatalf("unset must pass through")
	}
}

func TestVRAMBudgetFromEnvInvalid(t *testing.T) {
	t.Setenv("ZEROLLAMA_VRAM_BUDGET", "nope")
	if VRAMBudgetFromEnv().IsSet() {
		t.Fatal("invalid should be unset")
	}
	t.Setenv("ZEROLLAMA_VRAM_BUDGET", "50%")
	b := VRAMBudgetFromEnv()
	if !b.IsSet() || b.fraction != 0.5 {
		t.Fatalf("got %+v", b)
	}
}
