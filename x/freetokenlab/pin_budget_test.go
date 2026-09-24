package freetokenlab

import (
	"reflect"
	"testing"
)

func TestPinBudgetBytesUncapped(t *testing.T) {
	b, capped := PinBudgetBytes(PinBudgetOpts{HostRAMBytes: 64 << 30, WSL: false})
	if capped || b != 0 {
		t.Fatalf("plain host got budget=%d capped=%v", b, capped)
	}
}

func TestPinBudgetBytesWSL40(t *testing.T) {
	const ram = 64 << 30
	b, capped := PinBudgetBytes(PinBudgetOpts{HostRAMBytes: ram, WSL: true})
	wantF := float64(ram) * DefaultWSLPinFrac
	want := int64(wantF)
	if !capped || b != want {
		t.Fatalf("got %d capped=%v want %d", b, capped, want)
	}
	b2, _ := PinBudgetBytes(PinBudgetOpts{HostRAMBytes: ram, WSL: true, ReservedBytes: 1 << 30})
	if b2 != want-(1<<30) {
		t.Fatalf("reserved: %d", b2)
	}
}

func TestPinBudgetBytesOverride(t *testing.T) {
	b, capped := PinBudgetBytes(PinBudgetOpts{PinBudgetGiB: 10, WSL: false})
	if !capped || b != 10<<30 {
		t.Fatalf("override got %d capped=%v", b, capped)
	}
}

func TestAutoCPULayerIDsHeadTail(t *testing.T) {
	// 40 layers, banks 2× budget → lock ceil(40*(1-0.5))=20 → head 10 + tail 10
	ids := AutoCPULayerIDs(40, 20<<30, 10<<30)
	if len(ids) != 20 {
		t.Fatalf("n=%d %v", len(ids), ids)
	}
	wantHead := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	wantTail := []int{30, 31, 32, 33, 34, 35, 36, 37, 38, 39}
	if !reflect.DeepEqual(ids[:10], wantHead) || !reflect.DeepEqual(ids[10:], wantTail) {
		t.Fatalf("ids=%v", ids)
	}
	if AutoCPULayerCount(40, 5<<30, 10<<30) != 0 {
		t.Fatal("under budget should lock none")
	}
}

func TestAdvisePinOverBudget(t *testing.T) {
	a := AdvisePin(PinBudgetOpts{PinBudgetGiB: 8}, 20<<30, 32)
	if !a.Capped || !a.OverBudget || a.SuggestedCPULayers < 1 {
		t.Fatalf("%+v", a)
	}
	if len(a.Notes) < 1 || a.Notes[0] == "" {
		t.Fatal("expected note")
	}
	ok := AdvisePin(PinBudgetOpts{PinBudgetGiB: 32}, 20<<30, 32)
	if ok.OverBudget || ok.SuggestedCPULayers != 0 {
		t.Fatalf("under: %+v", ok)
	}
	unc := AdvisePin(PinBudgetOpts{HostRAMBytes: 64 << 30, WSL: false}, 20<<30, 32)
	if unc.Capped || unc.OverBudget {
		t.Fatalf("uncapped: %+v", unc)
	}
}
