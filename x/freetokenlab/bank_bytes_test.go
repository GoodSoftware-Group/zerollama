package freetokenlab

import "testing"

func TestBytesPerExpertLayerQ40(t *testing.T) {
	// FreeToken: 2*I*(H/32)*18 + H*(I/32)*18 for H=2048 I=768
	got := BytesPerExpertLayer("q4_0", 2048, 768)
	want := int64(2*768*(2048/32)*18 + 2048*(768/32)*18)
	if got != want {
		t.Fatalf("got %d want %d", got, want)
	}
	if BytesPerExpertLayer("q4_0", 0, 768) != 0 {
		t.Fatal("zero hidden")
	}
}

func TestBankBytesEstimateMatchesFreeToken(t *testing.T) {
	d := ExpertBankDims{Layers: 40, Experts: 256, Hidden: 2048, Inter: 768, Format: "q4_0"}
	per := BytesPerExpertLayer("q4_0", 2048, 768)
	want := int64(40) * 256 * per
	if got := BankBytesEstimate(d); got != want {
		t.Fatalf("estimate %d want %d", got, want)
	}
	if BytesPerExpertSlot(d) != want/256 {
		t.Fatalf("per-slot %d", BytesPerExpertSlot(d))
	}
}

func TestResolveExpertBankBytesPrefersMeasured(t *testing.T) {
	d := ExpertBankDims{Layers: 8, Experts: 64, Hidden: 1024, Inter: 512, Format: "q4_0"}
	est := BankBytesEstimate(d)
	got, src := ResolveExpertBankBytes(99, d)
	if got != 99 || src != "measured" {
		t.Fatalf("got %d %q", got, src)
	}
	got, src = ResolveExpertBankBytes(0, d)
	if got != est || src != "estimate" {
		t.Fatalf("fallback %d %q want %d estimate", got, src, est)
	}
}

func TestAdviseSlotBankDimsUsesEstimate(t *testing.T) {
	d := ExpertBankDims{Layers: 8, Experts: 256, Hidden: 2048, Inter: 768, Format: "q4_0"}
	a := AdviseSlotBankDims(256, 8, 128, 0, d)
	if a.BankSource != "estimate" || a.BankBytes < 1 || a.BytesPerSlot < 1 {
		t.Fatalf("%+v", a)
	}
	// slots * estimate / experts
	want := SlotBankBytes(a.Recommend, BankBytesEstimate(d), 256)
	if a.BankBytes != want {
		t.Fatalf("bank %d want %d", a.BankBytes, want)
	}
}

func TestNormalizeEmptyFormatDefaultsQ40(t *testing.T) {
	if normalizeBankFormat("") != "q4_0" {
		t.Fatal(normalizeBankFormat(""))
	}
	if BytesPerExpertLayer("", 2048, 768) != BytesPerExpertLayer("q4_0", 2048, 768) {
		t.Fatal("empty format should size as q4_0")
	}
}
