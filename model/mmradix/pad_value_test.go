package mmradix

import (
	"testing"

	"github.com/ollama/ollama/model/input"
)

func TestPadValueFromHash(t *testing.T) {
	p := PadValueFromHash(0)
	if p != MMPadShiftValue {
		t.Fatalf("got %d want %d", p, MMPadShiftValue)
	}
	p2 := PadValueFromHash(1 << 30)
	if p2 != MMPadShiftValue {
		t.Fatalf("mod wrap: got %d", p2)
	}
	if PadValueFromHash(42) == PadValueFromHash(43) {
		t.Fatal("different hashes must differ in low bits")
	}
}

func TestClampForEmbed(t *testing.T) {
	if ClampForEmbed(MMPadShiftValue+9, 151936) != 151935 {
		t.Fatal("pad_value should clamp to vocab-1")
	}
	if ClampForEmbed(100, 151936) != 100 {
		t.Fatal("in-vocab unchanged")
	}
	if ClampForEmbed(-1, 10) != 0 {
		t.Fatal("negative → 0")
	}
}

func TestApplyToInputs_qwenStyleSpan(t *testing.T) {
	hash := uint64(0xabc)
	want := PadValueFromHash(hash)
	inputs := []*input.Input{
		{Token: 1},
		{Token: 151652}, // vision_start — not rewritten (no hash)
		{
			Token:          151655,
			MultimodalHash: hash,
			SameBatch:      4,
			Multimodal:     []input.Multimodal{{}},
		},
		{Token: 151655},
		{Token: 151655},
		{Token: 151655},
		{Token: 151653}, // vision_end
		{Token: 2},
	}
	n := ApplyToInputs(inputs)
	if n < 4 {
		t.Fatalf("rewrote %d want >=4", n)
	}
	if inputs[2].Token != want || inputs[3].Token != want || inputs[5].Token != want {
		t.Fatalf("pad span tokens: %+v", []int32{inputs[2].Token, inputs[3].Token, inputs[5].Token})
	}
	if inputs[3].MultimodalHash != hash || inputs[5].MultimodalHash != hash {
		t.Fatal("trailing pads should inherit MultimodalHash")
	}
	if inputs[1].Token != 151652 || inputs[6].Token != 151653 {
		t.Fatal("boundary tokens must stay")
	}
}

func TestApplyToInputs_differentImagesDiverge(t *testing.T) {
	a := []*input.Input{{Token: 151655, MultimodalHash: 1, SameBatch: 2, Multimodal: []input.Multimodal{{}}}, {Token: 151655}}
	b := []*input.Input{{Token: 151655, MultimodalHash: 2, SameBatch: 2, Multimodal: []input.Multimodal{{}}}, {Token: 151655}}
	ApplyToInputs(a)
	ApplyToInputs(b)
	if a[0].Token == b[0].Token {
		t.Fatal("different image hashes must produce different pad_values")
	}
}
