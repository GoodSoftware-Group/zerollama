package discover

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestPackANEMILWeightBlobLayout(t *testing.T) {
	ch := 4
	fp16 := make([]byte, ch*ch*2)
	for i := range ch * ch {
		binary.LittleEndian.PutUint16(fp16[i*2:], 0x3400)
	}
	blob, err := PackANEMILWeightBlob(fp16)
	if err != nil {
		t.Fatal(err)
	}
	if len(blob) != 64+64+ch*ch*2 {
		t.Fatalf("len = %d", len(blob))
	}
	chunk := blob[64:]
	if chunk[0] != 0xEF || chunk[1] != 0xBE || chunk[2] != 0xAD || chunk[3] != 0xDE {
		t.Fatalf("chunk magic: % x", chunk[:4])
	}
	if chunk[4] != 0x01 || chunk[10] != 0x08 {
		t.Fatalf("chunk header: % x", chunk[:16])
	}
	if !bytes.Equal(blob[64+64:], fp16) {
		t.Fatal("fp16 payload mismatch")
	}
}

func TestExtractTopLeftSquareFP16FromF32(t *testing.T) {
	rows, cols, ch := 4, 6, 3
	raw := make([]byte, rows*cols*4)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := float32(float64(r*10+c) + 0.5)
			binary.LittleEndian.PutUint32(raw[(r*cols+c)*4:], math.Float32bits(v))
		}
	}
	tensor := &ggml.Tensor{
		Name:  "blk.0.ffn_gate.weight",
		Kind:  uint32(ggml.TensorTypeF32),
		Shape: []uint64{uint64(rows), uint64(cols)},
	}
	out, err := ExtractTopLeftSquareFP16(raw, tensor, ch)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != ch*ch*2 {
		t.Fatalf("len = %d", len(out))
	}
	want := float32(0.5)
	got := math.Float32frombits(float32ToFloat16BitsToFloat32(binary.LittleEndian.Uint16(out[0:2])))
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("(0,0) = %v want %v", got, want)
	}
	want = float32(22.5)
	got = math.Float32frombits(float32ToFloat16BitsToFloat32(binary.LittleEndian.Uint16(out[(2*ch+2)*2 : (2*ch+2)*2+2])))
	if math.Abs(float64(got-want)) > 1e-2 {
		t.Fatalf("(2,2) = %v want %v", got, want)
	}
}

func float32ToFloat16BitsToFloat32(h uint16) uint32 {
	// round-trip helper for test only (via float64)
	sign := uint32(h>>15) << 31
	exp := (h >> 10) & 0x1f
	mant := h & 0x3ff
	switch exp {
	case 0:
		if mant == 0 {
			return sign
		}
		// subnormal
		f := float64(mant) / 1024.0 * math.Pow(2, -14)
		if sign != 0 {
			f = -f
		}
		return math.Float32bits(float32(f))
	case 31:
		if mant != 0 {
			return sign | 0x7fc00000
		}
		return sign | 0x7f800000
	default:
		f := math.Ldexp(1+float64(mant)/1024, int(exp)-15)
		if sign != 0 {
			f = -f
		}
		return math.Float32bits(float32(f))
	}
}

func TestExtractTopLeftSquareBF16(t *testing.T) {
	rows, cols, ch := 4, 6, 3
	raw := make([]byte, rows*cols*2)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			f := float32(float64(r*10+c) + 0.25)
			binary.LittleEndian.PutUint16(raw[(r*cols+c)*2:], float32ToBFloat16Bits(f))
		}
	}
	tensor := &ggml.Tensor{
		Name:  "blk.0.ffn_gate.weight",
		Kind:  uint32(ggml.TensorTypeBF16),
		Shape: []uint64{uint64(rows), uint64(cols)},
	}
	out, err := ExtractTopLeftSquareFP16(raw, tensor, ch)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != ch*ch*2 {
		t.Fatalf("len = %d", len(out))
	}
}

func float32ToBFloat16Bits(f float32) uint16 {
	return uint16(math.Float32bits(f) >> 16)
}

func TestExtractTopLeftSquareTooSmall(t *testing.T) {
	tensor := &ggml.Tensor{
		Name:  "fc.weight",
		Kind:  uint32(ggml.TensorTypeF32),
		Shape: []uint64{128, 64},
	}
	_, err := ExtractTopLeftSquareFP16(make([]byte, 128*64*4), tensor, 256)
	if err == nil {
		t.Fatal("expected shape error")
	}
}
