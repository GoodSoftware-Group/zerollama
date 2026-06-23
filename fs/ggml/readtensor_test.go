package ggml

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func TestReadTensorBytes(t *testing.T) {
	t.Parallel()

	rows, cols := uint64(4), uint64(6)
	raw := make([]byte, rows*cols*4)
	for r := range rows {
		for c := range cols {
			v := float32(float64(r*10 + c))
			binary.LittleEndian.PutUint32(raw[(r*cols+c)*4:], math.Float32bits(v))
		}
	}

	w, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	ts := []*Tensor{{
		Name:     "blk.0.ffn_gate.weight",
		Kind:     uint32(TensorTypeF32),
		Shape:    []uint64{rows, cols},
		WriterTo: bytes.NewReader(raw),
	}}
	if err := WriteGGUF(w, KV{
		"general.architecture": "test",
		"general.alignment":    uint32(32),
	}, ts); err != nil {
		t.Fatal(err)
	}
	w.Close()

	got, tensor, err := ReadTensorBytes(w.Name(), "blk.0.ffn_gate.weight")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, raw) {
		t.Fatalf("bytes mismatch len=%d got=%d", len(raw), len(got))
	}
	if tensor == nil || tensor.Name != "blk.0.ffn_gate.weight" {
		t.Fatalf("tensor: %+v", tensor)
	}
}

func TestReadTensorBytesMissing(t *testing.T) {
	t.Parallel()

	w, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	ts := []*Tensor{{
		Name:     "token_embd.weight",
		Kind:     uint32(TensorTypeF32),
		Shape:    []uint64{2, 3},
		WriterTo: bytes.NewReader(make([]byte, 24)),
	}}
	if err := WriteGGUF(w, KV{
		"general.architecture": "test",
		"general.alignment":    uint32(32),
	}, ts); err != nil {
		t.Fatal(err)
	}
	w.Close()

	_, _, err = ReadTensorBytes(w.Name(), "missing.weight")
	if err == nil {
		t.Fatal("expected error")
	}
}
