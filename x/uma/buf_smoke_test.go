//go:build darwin && uma

package uma_test

import (
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/uma"
)

// Lab: BUF_* + RESIDUAL_ADD GRAPH via x/uma (F0627).
func TestBufResidualGraph(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_BUF_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_BUF_SMOKE=1")
	}
	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-buf")
	uma.Release()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()

	const D = 64
	y := make([]float32, D)
	x := make([]float32, D)
	ref := make([]float32, D)
	for i := 0; i < D; i++ {
		y[i] = 0.1 * float32(i)
		x[i] = -0.05 + 0.02*float32(i)
		ref[i] = y[i] + x[i]
	}
	yb := f32Bytes(y)
	xb := f32Bytes(x)

	uma.BufFree("gy")
	uma.BufFree("gx")
	if err := uma.BufAlloc("gy", len(yb)); err != nil {
		t.Fatalf("alloc gy: %v", err)
	}
	if err := uma.BufAlloc("gx", len(xb)); err != nil {
		t.Fatalf("alloc gx: %v", err)
	}
	defer uma.BufFree("gy")
	defer uma.BufFree("gx")
	if err := uma.BufPut("gy", yb); err != nil {
		t.Fatalf("put gy: %v", err)
	}
	if err := uma.BufPut("gx", xb); err != nil {
		t.Fatalf("put gx: %v", err)
	}

	job, err := uma.FormatGraph(1, "chain",
		"RESIDUAL_ADD@GPU! y=gy x=gx D=64 ; MARK@GPU?")
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	resp, err := uma.Graph("xuma-buf-add", job, 30)
	if err != nil {
		t.Fatalf("graph: %v", err)
	}
	if !strings.Contains(resp, "RESIDUAL_ADD") {
		t.Fatalf("reply: %s", resp)
	}

	gotb := make([]byte, len(yb))
	n, err := uma.BufGet("gy", gotb)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if n != len(yb) {
		t.Fatalf("got nbytes=%d want %d", n, len(yb))
	}
	got := bytesF32(gotb)
	var maxe float32
	for i := 0; i < D; i++ {
		e := float32(math.Abs(float64(got[i] - ref[i])))
		if e > maxe {
			maxe = e
		}
	}
	if maxe > 1e-5 {
		t.Fatalf("maxerr=%g", maxe)
	}
	t.Logf("residual maxerr=%g", maxe)
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func bytesF32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
