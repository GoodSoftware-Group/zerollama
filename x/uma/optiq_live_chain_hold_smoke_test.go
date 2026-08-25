//go:build darwin && uma

package uma_test

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/uma"
)

// TestOptiqLiveChainHoldGap — F0634: grain=op RunGPU (Eval stand-in) then live
// Wz→Wo GRAPH in the RELEASE gap. Graph must NOT run under RunGPU (deadlocks on
// self-HOLD).
func TestOptiqLiveChainHoldGap(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_OPTIQ_HOLD_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_OPTIQ_HOLD_SMOKE=1")
	}
	dump := os.Getenv("UMA_OPTIQ_DUMP_DIR")
	if dump == "" {
		t.Fatal("UMA_OPTIQ_DUMP_DIR required")
	}
	meta, err := readDumpMeta(filepath.Join(dump, "meta.txt"))
	if err != nil {
		t.Fatalf("meta: %v", err)
	}
	N, _ := strconv.Atoi(meta["N"])
	K, _ := strconv.Atoi(meta["K"])
	nw, _ := strconv.Atoi(meta["nw"])
	if N < 1 || K < 1 || nw < 1 {
		t.Fatalf("bad meta N=%d K=%d nw=%d", N, K, nw)
	}

	x, err := os.ReadFile(filepath.Join(dump, "x.bin"))
	if err != nil {
		t.Fatal(err)
	}
	wz, err := os.ReadFile(filepath.Join(dump, "wz.bin"))
	if err != nil {
		t.Fatal(err)
	}
	wo, err := os.ReadFile(filepath.Join(dump, "wo.bin"))
	if err != nil {
		t.Fatal(err)
	}
	yHostB, err := os.ReadFile(filepath.Join(dump, "y_host.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(x) != K*4 || len(wz) != nw || len(wo) != nw || len(yHostB) != N*4 {
		t.Fatalf("size mismatch")
	}
	yRef := bytesF32(yHostB)

	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("ZEROLLAMA_UMA_GRAIN", "op")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-optiq-hold")
	uma.Release()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()
	if g := uma.Grain(); g != "op" {
		t.Fatalf("grain want op got %q", g)
	}

	names := []string{"gh_x", "gh_mid", "gh_y", "gh_wz", "gh_wo"}
	for _, n := range names {
		uma.BufFree(n)
	}
	defer func() {
		for _, n := range names {
			uma.BufFree(n)
		}
	}()

	if err := uma.BufAlloc("gh_x", len(x)); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("gh_mid", N*4); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("gh_y", N*4); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("gh_wz", len(wz)); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("gh_wo", len(wo)); err != nil {
		t.Fatal(err)
	}
	zeros := make([]byte, N*4)
	for _, p := range []struct {
		n string
		b []byte
	}{
		{"gh_x", x}, {"gh_wz", wz}, {"gh_wo", wo}, {"gh_mid", zeros}, {"gh_y", zeros},
	} {
		if err := uma.BufPut(p.n, p.b); err != nil {
			t.Fatalf("put %s: %v", p.n, err)
		}
	}

	// Eval stand-in under one-shot HOLD (grain=op) — RELEASE before GRAPH.
	evalRan := false
	if err := uma.RunGPU(func() { evalRan = true }); err != nil {
		t.Fatalf("RunGPU eval: %v", err)
	}
	if !evalRan {
		t.Fatal("eval stand-in did not run")
	}

	nodes := strings.Join([]string{
		"GEMV_Q4_G64@GPU! x=gh_x y=gh_mid w=gh_wz N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"GEMV_Q4_G64@GPU! x=gh_mid y=gh_y w=gh_wo N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"MARK@GPU?",
	}, " ; ")
	job, err := uma.FormatGraph(1, "chain", nodes)
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	resp, err := uma.Graph("xuma-optiq-hold-gap", job, 180)
	if err != nil {
		t.Fatalf("graph in gap: %v", err)
	}
	if !strings.Contains(resp, "GEMV_Q4_G64") {
		t.Fatalf("reply: %s", resp)
	}

	gotb := make([]byte, N*4)
	n, err := uma.BufGet("gh_y", gotb)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if n != len(gotb) {
		t.Fatalf("nbytes=%d", n)
	}
	got := bytesF32(gotb)
	var maxe float32
	for i := 0; i < N; i++ {
		e := float32(math.Abs(float64(got[i] - yRef[i])))
		if e > maxe {
			maxe = e
		}
	}
	if maxe > 1e-3 {
		t.Fatalf("maxerr=%g", maxe)
	}
	t.Logf("go live chain in HOLD gap maxerr=%g N=%d grain=op", maxe, N)
}
