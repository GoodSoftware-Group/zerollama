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

// TestOptiqLiveChainGraph — F0633: x/uma Buf* + Wz→Wo GEMV chain on live OptiQ dumps.
// Requires UMA_OPTIQ_DUMP_DIR from test_uma_optiq_live_chain_smoke (wz/wo/x/y_host).
func TestOptiqLiveChainGraph(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_OPTIQ_CHAIN_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_OPTIQ_CHAIN_SMOKE=1")
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
		t.Fatalf("size mismatch x=%d wz=%d wo=%d y=%d", len(x), len(wz), len(wo), len(yHostB))
	}
	yRef := bytesF32(yHostB)

	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-optiq")
	uma.Release()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()

	names := []string{"go_x", "go_mid", "go_y", "go_wz", "go_wo"}
	for _, n := range names {
		uma.BufFree(n)
	}
	defer func() {
		for _, n := range names {
			uma.BufFree(n)
		}
	}()

	if err := uma.BufAlloc("go_x", len(x)); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("go_mid", N*4); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("go_y", N*4); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("go_wz", len(wz)); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufAlloc("go_wo", len(wo)); err != nil {
		t.Fatal(err)
	}
	zeros := make([]byte, N*4)
	if err := uma.BufPut("go_x", x); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufPut("go_wz", wz); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufPut("go_wo", wo); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufPut("go_mid", zeros); err != nil {
		t.Fatal(err)
	}
	if err := uma.BufPut("go_y", zeros); err != nil {
		t.Fatal(err)
	}

	nodes := strings.Join([]string{
		"GEMV_Q4_G64@GPU! x=go_x y=go_mid w=go_wz N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"GEMV_Q4_G64@GPU! x=go_mid y=go_y w=go_wo N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"MARK@GPU?",
	}, " ; ")
	job, err := uma.FormatGraph(1, "chain", nodes)
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	resp, err := uma.Graph("xuma-optiq-chain", job, 180)
	if err != nil {
		t.Fatalf("graph: %v", err)
	}
	if !strings.Contains(resp, "GEMV_Q4_G64") {
		t.Fatalf("reply: %s", resp)
	}

	gotb := make([]byte, N*4)
	n, err := uma.BufGet("go_y", gotb)
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
	t.Logf("go live chain maxerr=%g N=%d", maxe, N)
}

func readDumpMeta(path string) (map[string]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m := make(map[string]string)
	for _, line := range strings.Split(string(b), "\n") {
		if i := strings.IndexByte(line, '='); i > 0 {
			m[line[:i]] = line[i+1:]
		}
	}
	return m, nil
}
