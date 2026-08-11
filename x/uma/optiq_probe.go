//go:build darwin && uma

package uma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

var optiqProbeOnce sync.Once

// MaybeProbeOptiqLiveChain runs at most once when ZEROLLAMA_UMA_OPTIQ_GRAPH_PROBE
// is set and UMA_OPTIQ_DUMP_DIR points at a F0631/F0633 dump (wz/wo/x/y_host).
// Intended for the RELEASE gap after mlxrunner decode Eval (F0634/F0635).
// Never nests under RunGPU. did=false when probe is disabled or skipped.
func MaybeProbeOptiqLiveChain() (did bool, err error) {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_PROBE")))
	switch v {
	case "1", "on", "true", "yes", "require":
	default:
		return false, nil
	}
	dump := os.Getenv("UMA_OPTIQ_DUMP_DIR")
	if dump == "" {
		dump = "/tmp/uma_optiq_live_dump"
	}
	if _, err := os.Stat(filepath.Join(dump, "meta.txt")); err != nil {
		if v == "require" {
			return false, fmt.Errorf("optiq dump missing at %s (run: make -C uma_toolkit optiq-live-dump): %w", dump, err)
		}
		return false, nil
	}
	var runErr error
	var ran bool
	optiqProbeOnce.Do(func() {
		runErr = probeOptiqLiveChain(dump)
		ran = true
	})
	return ran, runErr
}

func probeOptiqLiveChain(dump string) error {
	if !Active() {
		return fmt.Errorf("uma not active")
	}
	meta, err := readOptiqMeta(filepath.Join(dump, "meta.txt"))
	if err != nil {
		return err
	}
	N, _ := strconv.Atoi(meta["N"])
	K, _ := strconv.Atoi(meta["K"])
	nw, _ := strconv.Atoi(meta["nw"])
	if N < 1 || K < 1 || nw < 1 {
		return fmt.Errorf("bad optiq meta N=%d K=%d nw=%d", N, K, nw)
	}
	x, err := os.ReadFile(filepath.Join(dump, "x.bin"))
	if err != nil {
		return err
	}
	wz, err := os.ReadFile(filepath.Join(dump, "wz.bin"))
	if err != nil {
		return err
	}
	wo, err := os.ReadFile(filepath.Join(dump, "wo.bin"))
	if err != nil {
		return err
	}
	yHostB, err := os.ReadFile(filepath.Join(dump, "y_host.bin"))
	if err != nil {
		return err
	}
	if len(x) != K*4 || len(wz) != nw || len(wo) != nw || len(yHostB) != N*4 {
		return fmt.Errorf("optiq dump size mismatch")
	}
	yRef := optiqBytesF32(yHostB)

	names := []string{"pr_x", "pr_mid", "pr_y", "pr_wz", "pr_wo"}
	for _, n := range names {
		BufFree(n)
	}
	defer func() {
		for _, n := range names {
			BufFree(n)
		}
	}()

	if err := BufAlloc("pr_x", len(x)); err != nil {
		return err
	}
	if err := BufAlloc("pr_mid", N*4); err != nil {
		return err
	}
	if err := BufAlloc("pr_y", N*4); err != nil {
		return err
	}
	if err := BufAlloc("pr_wz", len(wz)); err != nil {
		return err
	}
	if err := BufAlloc("pr_wo", len(wo)); err != nil {
		return err
	}
	zeros := make([]byte, N*4)
	for _, p := range []struct {
		n string
		b []byte
	}{
		{"pr_x", x}, {"pr_wz", wz}, {"pr_wo", wo}, {"pr_mid", zeros}, {"pr_y", zeros},
	} {
		if err := BufPut(p.n, p.b); err != nil {
			return fmt.Errorf("put %s: %w", p.n, err)
		}
	}

	nodes := strings.Join([]string{
		"GEMV_Q4_G64@GPU! x=pr_x y=pr_mid w=pr_wz N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"GEMV_Q4_G64@GPU! x=pr_mid y=pr_y w=pr_wo N=" + strconv.Itoa(N) + " K=" + strconv.Itoa(K),
		"MARK@GPU?",
	}, " ; ")
	job, err := FormatGraph(1, "chain", nodes)
	if err != nil {
		return err
	}
	resp, err := Graph("mlxrunner-optiq-probe", job, 180)
	if err != nil {
		return err
	}
	if !strings.Contains(resp, "GEMV_Q4_G64") {
		return fmt.Errorf("unexpected graph reply: %s", resp)
	}
	gotb := make([]byte, N*4)
	n, err := BufGet("pr_y", gotb)
	if err != nil {
		return err
	}
	if n != len(gotb) {
		return fmt.Errorf("buf get nbytes=%d", n)
	}
	got := optiqBytesF32(gotb)
	var maxe float32
	for i := 0; i < N; i++ {
		e := float32(math.Abs(float64(got[i] - yRef[i])))
		if e > maxe {
			maxe = e
		}
	}
	if maxe > 1e-3 {
		return fmt.Errorf("optiq probe maxerr=%g", maxe)
	}
	return nil
}

func readOptiqMeta(path string) (map[string]string, error) {
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

func optiqBytesF32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
