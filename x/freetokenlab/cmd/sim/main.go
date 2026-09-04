// Command sim prints FreeToken policy numbers for zerollama hardware profiles.
//
//	go run ./x/freetokenlab/cmd/sim
//	go run ./x/freetokenlab/cmd/sim -trace x/freetokenlab/testdata/anemll_mini.jsonl
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/ollama/ollama/x/freetokenlab"
)

func main() {
	tracePath := flag.String("trace", "", "anemll --moe-trace JSONL (n_tokens>1 = prefill)")
	nExpFlag := flag.Int("experts", 256, "expert pool size for Zipf sweep and slot-bank advice")
	kFlag := flag.Int("k", 6, "routed experts per token (GGUF expert_used_count)")
	ramFlag := flag.Float64("ram", 0, "if >0, print one AdviseSlotBankK row for this RAM GiB")
	flag.Parse()

	nExp, k := *nExpFlag, *kFlag
	if nExp < 2 {
		nExp = 256
	}
	if k < 1 {
		k = 6
	}

	const expertBytes int64 = 80 << 20
	const misses = 4
	fmt.Println("FreeToken q* miss split (m=4 unique misses, 80MiB expert)")
	fmt.Printf("%-14s %6s %6s %10s %10s %8s\n", "profile", "fill", "cpu", "layer_s", "allfill_s", "speedup")
	for _, name := range []string{"5090-server", "4090", "4060-laptop", "5090-desktop", "5080-est", "mac-uma"} {
		bw := freetokenlab.Profiles()[name]
		s := freetokenlab.SplitMisses(misses, expertBytes, bw)
		fmt.Printf("%-14s %6d %6d %10.4f %10.4f %8.2fx\n",
			name, s.Fill, s.CPU, s.LayerSec, s.AllFillSec, s.SpeedupVsAllFill())
	}

	o := freetokenlab.OverlapPrefill(40, 0.02, 0.05)
	fmt.Printf("\nPrefill overlap 40 layers compute=20ms xfer=50ms: serial=%.2fs pipe=%.2fs gain=%.2fx\n",
		o.SerialSec, o.PipelinedSec, o.ThroughputGain)

	if *tracePath != "" {
		tr, err := freetokenlab.LoadAnemllFile(*tracePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "trace: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("\nDecode expert miss rates — anemll %s (%d records, %d experts, prefill_steps=%d decode_steps=%d)\n",
			*tracePath, tr.Records, tr.NExperts, len(tr.Prefill), len(tr.Decode))
		printSweep(tr.NExperts, tr.Prefill, tr.Decode)
	} else {
		const layers = 8
		pre := freetokenlab.DensePrefill(layers, nExp, 64, k)
		iid := freetokenlab.ZipfDecode(layers, 200, nExp, k, 1.15, 1)
		loc := freetokenlab.ZipfStickyDecode(layers, 200, nExp, k, 1.15, 3, 4, 1)
		fmt.Printf("\nMiss rates — i.i.d. Zipf (experts=%d k=%d; static can pin the head)\n", nExp, k)
		printSweep(nExp, pre, iid)
		fmt.Println("\nMiss rates — Zipf + 75% stay (paper-style routing locality)")
		printSweep(nExp, pre, loc)
	}
	fmt.Println("(11% ≈ DSV4-Flash cache fraction in the paper; 37% ≈ Qwen3.6 on 5090)")

	ed := freetokenlab.SuffixStripEdit(24000, 400)
	fmt.Printf("\nSuffix-strip (FreeToken/OpenClaw): sparse-ckpt@4k=%d tokens  semantic-anchor=%d tokens\n",
		freetokenlab.PrefillTokensWithoutAnchor(ed, 4096),
		freetokenlab.PrefillTokensWithSemanticAnchor(ed))
	reuse, recompute := freetokenlab.KeepTailCompressReplay(80, 120, 2000)
	fmt.Printf("chat_compress summary: radix reuse=%d  re-prefill summary+tail=%d\n", reuse, recompute)
	fmt.Println("chat_compress placeholder: longest exact message prefix stays in KV (see compression.mode)")

	fmt.Println("\nOperator advice (do not load 22GiB MoE on this Mac while :11434 is up)")
	for _, name := range []string{"mac-uma", "5080-est"} {
		a := freetokenlab.AdviseProfileFor(name, nExp, k)
		fmt.Println(" ", a.DoctorLine())
	}
	fmt.Printf("\n--moe-slot-bank advice (%d experts k=%d; not auto-passed to llama-server)\n", nExp, k)
	packed := int64(80) << 20 * 8 * int64(nExp)
	rams := []float64{8, 16, 36, 128}
	if *ramFlag > 0 {
		rams = []float64{*ramFlag}
	}
	for _, ram := range rams {
		b := freetokenlab.AdviseSlotBankK(nExp, k, ram, packed)
		fmt.Printf("  ram=%.0fGiB  routing=%d  ram_cap=%d  recommend=%d  bank~%.2fGiB  miss≈%.3f\n",
			ram, b.Routing, b.RamCap, b.Recommend, float64(b.BankBytes)/(1<<30), b.MissRate)
	}
}

func printSweep(nExp int, pre, dec []freetokenlab.TraceStep) {
	fractions := []float64{0.11, 0.37, 0.50}
	policies := []freetokenlab.CachePolicy{
		freetokenlab.PolicyStatic,
		freetokenlab.PolicyPrefillHot,
		freetokenlab.PolicyLRU,
		freetokenlab.PolicyLRUPrefetch,
	}
	fmt.Printf("%-14s", "policy")
	for _, f := range fractions {
		fmt.Printf("  %4.0f%%", f*100)
	}
	fmt.Println()
	for _, p := range policies {
		fmt.Printf("%-14s", p.String())
		for _, r := range freetokenlab.SweepMissRates(p, nExp, fractions, pre, dec) {
			fmt.Printf("  %.3f", r.MissRate)
		}
		fmt.Println()
	}
}
