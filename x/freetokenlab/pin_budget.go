package freetokenlab

import (
	"fmt"
	"math"
)

// DefaultWSLPinFrac matches FreeToken _pin_budget_bytes on WSL (WDDM CUDA
// pin cap ≈ half of RAM shared across processes → budget 40%).
const DefaultWSLPinFrac = 0.40

// PinBudgetOpts mirrors FreeToken engine._pin_budget_bytes inputs.
type PinBudgetOpts struct {
	HostRAMBytes  int64   // SC_PHYS_PAGES × PAGE_SIZE (or discover host RAM)
	ReservedBytes int64   // already pinned outside expert banks (e.g. PLE)
	PinBudgetGiB  float64 // >0 forces a cap (ZEROLLAMA_FLASH_MOE_PIN_BUDGET_GB / FREETOKEN_PIN_BUDGET_GB)
	WSL           bool    // microsoft in uname.release — only auto-capped platform today
}

// PinBudgetBytes is how many bytes this process can still cudaHostRegister, or
// (0, false) when the platform does not cap pinning (plain Linux / Darwin).
// FreeToken returns None for uncapped; we use budget=0 + Capped=false.
func PinBudgetBytes(o PinBudgetOpts) (budget int64, capped bool) {
	if o.PinBudgetGiB > 0 {
		cap := int64(o.PinBudgetGiB * float64(1<<30))
		return max64(0, cap-o.ReservedBytes), true
	}
	if !o.WSL {
		return 0, false
	}
	if o.HostRAMBytes < 1 {
		return 0, false
	}
	cap := int64(float64(o.HostRAMBytes) * DefaultWSLPinFrac)
	return max64(0, cap-o.ReservedBytes), true
}

// PinAdvice is FreeToken split-residency guidance when expert banks exceed the
// pin budget (--moe-cpu-layers auto). anemll has no matching flag yet.
type PinAdvice struct {
	BudgetBytes          int64
	Capped               bool
	BankBytes            int64 // full expert banks (all experts), not slot-bank
	OverBudget           bool
	SuggestedCPULayers   int
	SuggestedCPULayerIDs []int
	Notes                []string
}

// AdvisePin rematches FreeToken --moe-cpu-layers auto: none while banks fit
// the pin budget; otherwise lock just enough head+tail MoE layers (U-shaped
// decode miss rates).
func AdvisePin(o PinBudgetOpts, fullBankBytes int64, numMoELayers int) PinAdvice {
	budget, capped := PinBudgetBytes(o)
	a := PinAdvice{
		BudgetBytes: budget,
		Capped:      capped,
		BankBytes:   fullBankBytes,
	}
	if !capped {
		// Silent on plain Linux / Darwin — avoid noisy Mac UMA doctor lines.
		// Force a lab cap with ZEROLLAMA_FLASH_MOE_PIN_BUDGET_GB (or WSL 40%).
		return a
	}
	if fullBankBytes < 1 {
		a.Notes = append(a.Notes, fmt.Sprintf(
			"pin budget ~%.2f GiB (capped); bank bytes unknown — pull MoE GGUF dims or measure *_exps",
			float64(budget)/float64(1<<30)))
		return a
	}
	if fullBankBytes <= budget {
		a.Notes = append(a.Notes, fmt.Sprintf(
			"expert banks ~%.2f GiB ≤ pin budget ~%.2f GiB — keep banks PINNED; --moe-cpu-layers auto → none",
			float64(fullBankBytes)/float64(1<<30), float64(budget)/float64(1<<30)))
		return a
	}
	a.OverBudget = true
	a.SuggestedCPULayerIDs = AutoCPULayerIDs(numMoELayers, fullBankBytes, budget)
	a.SuggestedCPULayers = len(a.SuggestedCPULayerIDs)
	a.Notes = append(a.Notes, fmt.Sprintf(
		"expert banks ~%.2f GiB > pin budget ~%.2f GiB — FreeToken --moe-cpu-layers auto would lock %d head+tail MoE layers %v; anemll has no --moe-cpu-layers — prefer smaller --moe-slot-bank / fewer GPU-resident experts",
		float64(fullBankBytes)/float64(1<<30), float64(budget)/float64(1<<30),
		a.SuggestedCPULayers, a.SuggestedCPULayerIDs))
	return a
}

// AutoCPULayerCount is how many MoE layers to lock for CPU decode when banks
// exceed the pin budget (FreeToken _auto_cpu_layers).
func AutoCPULayerCount(numMoELayers int, bankBytes, budgetBytes int64) int {
	if numMoELayers < 1 || bankBytes < 1 || budgetBytes < 1 || bankBytes <= budgetBytes {
		return 0
	}
	n := int(math.Ceil(float64(numMoELayers) * (1 - float64(budgetBytes)/float64(bankBytes))))
	if n > numMoELayers {
		n = numMoELayers
	}
	if n < 1 {
		n = 1
	}
	return n
}

// AutoCPULayerIDs locks head+tail MoE layer ids (FreeToken _auto_cpu_layers).
func AutoCPULayerIDs(numMoELayers int, bankBytes, budgetBytes int64) []int {
	n := AutoCPULayerCount(numMoELayers, bankBytes, budgetBytes)
	if n < 1 {
		return nil
	}
	head := (n + 1) / 2
	ids := make([]int, 0, n)
	for i := 0; i < head; i++ {
		ids = append(ids, i)
	}
	tail := n - head
	start := numMoELayers - tail
	for i := start; i < numMoELayers; i++ {
		ids = append(ids, i)
	}
	return ids
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
