package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/freetokenlab"
)

func doctorCheckFlashMoE(_ string) doctorCheck {
	// Why report binary when disabled: operators should see "binary ready" before setting env.
	name := "flash-moe llama-server (experimental)"
	if runtime.GOOS != "darwin" {
		return doctorCheck{Name: name, Status: "warn", Detail: "Flash-MoE build is darwin-only today"}
	}

	repo := envconfig.FlashMoERepo()
	repoOK := pathExists(filepath.Join(repo, "CMakeLists.txt"))

	bin, binErr := llm.FindFlashMoELlamaServer()
	binOK := binErr == nil

	enabled := envconfig.FlashMoEEnabled()
	sidecar := envconfig.FlashMoESidecar()

	if !enabled && sidecar == "" {
		detail := "not enabled — set ZEROLLAMA_FLASH_MOE=1 + ZEROLLAMA_FLASH_MOE_SIDECAR for RAM-busting MoE"
		switch {
		case binOK:
			detail = fmt.Sprintf("not enabled (binary ready @ %s)", bin)
		case repoOK:
			detail += "; run ./scripts/build_flash_moe_llama_server.sh"
		default:
			detail += fmt.Sprintf("; repo missing at %s", repo)
		}
		detail += flashMoEInventoryHint()
		fixHint := "export ZEROLLAMA_FLASH_MOE=1 ZEROLLAMA_FLASH_MOE_SIDECAR=/path/to/sidecar"
		if !repoOK {
			fixHint = "git clone --branch Server-Flash-Moe --depth 1 https://github.com/Anemll/anemll-flash-llama.cpp.git " + repo
		} else if !binOK {
			fixHint = "./scripts/build_flash_moe_llama_server.sh"
		}
		return doctorCheck{Name: name, Status: "warn", Detail: detail, FixHint: fixHint}
	}

	if !repoOK {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("anemll-flash-llama.cpp missing at %s", repo),
			FixHint: "git clone --branch Server-Flash-Moe --depth 1 https://github.com/Anemll/anemll-flash-llama.cpp.git " + repo,
		}
	}

	if !binOK {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "flash-moe llama-server not built",
			FixHint: "./scripts/build_flash_moe_llama_server.sh",
		}
	}

	if sidecar == "" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("ZEROLLAMA_FLASH_MOE=1 but sidecar unset; binary ok @ %s", bin),
			FixHint: "export ZEROLLAMA_FLASH_MOE_SIDECAR=/path/to/extracted/sidecar",
		}
	}
	if st, err := os.Stat(sidecar); err != nil || !st.IsDir() {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("sidecar missing: %s", sidecar),
			FixHint: "see docs/flash-moe.md — flashmoe_sidecar.py extract",
		}
	}

	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("binary=%s sidecar=%s mode=%s%s", bin, sidecar, envconfig.FlashMoEMode(), flashMoEInventoryHint()),
	}
}

// flashMoEInventoryHint is lab-only slot-bank advice from GGUF expert_count.
// It is never injected as --moe-slot-bank (0 still omits the flag).
func flashMoEInventoryHint() string {
	inv, err := doctorMoEInventory()
	if err != nil || len(inv) == 0 {
		return ""
	}
	ram := 0.0
	if m, err := discover.GetCPUMem(); err == nil && m.TotalMemory > 0 {
		ram = float64(m.TotalMemory) / float64(1<<30)
	}
	a := freetokenlab.AdviseSlotBankK(int(inv[0].ExpertCount), int(inv[0].ExpertUsedCount), ram, inv[0].ExpertWeightBytes)
	hint := fmt.Sprintf("; found %d local MoE tag(s) — ./zerollama flash-moe-resolve --list; %s recommend_slot_bank=%d (routing=%d ram_cap=%d; not auto-passed)",
		len(inv), inv[0].Tag, a.Recommend, a.Routing, a.RamCap)
	if a.BankBytes > 0 {
		hint += fmt.Sprintf(" bank~%s", format.HumanBytes2(uint64(a.BankBytes)))
	}
	if a.MissRate > 0 && a.MissRate < 1 {
		hint += fmt.Sprintf(" miss≈%.3f", a.MissRate)
	}
	return hint
}

var (
	doctorMoEInvOnce sync.Once
	doctorMoEInv     []discover.FlashMoEInventoryEntry
	doctorMoEInvErr  error
)

// doctorMoEInventory is one GGUF-header walk per doctor process (Flash-MoE + FreeToken).
func doctorMoEInventory() ([]discover.FlashMoEInventoryEntry, error) {
	doctorMoEInvOnce.Do(func() {
		doctorMoEInv, doctorMoEInvErr = discover.ListFlashMoEInventory()
	})
	return doctorMoEInv, doctorMoEInvErr
}

func pathExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}
