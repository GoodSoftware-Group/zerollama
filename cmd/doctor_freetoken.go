package cmd

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/x/freetokenlab"
)

func doctorCheckFreeToken() doctorCheck {
	const name = "freetoken MoE policy (lab)"
	profile := "mac-uma"
	if runtime.GOOS != "darwin" {
		profile = "5080-est"
	}
	nExp, k := 256, 6
	if inv, err := doctorMoEInventory(); err == nil && len(inv) > 0 {
		if inv[0].ExpertCount >= 2 {
			nExp = int(inv[0].ExpertCount)
		}
		if inv[0].ExpertUsedCount > 0 {
			k = int(inv[0].ExpertUsedCount)
		}
	}
	a := freetokenlab.AdviseProfileFor(profile, nExp, k)
	status := "ok"
	fix := ""
	detail := a.DoctorLine()
	if envconfig.FlashMoEPrefetchTemporal() && !a.PrefetchTemporal {
		status = "warn"
		fix = "unset ZEROLLAMA_FLASH_MOE_PREFETCH — lab: LRU already holds sticky experts"
		detail += "; PREFETCH env is on"
	}
	if len(a.Notes) > 0 {
		detail += "; " + strings.Join(a.Notes[:1], "")
	}
	detail += doctorFreeTokenInventoryHint()
	detail += "; " + server.ChatCompressLabSummary()
	return doctorCheck{Name: name, Status: status, Detail: detail, FixHint: fix}
}

func doctorFreeTokenInventoryHint() string {
	inv, err := doctorMoEInventory()
	if err != nil || len(inv) == 0 {
		return ""
	}
	ram := 0.0
	if m, err := discover.GetCPUMem(); err == nil && m.TotalMemory > 0 {
		ram = float64(m.TotalMemory) / float64(1<<30)
	}
	e := inv[0]
	b := freetokenlab.AdviseSlotBankK(int(e.ExpertCount), int(e.ExpertUsedCount), ram, e.ExpertWeightBytes)
	hint := fmt.Sprintf("; local %s k=%d recommend=%d sticky-miss≈%.3f", e.Tag, e.ExpertUsedCount, b.Recommend, b.MissRate)
	if b.BankBytes > 0 {
		hint += fmt.Sprintf(" bank~%s", format.HumanBytes2(uint64(b.BankBytes)))
	}
	if !e.SidecarReady {
		hint += " sidecar=missing (do not export on Metal serve)"
	}
	return hint
}
