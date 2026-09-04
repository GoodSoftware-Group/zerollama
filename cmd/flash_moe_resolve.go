package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/x/freetokenlab"
)

type flashMoEResolveRow struct {
	discover.FlashMoEInventoryEntry
	FreetokenSlotBank int     `json:"freetoken_slot_bank"`
	RamCapSlots       int     `json:"ram_cap_slots"`
	RecommendSlotBank int     `json:"recommend_slot_bank"`
	StickyMissRate    float64 `json:"sticky_miss_rate,omitempty"`
	SlotBankBytes     int64   `json:"slot_bank_bytes,omitempty"`
	HostRAMGiB        float64 `json:"host_ram_gib,omitempty"`
}

func hostRAMGiB() float64 {
	m, err := discover.GetCPUMem()
	if err != nil || m.TotalMemory == 0 {
		return 0
	}
	return float64(m.TotalMemory) / float64(1<<30)
}

func flashMoEResolveRowFrom(e discover.FlashMoEInventoryEntry, ramGiB float64) flashMoEResolveRow {
	a := freetokenlab.AdviseSlotBankK(int(e.ExpertCount), int(e.ExpertUsedCount), ramGiB, e.ExpertWeightBytes)
	return flashMoEResolveRow{
		FlashMoEInventoryEntry: e,
		FreetokenSlotBank:      a.Routing,
		RamCapSlots:            a.RamCap,
		RecommendSlotBank:      a.Recommend,
		StickyMissRate:         a.MissRate,
		SlotBankBytes:          a.BankBytes,
		HostRAMGiB:             ramGiB,
	}
}

func flashMoESlotBankEnvLine(row flashMoEResolveRow) string {
	line := fmt.Sprintf("export ZEROLLAMA_FLASH_MOE_SLOT_BANK=%d", row.RecommendSlotBank)
	bits := []string{"lab; not auto-passed"}
	if row.SlotBankBytes > 0 {
		bits = append([]string{fmt.Sprintf("~%s packed experts", format.HumanBytes2(uint64(row.SlotBankBytes)))}, bits...)
	}
	if row.StickyMissRate > 0 && row.StickyMissRate < 1 {
		bits = append(bits, fmt.Sprintf("sticky miss≈%.3f", row.StickyMissRate))
	}
	return line + "  # " + strings.Join(bits, "; ")
}

type flashMoEEnvGroup struct {
	row  flashMoEResolveRow
	tags []string
}

// groupFlashMoERowsByGGUF collapses alias tags that share a blob (one export).
func groupFlashMoERowsByGGUF(rows []flashMoEResolveRow) []flashMoEEnvGroup {
	order := make([]string, 0)
	by := map[string]*flashMoEEnvGroup{}
	for _, r := range rows {
		key := r.GGUFPath
		if key == "" {
			key = r.Tag
		}
		g, ok := by[key]
		if !ok {
			cp := r
			g = &flashMoEEnvGroup{row: cp}
			by[key] = g
			order = append(order, key)
		}
		tag := r.Tag
		if tag == "" {
			tag = r.Name
		}
		seen := false
		for _, t := range g.tags {
			if t == tag {
				seen = true
				break
			}
		}
		if !seen {
			g.tags = append(g.tags, tag)
		}
		if r.SidecarReady && !g.row.SidecarReady {
			g.row = r
		}
	}
	out := make([]flashMoEEnvGroup, 0, len(order))
	for _, k := range order {
		out = append(out, *by[k])
	}
	return out
}

func printFlashMoEEnvGroups(groups []flashMoEEnvGroup) {
	for _, g := range groups {
		fmt.Printf("# %s\n", strings.Join(g.tags, ", "))
		if !g.row.SidecarReady {
			fmt.Println("# sidecar=missing — do not export; do not load this GGUF on production Metal serve")
			fmt.Printf("# %s\n", flashMoESlotBankEnvLine(g.row))
			continue
		}
		fmt.Println(flashMoESlotBankEnvLine(g.row))
	}
}

func NewFlashMoEResolveCommand() *cobra.Command {
	var (
		asJSON    bool
		preferred string
		listAll   bool
		printEnv  bool
	)
	cmd := &cobra.Command{
		Use:    "flash-moe-resolve",
		Hidden: true,
		Short:  "Resolve local MoE GGUF + sidecar paths from zerollama model store",
		Long: `Scan ~/.ollama/models for pulled MoE tags and infer Flash-MoE sidecar paths.

Why: smoke scripts and operators should not hand-copy blob paths when zerollama
already knows model names from pull/create.`,
		RunE: func(_ *cobra.Command, _ []string) error {
			if listAll {
				entries, err := discover.ListFlashMoEInventory()
				if err != nil {
					return err
				}
				ram := hostRAMGiB()
				rows := make([]flashMoEResolveRow, 0, len(entries))
				for _, e := range entries {
					rows = append(rows, flashMoEResolveRowFrom(e, ram))
				}
				if asJSON {
					enc := json.NewEncoder(os.Stdout)
					enc.SetIndent("", "  ")
					return enc.Encode(rows)
				}
				if len(rows) == 0 {
					fmt.Println("no local MoE models found")
					return nil
				}
				if printEnv {
					printFlashMoEEnvGroups(groupFlashMoERowsByGGUF(rows))
					return nil
				}
				for _, e := range rows {
					status := "sidecar=missing"
					if e.SidecarReady {
						status = "sidecar=ready"
					}
					extra := ""
					if e.SlotBankBytes > 0 {
						extra = fmt.Sprintf("  bank~%s", format.HumanBytes2(uint64(e.SlotBankBytes)))
					}
					if e.StickyMissRate > 0 && e.StickyMissRate < 1 {
						extra += fmt.Sprintf("  miss≈%.3f", e.StickyMissRate)
					}
					fmt.Printf("%s  %s  experts=%d k=%d  freetoken_slot_bank=%d  ram_cap=%d  recommend=%d%s  %s\n",
						e.Tag, e.GGUFPath, e.ExpertCount, e.ExpertUsedCount, e.FreetokenSlotBank, e.RamCapSlots, e.RecommendSlotBank, extra, status)
				}
				return nil
			}

			if preferred == "" {
				preferred = os.Getenv("FLASH_MOE_MODEL")
			}
			entry, err := discover.ResolveFlashMoEModel(preferred)
			if err != nil {
				return fmt.Errorf("no local MoE model found — pull a MoE tag or set FLASH_MOE_MODEL: %w", err)
			}
			row := flashMoEResolveRowFrom(entry, hostRAMGiB())
			if asJSON {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(row)
			}
			if printEnv && !asJSON {
				printFlashMoEEnvGroups(groupFlashMoERowsByGGUF([]flashMoEResolveRow{row}))
				return nil
			}
			fmt.Printf("tag=%s\ngguf=%s\nsidecar=%s\nsidecar_ready=%v\nexperts=%d\nexpert_used_count=%d\nfreetoken_slot_bank=%d\nram_cap_slots=%d\nrecommend_slot_bank=%d\nsticky_miss_rate=%.3f\nslot_bank_bytes=%d\nhost_ram_gib=%.1f\n%s\n",
				row.Tag, row.GGUFPath, row.Sidecar, row.SidecarReady, row.ExpertCount, row.ExpertUsedCount, row.FreetokenSlotBank,
				row.RamCapSlots, row.RecommendSlotBank, row.StickyMissRate, row.SlotBankBytes, row.HostRAMGiB, flashMoESlotBankEnvLine(row))
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "JSON output")
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer this tag or model name")
	cmd.Flags().BoolVar(&listAll, "list", false, "List all local MoE models")
	cmd.Flags().BoolVar(&printEnv, "print-env", false, "Print ZEROLLAMA_FLASH_MOE_SLOT_BANK export line only (not applied)")
	return cmd
}
