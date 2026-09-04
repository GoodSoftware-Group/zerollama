package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/x/freetokenlab"
)

type freetokenReport struct {
	Profile    string                      `json:"profile"`
	DoctorLine string                      `json:"doctor_line"`
	Notes      []string                    `json:"notes,omitempty"`
	PrefetchOn bool                        `json:"prefetch_env_on"`
	Inventory  []flashMoEResolveRow        `json:"inventory,omitempty"`
	Blobs      []freetokenBlobAdvice       `json:"blobs,omitempty"`
	Chat       []server.ChatCompressLabRow `json:"chat,omitempty"`
}

type freetokenBlobAdvice struct {
	Tags         []string `json:"tags"`
	SidecarReady bool     `json:"sidecar_ready"`
	EnvLine      string   `json:"env_line"`
	Apply        bool     `json:"apply"` // always false today; export is operator-only
}

func NewFreetokenCommand() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:    "freetoken",
		Hidden: true,
		Short:  "Print FreeToken / Flash-MoE lab advice from local MoE GGUF headers",
		Long: `Header-only lab report (expert_count, expert_used_count, *_exps sizes).

Does not load GGUF weights, does not bind :11434/:8081, and does not export
ZEROLLAMA_FLASH_MOE_SLOT_BANK. Sidecar-missing blobs print a commented export.`,
		RunE: func(_ *cobra.Command, _ []string) error {
			rep, err := buildFreetokenReport()
			if err != nil {
				return err
			}
			if asJSON {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(rep)
			}
			fmt.Println(rep.DoctorLine)
			if rep.PrefetchOn {
				fmt.Println("ZEROLLAMA_FLASH_MOE_PREFETCH is on — lab: unset on Mac UMA sticky routing")
			}
			if len(rep.Notes) > 0 {
				fmt.Println(rep.Notes[0])
			}
			if len(rep.Inventory) == 0 {
				fmt.Println("no local MoE tags — paper defaults 256/k=6; pull a MoE GGUF to size the slot-bank")
			} else {
				printFlashMoEEnvGroups(groupFlashMoERowsByGGUF(rep.Inventory))
			}
			return printFreetokenChatLab(rep.Chat)
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "JSON report")
	cmd.Flags().Bool("chat", true, "Deprecated: agent prefill lab is always printed")
	return cmd
}

func buildFreetokenReport() (freetokenReport, error) {
	profile := "mac-uma"
	if runtime.GOOS != "darwin" {
		profile = "5080-est"
	}
	entries, err := discover.ListFlashMoEInventory()
	if err != nil {
		return freetokenReport{}, err
	}
	ram := hostRAMGiB()
	nExp, k := 256, 6
	if len(entries) > 0 {
		if entries[0].ExpertCount >= 2 {
			nExp = int(entries[0].ExpertCount)
		}
		if entries[0].ExpertUsedCount > 0 {
			k = int(entries[0].ExpertUsedCount)
		}
	}
	a := freetokenlab.AdviseProfileFor(profile, nExp, k)
	rows := make([]flashMoEResolveRow, 0, len(entries))
	for _, e := range entries {
		rows = append(rows, flashMoEResolveRowFrom(e, ram))
	}
	groups := groupFlashMoERowsByGGUF(rows)
	blobs := make([]freetokenBlobAdvice, 0, len(groups))
	for _, g := range groups {
		blobs = append(blobs, freetokenBlobAdvice{
			Tags:         g.tags,
			SidecarReady: g.row.SidecarReady,
			EnvLine:      flashMoESlotBankEnvLine(g.row),
			Apply:        false,
		})
	}
	chatRows, err := server.ChatCompressLabCompare(4096)
	if err != nil {
		return freetokenReport{}, err
	}
	return freetokenReport{
		Profile:    profile,
		DoctorLine: a.DoctorLine(),
		Notes:      a.Notes,
		PrefetchOn: envconfig.FlashMoEPrefetchTemporal(),
		Inventory:  rows,
		Blobs:      blobs,
		Chat:       chatRows,
	}, nil
}

func printFreetokenChatLab(rows []server.ChatCompressLabRow) error {
	fmt.Println("\nagent-thread prefill lab (est. tokens, num_ctx=4096, no model load)")
	fmt.Printf("%-22s %8s %8s %8s\n", "policy", "reuse", "recompute", "compressed")
	for _, r := range rows {
		fmt.Printf("%-22s %8d %8d %8d\n", r.Name, r.Reuse, r.Recompute, r.Compressed)
	}
	fmt.Println("Mac steal: placeholder keeps exact prefix; summary invalidates tail KV; suffix-strip+anchor is the FreeToken bound")
	fmt.Println("sticky elide_from: echo compression.elide_from or send prompt_cache_key (server remembers 30m)")
	return nil
}
