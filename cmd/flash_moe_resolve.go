package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewFlashMoEResolveCommand() *cobra.Command {
	var (
		asJSON   bool
		preferred string
		listAll  bool
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
				if asJSON {
					enc := json.NewEncoder(os.Stdout)
					enc.SetIndent("", "  ")
					return enc.Encode(entries)
				}
				if len(entries) == 0 {
					fmt.Println("no local MoE models found")
					return nil
				}
				for _, e := range entries {
					status := "sidecar=missing"
					if e.SidecarReady {
						status = "sidecar=ready"
					}
					fmt.Printf("%s  %s  experts=%d  %s\n", e.Tag, e.GGUFPath, e.ExpertCount, status)
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
			if asJSON {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(entry)
			}
			fmt.Printf("tag=%s\ngguf=%s\nsidecar=%s\nsidecar_ready=%v\n", entry.Tag, entry.GGUFPath, entry.Sidecar, entry.SidecarReady)
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "JSON output")
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer this tag or model name")
	cmd.Flags().BoolVar(&listAll, "list", false, "List all local MoE models")
	return cmd
}
