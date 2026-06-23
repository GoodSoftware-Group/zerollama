package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEModelResolveCommand() *cobra.Command {
	var preferred string

	cmd := &cobra.Command{
		Use:    "ane-model-resolve",
		Hidden: true,
		Short:  "List local GGUF tags with embedding_length for ANE prefill probes",
		RunE: func(_ *cobra.Command, _ []string) error {
			if preferred != "" {
				return discover.RunANEModelResolveJSON(os.Stdout, preferred)
			}
			entries, err := discover.ListANEModelInventory()
			if err != nil {
				return err
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(entries)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name (default: list all)")
	return cmd
}

func doctorANEModelInventoryDetail() string {
	entries, err := discover.ListANEModelInventory()
	if err != nil {
		return fmt.Sprintf("model inventory: %v", err)
	}
	if len(entries) == 0 {
		return "model inventory: none"
	}
	withEmbed := 0
	for _, e := range entries {
		if e.EmbeddingLength > 0 {
			withEmbed++
		}
	}
	return fmt.Sprintf("model inventory: %d tags (%d with embedding_length)", len(entries), withEmbed)
}
