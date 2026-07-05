package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/internal/blobaudit"
)

// NewBlobsCommand registers blob storage inspection commands.
func NewBlobsCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "blobs",
		Short: "Inspect OLLAMA_MODELS blob storage",
	}
	cmd.AddCommand(newBlobsAuditCommand())
	return cmd
}

func newBlobsAuditCommand() *cobra.Command {
	var jsonOut bool
	cmd := &cobra.Command{
		Use:   "audit",
		Short: "Summarize blob usage, orphans, and per-tag storage",
		Long: `Walk ~/.ollama/models (or OLLAMA_MODELS) and report:

  • total on-disk blob bytes vs manifest-referenced bytes
  • orphan blobs not referenced by any manifest (PruneLayers candidates)
  • per-tag rollups (MLX tensor layer counts vs GGUF)
  • content-addressed dedupe across tags

Does not delete anything. Orphan blobs are removed on serve startup by
PruneLayers unless OLLAMA_NOPRUNE=1.`,
		RunE: func(_ *cobra.Command, _ []string) error {
			report, err := blobaudit.Audit()
			if err != nil {
				return err
			}
			if jsonOut {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(report)
			}
			fmt.Fprint(os.Stdout, blobaudit.FormatHuman(report))
			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "Print JSON report")
	return cmd
}
