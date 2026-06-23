package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftMILExtractCommand() *cobra.Command {
	var preferred, tensorName, outputPath string

	cmd := &cobra.Command{
		Use:    "ane-draft-mil-extract",
		Hidden: true,
		Short:  "Extract Eagle3 sidecar tensor → ANE MIL weight blob for draft conv proxy",
		RunE: func(c *cobra.Command, _ []string) error {
			return discover.RunANEDraftMILExtractJSON(c.Context(), os.Stdout, preferred, tensorName, outputPath)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name")
	cmd.Flags().StringVar(&tensorName, "tensor", "", "Sidecar tensor (default blk.0.ffn_gate.weight)")
	cmd.Flags().StringVar(&outputPath, "out", "", "Write BLOBFILE weight blob to path")
	return cmd
}
