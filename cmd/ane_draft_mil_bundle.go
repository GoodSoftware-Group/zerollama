package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftMILBundleCommand() *cobra.Command {
	var preferred string

	cmd := &cobra.Command{
		Use:    "ane-draft-mil-bundle",
		Hidden: true,
		Short:  "Materialize dflash sidecar weights (ffn_gate + ffn_up conv proxy + gamma) for in-process ANE session",
		RunE: func(c *cobra.Command, _ []string) error {
			return discover.RunANEDraftMILBundleJSON(c.Context(), os.Stdout, preferred)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name")
	return cmd
}
