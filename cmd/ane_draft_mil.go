package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftMILStatusCommand() *cobra.Command {
	var preferred string

	cmd := &cobra.Command{
		Use:    "ane-draft-mil-status",
		Hidden: true,
		Short:  "Eagle3 sidecar → MIL compile readiness and blockers",
		RunE: func(c *cobra.Command, _ []string) error {
			return discover.RunANEDraftMILStatusJSON(c.Context(), os.Stdout, preferred)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name")
	return cmd
}
