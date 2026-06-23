package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftMILMapCommand() *cobra.Command {
	var preferred string

	cmd := &cobra.Command{
		Use:    "ane-draft-mil-map",
		Hidden: true,
		Short:  "Eagle3 GGUF tensor → ANE MIL slot mapping plan",
		RunE: func(c *cobra.Command, _ []string) error {
			return discover.RunANEDraftMILMapJSON(c.Context(), os.Stdout, preferred)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name")
	return cmd
}
