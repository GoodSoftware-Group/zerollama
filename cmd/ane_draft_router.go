package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftRouterSmokeCommand() *cobra.Command {
	var preferred string
	var quick bool
	var steps int

	cmd := &cobra.Command{
		Use:    "ane-draft-router-smoke",
		Hidden: true,
		Short:  "Multi-step ANE draft router smoke (requires ZEROLLAMA_ANE_DRAFT=1)",
		RunE: func(c *cobra.Command, _ []string) error {
			return discover.RunANEDraftRouterSmokeJSON(c.Context(), os.Stdout, preferred, steps, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name for proxy dims")
	cmd.Flags().IntVar(&steps, "steps", 0, "Draft steps (default 3 quick / 5 full)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer steps")
	return cmd
}
