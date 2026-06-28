package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEInprocessSmokeCommand() *cobra.Command {
	var preferred string
	var quick bool
	var steps int

	cmd := &cobra.Command{
		Use:    "ane-inprocess-smoke",
		Hidden: true,
		Short:  "Same-process ANE draft ggml map + eval loop (B1 integration contract)",
		RunE: func(c *cobra.Command, _ []string) error {
			return discover.RunANEInprocessSmokeJSON(c.Context(), os.Stdout, preferred, steps, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag for proxy dims and sidecar weight extract")
	cmd.Flags().IntVar(&steps, "steps", 0, "Draft steps (default 3 quick / 5 full)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer steps")
	return cmd
}
