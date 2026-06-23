package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEHybridSmokeCommand() *cobra.Command {
	var preferred string
	var quick bool

	cmd := &cobra.Command{
		Use:    "ane-hybrid-smoke",
		Hidden: true,
		Short:  "Hybrid lab: Metal IOSurface handoff + ANE draft conv at GGUF proxy dims",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANEHybridLabForModel(cmd.Context(), os.Stdout, preferred, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Local GGUF tag — draft or base model (see ane-model-resolve)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	return cmd
}
