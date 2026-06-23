package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftSurfaceSmokeCommand() *cobra.Command {
	var (
		preferred string
		quick     bool
	)

	cmd := &cobra.Command{
		Use:    "ane-draft-surface-smoke",
		Hidden: true,
		Short:  "Metal→IOSurface→ANE draft conv at model proxy dims (ggml surface_id export target)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANEDraftSurfaceHandoffForModel(cmd.Context(), os.Stdout, preferred, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Local GGUF tag (default: first in ane-model-resolve)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	return cmd
}
