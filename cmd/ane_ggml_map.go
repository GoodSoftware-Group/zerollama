package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEGGMLMapSmokeCommand() *cobra.Command {
	var preferred string
	var quick bool

	cmd := &cobra.Command{
		Use:    "ane-ggml-map-smoke",
		Hidden: true,
		Short:  "ggml_metal_buffer_map-equivalent fill on daemon IOSurface, then ANE eval",
		RunE: func(c *cobra.Command, _ []string) error {
			return discover.RunANEGGMLMapSmokeJSON(c.Context(), os.Stdout, preferred, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name for proxy dims")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer fill iterations")
	return cmd
}
