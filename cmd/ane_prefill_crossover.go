package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEPrefillCrossoverCommand() *cobra.Command {
	var (
		preferred string
		quick     bool
		fullEmbed bool
		aneOnly   bool
		seq       int
		widthsRaw string
		variant   string
	)

	cmd := &cobra.Command{
		Use:    "ane-prefill-crossover",
		Hidden: true,
		Short:  "Scan IC widths to find ANE vs MPS crossover at fixed SEQ",
		RunE: func(cmd *cobra.Command, _ []string) error {
			widths, err := discover.ParseCrossoverWidths(widthsRaw)
			if err != nil {
				return err
			}
			return discover.RunANEPrefillCrossoverJSON(cmd.Context(), os.Stdout, preferred, seq, widths, quick, fullEmbed, aneOnly, variant)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Scan widths up to model embedding (see ane-model-resolve)")
	cmd.Flags().IntVar(&seq, "seq", 512, "Fixed prompt length for width scan")
	cmd.Flags().StringVar(&widthsRaw, "widths", "", "IC grid: comma list or from:to:step (default quick grid 512..2048)")
	cmd.Flags().StringVar(&variant, "variant", "baseline", "ANE MIL variant for width scan")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations; default width grid")
	cmd.Flags().BoolVar(&fullEmbed, "full-embed", false, "Include full model embedding width with --model")
	cmd.Flags().BoolVar(&aneOnly, "ane-only", false, "Skip Metal/MPS legs — ANE throughput scan only (GPU busy safe)")
	return cmd
}
