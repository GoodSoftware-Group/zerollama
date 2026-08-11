package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEPrefillSwiGLUSmokeCommand() *cobra.Command {
	var (
		quick  bool
		dim    int
		hidden int
		seq    int
	)

	cmd := &cobra.Command{
		Use:    "ane-prefill-swiglu-smoke",
		Hidden: true,
		Short:  "In-process ANE SwiGLU session: fused gate+up+silu*+down + CPU golden (lab)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANEPrefillSwiGLUSmoke(cmd.Context(), os.Stdout, dim, hidden, seq, quick)
		},
	}
	cmd.Flags().IntVar(&dim, "dim", 512, "Model / residual dim")
	cmd.Flags().IntVar(&hidden, "hidden", 256, "FFN intermediate (gate/up width)")
	cmd.Flags().IntVar(&seq, "seq", 128, "Sequence / prompt length")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	return cmd
}
