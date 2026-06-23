package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANELabStatusCommand() *cobra.Command {
	var (
		sweep     bool
		preferred string
		fullEmbed bool
		aneOnly   bool
	)

	cmd := &cobra.Command{
		Use:    "ane-lab-status",
		Hidden: true,
		Short:  "ANE lab binary inventory and optional prefill sweep snapshot",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANELabStatusJSON(cmd.Context(), os.Stdout, discover.ANELabStatusOpts{
				WithPrefillSweep: sweep,
				Model:            preferred,
				FullEmbed:        fullEmbed,
				AneOnly:          aneOnly,
			})
		},
	}
	cmd.Flags().BoolVar(&sweep, "sweep", false, "Run quick prefill SEQ sweep (256×256, or --model IC×OC)")
	cmd.Flags().StringVar(&preferred, "model", "", "Sweep at this model's hidden size (with --sweep)")
	cmd.Flags().BoolVar(&fullEmbed, "full-embed", false, "Use full embedding_length for --model (default caps at 2048)")
	cmd.Flags().BoolVar(&aneOnly, "ane-only", false, "Prefill sweep skips Metal/MPS (GPU busy safe)")
	return cmd
}
