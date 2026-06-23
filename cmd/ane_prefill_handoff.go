package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEPrefillHandoffSmokeCommand() *cobra.Command {
	var (
		preferred string
		quick     bool
		fullEmbed bool
		ic        int
		oc        int
		seq       int
		tokens    int
	)

	cmd := &cobra.Command{
		Use:    "ane-prefill-handoff-smoke",
		Hidden: true,
		Short:  "Metal activation fill → IOSurface → ANE prefill matmul (ggml hook prototype)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if preferred != "" {
				return discover.RunANEPrefillHandoffForModel(cmd.Context(), os.Stdout, preferred, tokens, quick, fullEmbed)
			}
			return discover.RunANEPrefillHandoffSmoke(cmd.Context(), os.Stdout, ic, oc, seq, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Local GGUF tag — derive IC/OC from embedding, SEQ from --tokens")
	cmd.Flags().IntVar(&tokens, "tokens", 128, "Prompt tokens for --model handoff proxy")
	cmd.Flags().IntVar(&ic, "ic", 256, "Input channels (ignored with --model)")
	cmd.Flags().IntVar(&oc, "oc", 256, "Output channels (ignored with --model)")
	cmd.Flags().IntVar(&seq, "seq", 128, "Sequence length (ignored with --model)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	cmd.Flags().BoolVar(&fullEmbed, "full-embed", false, "Use full embedding_length for --model (default caps IC/OC at 2048)")
	return cmd
}
