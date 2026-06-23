package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEPrefillBenchCommand() *cobra.Command {
	var (
		preferred    string
		quick        bool
		fullEmbed    bool
		compareMetal bool
		ic           int
		oc           int
		seq          int
		tokens       int
	)

	cmd := &cobra.Command{
		Use:    "ane-prefill-bench",
		Hidden: true,
		Short:  "ANE dynamic matmul bench at prefill-like IC×OC×SEQ (maderix mil_dynamic proxy)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if compareMetal {
				withMPS := discover.FindMetalMPSPrefillBenchBin() != ""
				if preferred != "" {
					return discover.RunANEPrefillCompareForModel(cmd.Context(), os.Stdout, preferred, tokens, quick, withMPS, fullEmbed)
				}
				return discover.RunANEPrefillCompare(cmd.Context(), os.Stdout, ic, oc, seq, quick, withMPS)
			}
			if preferred != "" {
				return discover.RunANEPrefillBenchForModel(cmd.Context(), os.Stdout, preferred, tokens, quick, fullEmbed)
			}
			return discover.RunANEPrefillBench(cmd.Context(), os.Stdout, ic, oc, seq, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Local GGUF tag — derive IC/OC from embedding, SEQ from --tokens")
	cmd.Flags().IntVar(&tokens, "tokens", 512, "Prompt tokens for --model prefill proxy (capped at 4096)")
	cmd.Flags().IntVar(&ic, "ic", 256, "Input channels (ignored with --model)")
	cmd.Flags().IntVar(&oc, "oc", 256, "Output channels (ignored with --model)")
	cmd.Flags().IntVar(&seq, "seq", 512, "Sequence / prompt length (ignored with --model)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Cap --model SEQ at 128; fewer iterations")
	cmd.Flags().BoolVar(&fullEmbed, "full-embed", false, "Use full embedding_length for --model (default caps IC/OC at 2048)")
	cmd.Flags().BoolVar(&compareMetal, "compare-metal", false, "Run ANE + naive Metal matmul at same IC×OC×SEQ (+ MPS when built)")
	return cmd
}
