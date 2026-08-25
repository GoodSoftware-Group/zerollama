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
		expertUp     bool
		variant      string
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
			if expertUp && preferred == "" && !cmd.Flags().Changed("oc") {
				oc = discover.PrefillExpertOC(ic)
			}
			if compareMetal {
				withMPS := discover.FindMetalMPSPrefillBenchBin() != ""
				if preferred != "" {
					return discover.RunANEPrefillCompareForModel(cmd.Context(), os.Stdout, preferred, tokens, quick, withMPS, fullEmbed, variant, expertUp)
				}
				return discover.RunANEPrefillCompare(cmd.Context(), os.Stdout, ic, oc, seq, quick, withMPS, variant)
			}
			if preferred != "" {
				return discover.RunANEPrefillBenchForModel(cmd.Context(), os.Stdout, preferred, tokens, quick, fullEmbed, variant, expertUp)
			}
			return discover.RunANEPrefillBenchVariant(cmd.Context(), os.Stdout, ic, oc, seq, quick, variant)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Local GGUF tag — derive IC/OC from embedding, SEQ from --tokens")
	cmd.Flags().IntVar(&tokens, "tokens", 512, "Prompt tokens for --model prefill proxy (capped at 4096)")
	cmd.Flags().IntVar(&ic, "ic", 256, "Input channels (ignored with --model)")
	cmd.Flags().IntVar(&oc, "oc", 256, "Output channels (ignored with --model unless --expert-up)")
	cmd.Flags().IntVar(&seq, "seq", 512, "Sequence / prompt length (ignored with --model)")
	cmd.Flags().StringVar(&variant, "variant", "baseline", "ANE MIL variant: baseline|fp16|fp16-blob|fp16-native|fp16-conv|fp16-dyn|int8-conv")
	cmd.Flags().BoolVar(&expertUp, "expert-up", false, "Rectangular expert OC=IC/4 (MoE expert-up proxy)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Cap --model SEQ at 128; fewer iterations")
	cmd.Flags().BoolVar(&fullEmbed, "full-embed", false, "Use full embedding_length for --model (default caps IC/OC at 2048)")
	cmd.Flags().BoolVar(&compareMetal, "compare-metal", false, "Run ANE + naive Metal matmul at same IC×OC×SEQ (+ MPS when built)")
	return cmd
}
