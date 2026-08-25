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
		expertUp  bool
		steady    bool
		variant   string
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
			if expertUp && preferred == "" && !cmd.Flags().Changed("oc") {
				oc = discover.PrefillExpertOC(ic)
			}
			if preferred != "" {
				return discover.RunANEPrefillHandoffForModel(cmd.Context(), os.Stdout, preferred, tokens, quick, fullEmbed, variant, expertUp, steady)
			}
			return discover.RunANEPrefillHandoffSmoke(cmd.Context(), os.Stdout, ic, oc, seq, quick, variant, steady)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Local GGUF tag — derive IC/OC from embedding, SEQ from --tokens")
	cmd.Flags().IntVar(&tokens, "tokens", 128, "Prompt tokens for --model handoff proxy")
	cmd.Flags().IntVar(&ic, "ic", 256, "Input channels (ignored with --model)")
	cmd.Flags().IntVar(&oc, "oc", 256, "Output channels (ignored with --model unless --expert-up)")
	cmd.Flags().IntVar(&seq, "seq", 128, "Sequence length (ignored with --model)")
	cmd.Flags().StringVar(&variant, "variant", "baseline", "ANE MIL variant: baseline|fp16-conv|fp16-dyn")
	cmd.Flags().BoolVar(&expertUp, "expert-up", false, "Rectangular expert OC=IC/4 (MoE expert-up proxy)")
	cmd.Flags().BoolVar(&steady, "steady", false, "Fill once; time ANE eval only (ggml zero-copy proxy)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	cmd.Flags().BoolVar(&fullEmbed, "full-embed", false, "Use full embedding_length for --model (default caps IC/OC at 2048)")
	return cmd
}
