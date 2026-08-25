package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEPrefillSweepCommand() *cobra.Command {
	var (
		preferred string
		quick     bool
		fullEmbed bool
		aneOnly   bool
		expertUp  bool
		variant   string
		ic        int
		oc        int
		seqsRaw   string
	)

	cmd := &cobra.Command{
		Use:    "ane-prefill-sweep",
		Hidden: true,
		Short:  "Sweep SEQ lengths — ANE vs Metal prefill matmul at fixed IC×OC",
		RunE: func(cmd *cobra.Command, _ []string) error {
			var seqs []int
			if seqsRaw != "" {
				parsed, err := discover.ParsePrefillSweepSeqs(seqsRaw)
				if err != nil {
					return err
				}
				seqs = parsed
			}
			if preferred != "" {
				return discover.RunANEPrefillSweepForModel(cmd.Context(), os.Stdout, preferred, seqs, quick, fullEmbed, aneOnly, variant, expertUp)
			}
			if ic <= 0 {
				ic = 256
			}
			if oc <= 0 {
				oc = ic
			}
			if expertUp && !cmd.Flags().Changed("oc") {
				oc = discover.PrefillExpertOC(ic)
			}
			return discover.RunANEPrefillSweep(cmd.Context(), os.Stdout, ic, oc, seqs, quick, aneOnly, variant)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Local GGUF tag — derive IC/OC from embedding (see ane-model-resolve)")
	cmd.Flags().IntVar(&ic, "ic", 256, "Input channels (ignored with --model)")
	cmd.Flags().IntVar(&oc, "oc", 256, "Output channels (ignored with --model unless --expert-up)")
	cmd.Flags().StringVar(&seqsRaw, "seqs", "", "Comma-separated SEQ grid (default: 128..4096 or quick subset)")
	cmd.Flags().StringVar(&variant, "variant", "baseline", "ANE MIL variant for compare/sweep")
	cmd.Flags().BoolVar(&expertUp, "expert-up", false, "Rectangular expert OC=IC/4 (MoE expert-up proxy)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations; default grid 128,512,2048")
	cmd.Flags().BoolVar(&fullEmbed, "full-embed", false, "Use full embedding_length for --model (default caps IC/OC at 2048)")
	cmd.Flags().BoolVar(&aneOnly, "ane-only", false, "Skip Metal/MPS legs — safe while GPU is busy")
	return cmd
}
