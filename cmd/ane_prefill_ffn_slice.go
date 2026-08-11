package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEPrefillFFNSliceSmokeCommand() *cobra.Command {
	var (
		quick    bool
		expertUp bool
		swiglu   bool
		int8     bool
		fuseGU   bool
		w8a8     bool
		w8a8x    bool
		int8In   bool
		tile     string
		ic       int
		oc       int
		seq      int
	)

	cmd := &cobra.Command{
		Use:    "ane-prefill-ffn-slice-smoke",
		Hidden: true,
		Short:  "In-process ANE FFN-slice: matmul / int8 / SwiGLU (+ w8a8-x, int8-in, tile) + parity (lab)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if expertUp && !cmd.Flags().Changed("oc") {
				oc = discover.PrefillExpertOC(ic)
			}
			return discover.RunANEPrefillFFNSliceSmoke(cmd.Context(), os.Stdout, ic, oc, seq, quick, swiglu, int8, fuseGU, w8a8, w8a8x, int8In, tile)
		},
	}
	cmd.Flags().IntVar(&ic, "ic", 2048, "Input channels (expert-up default 2048)")
	cmd.Flags().IntVar(&oc, "oc", 512, "Matmul OC / SwiGLU hidden (expert-up default 512)")
	cmd.Flags().IntVar(&seq, "seq", 512, "Sequence / prompt length")
	cmd.Flags().BoolVar(&expertUp, "expert-up", false, "Set OC=IC/4")
	cmd.Flags().BoolVar(&swiglu, "swiglu", false, "Fused SwiGLU (gate+up+silu+down) instead of single matmul")
	cmd.Flags().BoolVar(&int8, "int8", false, "int8 weight BLOBFILE (matmul or with --swiglu)")
	cmd.Flags().BoolVar(&fuseGU, "fuse-gu", false, "With --swiglu: fuse gate∥up into one 1x1 conv + slice")
	cmd.Flags().BoolVar(&w8a8, "w8a8", false, "With --swiglu --int8: quantize/dequantize hid before down")
	cmd.Flags().BoolVar(&w8a8x, "w8a8-x", false, "Also W8A8-quantize input x (implies --w8a8; auto-tiles)")
	cmd.Flags().BoolVar(&int8In, "int8-in", false, "Host writes int8 acts (½ surface; implies w8a8-x)")
	cmd.Flags().StringVar(&tile, "tile", "", "Spatial tile HxW or auto (default auto when --w8a8-x/--int8-in)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	return cmd
}
