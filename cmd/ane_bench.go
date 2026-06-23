package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEBenchCommand() *cobra.Command {
	var quick bool
	cmd := &cobra.Command{
		Use:    "ane-bench",
		Hidden: true,
		Short:  "Run ANE peak throughput bench (conv-stack matmul proxy)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANEMatmulBench(cmd.Context(), os.Stdout, quick)
		},
	}
	cmd.Flags().BoolVar(&quick, "quick", false, "Shorter depth/iters for CI")
	return cmd
}

func NewANEDraftBenchCommand() *cobra.Command {
	var quick bool
	cmd := &cobra.Command{
		Use:    "ane-draft-bench",
		Hidden: true,
		Short:  "Run ANE draft-step matmul latency bench (speculative decode proxy)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANEDraftBench(cmd.Context(), os.Stdout, quick)
		},
	}
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	return cmd
}

func doctorANEBenchDetail() string {
	peak, err := discover.ProbeANEMatmulBench(nil, true)
	if err != nil {
		return fmt.Sprintf("peak bench unavailable: %v", err)
	}
	draft, derr := discover.ProbeANEDraftBench(nil, true)
	if derr != nil {
		return fmt.Sprintf("peak %.1f TFLOPS (eval %.2f ms); draft bench: %v",
			peak.TFLOPS, peak.EvalMS, derr)
	}
	return fmt.Sprintf("peak %.1f TFLOPS (eval %.2f ms); draft step %.3f ms (ch=%d sp=%d)",
		peak.TFLOPS, peak.EvalMS, draft.EvalMS, draft.Channels, draft.Spatial)
}
