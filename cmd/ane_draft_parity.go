package cmd

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftParityCommand() *cobra.Command {
	var preferred string
	var quick bool
	var telemetry bool
	var useConv bool
	var forceDrive bool
	var tokenShadow bool

	cmd := &cobra.Command{
		Use:    "ane-draft-parity-smoke",
		Hidden: true,
		Short:  "Shadow ANE vs Metal draft token parity on lab port 11435 (matmul ffn_gate, not 8-conv chain)",
		RunE: func(c *cobra.Command, _ []string) error {
			driveMode := "shadow"
			if forceDrive {
				driveMode = "force"
			}
			driveMetrics := "hidden"
			if tokenShadow {
				driveMetrics = "both"
			}
			return discover.RunANEDraftParityJSON(c.Context(), os.Stdout, preferred, discover.ANEDraftParityOpts{
				Quick:        quick,
				UseMatmul:    !useConv,
				Telemetry:    telemetry,
				DriveMode:    driveMode,
				DriveMetrics: driveMetrics,
			})
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name (e.g. eliza-1-2b-dflash)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Shorter e2e run (16 tokens)")
	cmd.Flags().BoolVar(&telemetry, "telemetry", false, "Golden CPU vs ANE cosine on ANE leg")
	cmd.Flags().BoolVar(&useConv, "conv", false, "Use single conv proxy instead of default matmul ffn_gate kernel")
	cmd.Flags().BoolVar(&forceDrive, "force", false, "B7 force: ANE tied-embed token replaces Metal sampler")
	cmd.Flags().BoolVar(&tokenShadow, "token-shadow", false, "Matmul shadow with tied-embed token match + hidden_cos (DRIVE_METRICS=both)")
	return cmd
}
