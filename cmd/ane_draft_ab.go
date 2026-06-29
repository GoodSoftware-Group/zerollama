package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftABCommand() *cobra.Command {
	var preferred string
	var quick bool
	var steps int
	var e2e bool
	var e2eTelemetry bool
	var e2eDrive bool
	var e2eDriveMode string
	var convDepth int

	cmd := &cobra.Command{
		Use:    "ane-draft-ab-smoke",
		Hidden: true,
		Short:  "B4/B6/B7 A/B: ANE in-process draft step vs Metal dflash e2e (lab port 11435)",
		RunE: func(c *cobra.Command, _ []string) error {
			driveMode := ""
			if e2eDrive || strings.TrimSpace(e2eDriveMode) != "" {
				driveMode = strings.ToLower(strings.TrimSpace(e2eDriveMode))
				if driveMode == "" {
					driveMode = "shadow"
				}
				switch driveMode {
				case "shadow", "force":
				default:
					return fmt.Errorf("invalid --e2e-drive-mode %q (use shadow or force)", e2eDriveMode)
				}
			}
			return discover.RunANEDraftABJSON(c.Context(), os.Stdout, preferred, steps, quick, e2e, e2eTelemetry, driveMode, convDepth)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name (e.g. eliza-1-2b-dflash)")
	cmd.Flags().IntVar(&steps, "steps", 0, "In-process ANE micro steps (default 5 quick / 10 full)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Shorter micro + e2e runs")
	cmd.Flags().BoolVar(&e2e, "e2e", false, "Also run llama-server Metal vs ANE-hook dflash legs on lab port")
	cmd.Flags().BoolVar(&e2eTelemetry, "e2e-telemetry", false, "Enable B6 golden telemetry on ANE e2e leg (adds overhead)")
	cmd.Flags().BoolVar(&e2eDrive, "e2e-drive", false, "B7: extract token_embd cache and enable drive mode (default shadow)")
	cmd.Flags().StringVar(&e2eDriveMode, "e2e-drive-mode", "", "B7 drive mode: shadow (log parity) or force (ANE token replaces Metal sampler)")
	cmd.Flags().IntVar(&convDepth, "conv-depth", 0, "Cap active ANE conv kernels (1=WEIGHT_FILE only; 0=full manifest chain)")
	return cmd
}
