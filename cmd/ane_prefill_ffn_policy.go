package cmd

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEPrefillFFNPolicySmokeCommand() *cobra.Command {
	return &cobra.Command{
		Use:    "ane-prefill-ffn-policy-smoke",
		Hidden: true,
		Short:  "Unit smoke for fail-closed ZEROLLAMA_ANE_FFN_* policy (lab)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			bin := discover.FindANEPrefillFFNPolicySmokeBin()
			if bin == "" {
				return fmt.Errorf("ane-prefill-ffn-policy-smoke not found — run ./scripts/ane/ane_probe_build.sh")
			}
			c := exec.CommandContext(cmd.Context(), bin)
			c.Stdout = os.Stdout
			c.Stderr = os.Stderr
			return c.Run()
		},
	}
}
