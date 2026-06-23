package cmd

import (
	"encoding/json"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEGGMLHookStatusCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:    "ane-ggml-hook-status",
		Hidden: true,
		Short:  "Report in-tree ggml IOSurface map API readiness",
		RunE: func(_ *cobra.Command, _ []string) error {
			st := discover.ProbeGGMLIOSurfaceHookStatus()
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(st)
		},
	}
	return cmd
}
