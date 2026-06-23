package cmd

import (
	"encoding/json"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftDaemonSmokeCommand() *cobra.Command {
	var preferred string
	var quick bool
	var benchOnly bool

	cmd := &cobra.Command{
		Use:    "ane-draft-daemon-smoke",
		Hidden: true,
		Short:  "Persistent ANE draft daemon — compile once, reuse kernel across eval sessions",
		RunE: func(c *cobra.Command, _ []string) error {
			if benchOnly {
				ch, sp := 0, 0
				if preferred != "" {
					proxy, err := discover.ResolveANEModelProxyDims(preferred)
					if err != nil {
						return err
					}
					ch, sp = proxy.ProxyChannels, proxy.ProxySpatial
				}
				res, err := discover.ProbeANEDraftDaemonBench(c.Context(), ch, sp, quick)
				if err != nil {
					enc := json.NewEncoder(os.Stdout)
					_ = enc.Encode(res)
					return err
				}
				return json.NewEncoder(os.Stdout).Encode(res)
			}
			return discover.RunANEDraftDaemonSmokeJSON(c.Context(), os.Stdout, preferred, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name for proxy dims")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer bench iterations")
	cmd.Flags().BoolVar(&benchOnly, "bench", false, "One-shot bench mode (no session reuse demo)")
	return cmd
}
