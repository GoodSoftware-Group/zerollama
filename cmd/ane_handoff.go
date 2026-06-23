package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func NewANEDraftResolveCommand() *cobra.Command {
	var preferred string

	cmd := &cobra.Command{
		Use:    "ane-draft-resolve",
		Hidden: true,
		Short:  "List local eliza DFlash / draft-eagle3 tags for ANE hybrid research",
		RunE: func(_ *cobra.Command, _ []string) error {
			if preferred != "" {
				return discover.RunANEDraftResolveJSON(os.Stdout, preferred)
			}
			entries, err := discover.ListANEDraftInventory()
			if err != nil {
				return err
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(entries)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name")
	return cmd
}

func NewANEDraftInspectCommand() *cobra.Command {
	var preferred string

	cmd := &cobra.Command{
		Use:    "ane-draft-inspect",
		Hidden: true,
		Short:  "Inspect local eliza GGUF draft metadata and ANE proxy dimensions",
		RunE: func(_ *cobra.Command, _ []string) error {
			entries, err := discover.ListANEDraftInventory()
			if err != nil {
				return err
			}
			entry, ok := discover.SelectANEDraftModel(entries, preferred)
			if !ok {
				return fmt.Errorf("no ANE draft target in local inventory")
			}
			info, err := discover.ProbeANEDraftGGUF(entry.BaseGGUF, entry.DraftGGUF)
			if err != nil {
				return err
			}
			payload := struct {
				discover.ANEDraftEntry
				GGUF discover.ANEDraftGGUFInfo `json:"gguf"`
			}{
				ANEDraftEntry: entry,
				GGUF:          info,
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(payload)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name (default: first inventory entry)")
	return cmd
}

func NewANEDraftSmokeCommand() *cobra.Command {
	var preferred string
	var quick bool

	cmd := &cobra.Command{
		Use:    "ane-draft-smoke",
		Hidden: true,
		Short:  "Run ANE draft conv bench at GGUF-derived proxy dimensions",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANEDraftBenchForModel(cmd.Context(), os.Stdout, preferred, quick)
		},
	}
	cmd.Flags().StringVar(&preferred, "model", "", "Prefer tag or name (default: first inventory entry)")
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	return cmd
}

func NewANEHandoffSmokeCommand() *cobra.Command {
	var quick bool
	var suite bool
	var metal bool

	cmd := &cobra.Command{
		Use:    "ane-handoff-smoke",
		Hidden: true,
		Short:  "IOSurface handoff timing smoke (CPU or Metal producer → ANE → CPU consumer)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if suite {
				return discover.RunANEHandoffSuite(cmd.Context(), os.Stdout, quick)
			}
			if metal {
				return discover.RunANEMetalHandoffSmoke(cmd.Context(), os.Stdout, quick)
			}
			return discover.RunANEIOSurfaceSmoke(cmd.Context(), os.Stdout, quick)
		},
	}
	cmd.Flags().BoolVar(&quick, "quick", false, "Fewer iterations")
	cmd.Flags().BoolVar(&suite, "suite", false, "Run probe + draft + iosurface + metal handoff smokes")
	cmd.Flags().BoolVar(&metal, "metal", false, "Use Metal compute fill on ANE IOSurface (shared-memory handoff)")
	return cmd
}

func doctorANEHandoffDetail() string {
	h, err := discover.ProbeANEIOSurfaceSmoke(nil, true)
	if err != nil {
		return fmt.Sprintf("iosurface handoff: %v", err)
	}
	m, merr := discover.ProbeANEMetalHandoffSmoke(nil, true)
	if merr != nil {
		return fmt.Sprintf("iosurface write %.3f + eval %.3f + read %.3f ms; metal handoff: %v",
			h.WriteMS, h.EvalMS, h.ReadMS, merr)
	}
	line := fmt.Sprintf("iosurface write %.3f + eval %.3f + read %.3f ms; metal fill %.3f + eval %.3f ms",
		h.WriteMS, h.EvalMS, h.ReadMS, m.MetalFillMS, m.EvalMS)
	if discover.FindANEPrefillHandoffSmokeBin() != "" {
		if p, perr := discover.ProbeANEPrefillHandoffSmoke(nil, 256, 256, 128, true); perr == nil {
			line += fmt.Sprintf("; prefill handoff fill %.3f + eval %.3f ms", p.MetalFillMS, p.EvalMS)
		}
	}
	return line
}
