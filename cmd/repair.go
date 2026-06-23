package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/server"
)

// NewRepairCommand registers `zerollama repair` — manifest metadata hygiene without re-download.
func NewRepairCommand() *cobra.Command {
	var all bool
	var write bool

	cmd := &cobra.Command{
		Use:   "repair [MODEL...]",
		Short: "Refresh manifest params/config from GGUF headers (dry-run by default)",
		Long: `Rewrite manifest metadata layers from local GGUF headers without re-downloading weights.

Caps excessive manifest num_ctx (8192), fills missing parser/arch/template, and updates config.
Pass --write to apply; default is dry-run. Does not require a running server.`,
		Args: cobra.ArbitraryArgs,
		RunE: func(_ *cobra.Command, args []string) error {
			if all && len(args) > 0 {
				return fmt.Errorf("use either --all or model names, not both")
			}
			if !all && len(args) == 0 {
				return fmt.Errorf("specify model name(s) or --all")
			}

			opts := server.RepairOptions{Write: write}
			var results []*server.RepairResult
			var err error
			if all {
				results, err = server.RepairAll(opts)
			} else {
				for _, name := range args {
					r, rerr := server.RepairModel(name, opts)
					if rerr != nil {
						return rerr
					}
					results = append(results, r)
				}
			}
			if err != nil {
				return err
			}

			printRepairResults(results, write)
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "Repair every local manifest")
	cmd.Flags().BoolVar(&write, "write", false, "Apply changes (default is dry-run)")
	return cmd
}

func printRepairResults(results []*server.RepairResult, write bool) {
	for _, r := range results {
		fmt.Fprintf(os.Stdout, "%s", r.Name)
		if r.Skipped {
			fmt.Fprintf(os.Stdout, ": skipped (%s)\n", r.Reason)
			continue
		}
		if len(r.Changes) == 0 {
			fmt.Fprintf(os.Stdout, ": ok (no changes)\n")
			continue
		}
		fmt.Fprintln(os.Stdout)
		for _, c := range r.Changes {
			switch {
			case c.From == "":
				fmt.Fprintf(os.Stdout, "  %s: (missing) -> %s\n", c.Field, c.To)
			case c.To == "":
				fmt.Fprintf(os.Stdout, "  %s: %s -> (removed)\n", c.Field, c.From)
			default:
				fmt.Fprintf(os.Stdout, "  %s: %s -> %s\n", c.Field, c.From, c.To)
			}
		}
		if r.Written {
			fmt.Fprintf(os.Stdout, "  (written)\n")
		}
	}
	if !write {
		changed := false
		for _, r := range results {
			if !r.Skipped && len(r.Changes) > 0 && !r.Written {
				changed = true
				break
			}
		}
		if changed {
			fmt.Fprintln(os.Stdout, strings.TrimSpace(`
dry-run only — pass --write to apply`))
		}
	}
}
