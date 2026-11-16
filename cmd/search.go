package cmd

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/server"
)

func NewSearchCommand() *cobra.Command {
	var all bool
	cmd := &cobra.Command{
		Use:   "search QUERY",
		Short: "Search Pirate Face for GGUF model torrents",
		Long: `Search the Pirate Face catalog (https://pirateface.co/s/QUERY).

Does not need zerollama serve. Default view keeps seeded GGUF-like rows
(names with gguf or a quant tag such as Q4_K_M). Use --all for other formats.

  zerollama search qwen2.5
  zerollama search "kimi k3" --all

The printed pull command is the next lookup. If that repo has several GGUFs,
pull lists them and stops before downloading.`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			query := joinSearchQuery(args)
			rows, err := server.FetchPirateFaceSearch(cmd.Context(), query)
			if err != nil {
				return err
			}
			shown, hidden, zeroOnly := server.FilterPirateFaceSearch(rows, all)
			if len(shown) == 0 {
				fmt.Fprintf(os.Stderr, "no pirateface results for %q\n", query)
				return nil
			}
			if zeroOnly {
				fmt.Fprintln(os.Stderr, "nobody is seeding these matches")
			}
			w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
			fmt.Fprintln(w, "REPO\tSIZE\tSEEDS\tPULL")
			for _, r := range shown {
				fmt.Fprintf(w, "%s\t%s\t%d\t%s\n", r.Repo, r.Size, r.Seeds, server.FormatPirateFaceSearchPull(r.Repo))
			}
			if err := w.Flush(); err != nil {
				return err
			}
			if hidden > 0 {
				if all {
					fmt.Fprintf(os.Stderr, "(%d more truncated; refine the query)\n", hidden)
				} else {
					fmt.Fprintf(os.Stderr, "(%d non-GGUF or extra rows hidden; pass --all)\n", hidden)
				}
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "Include non-GGUF formats (MLX, AWQ, …)")
	return cmd
}

func joinSearchQuery(args []string) string {
	out := ""
	for i, a := range args {
		if i > 0 {
			out += " "
		}
		out += a
	}
	return out
}
