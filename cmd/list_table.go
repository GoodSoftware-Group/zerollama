package cmd

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/mattn/go-runewidth"
	"github.com/olekukonko/tablewriter"
	"golang.org/x/term"
)

// terminalWidth returns the stdout terminal columns when attached to a TTY.
// Falls back to $COLUMNS. Returns 0 when unknown (piped / tests) so callers can
// keep the wide single-line table for scripts.
func terminalWidth() int {
	if w, _, err := term.GetSize(int(os.Stdout.Fd())); err == nil && w > 0 {
		return w
	}
	if c := strings.TrimSpace(os.Getenv("COLUMNS")); c != "" {
		if n, err := strconv.Atoi(c); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

// useCompactList is true when we know the terminal is too narrow for the wide
// PARAMS/CTX/PERF/MODIFIED single-line table (~100+ cols with long names).
func useCompactList(width int) bool {
	return width > 0 && width < 100
}

// useCompactPs is true when PROJECT/SESSION columns would overflow.
func useCompactPs(width int, showProjects bool) bool {
	if width <= 0 {
		return false
	}
	if showProjects {
		return width < 120
	}
	return width < 90
}

func newPlainTable(w io.Writer) *tablewriter.Table {
	table := tablewriter.NewWriter(w)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("    ")
	return table
}

type listTableRow struct {
	name     string
	id       string
	size     string
	params   string
	ctx      string
	perf     string
	modified string
}

func printListTable(w io.Writer, rows []listTableRow, width int) {
	if useCompactList(width) {
		printListCompact(w, rows, width)
		return
	}
	printListWide(w, rows)
}

func printListWide(w io.Writer, rows []listTableRow) {
	table := newPlainTable(w)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "PARAMS", "CTX", "PERF", "MODIFIED"})
	for _, r := range rows {
		table.Append([]string{r.name, r.id, r.size, r.params, r.ctx, r.perf, r.modified})
	}
	table.Render()
}

func printListCompact(w io.Writer, rows []listTableRow, width int) {
	if width < 40 {
		width = 80
	}
	// Fixed trailing columns so every line stays ≤ width.
	const (
		sizeW = 7
		ctxW  = 10
		perfW = 6
		modW  = 14
		idW   = 12
		gap   = 2
	)
	// NAME + gaps + SIZE + CTX + PERF + MODIFIED
	nameW := width - (sizeW + ctxW + perfW + modW + gap*4)
	if nameW < 12 {
		nameW = 12
	}

	fmt.Fprintf(w, "%s%s%s%s%s%s%s%s%s\n",
		padRight("NAME", nameW), strings.Repeat(" ", gap),
		padRight("SIZE", sizeW), strings.Repeat(" ", gap),
		padRight("CTX", ctxW), strings.Repeat(" ", gap),
		padRight("PERF", perfW), strings.Repeat(" ", gap),
		padRight("MODIFIED", modW),
	)
	fmt.Fprintf(w, "%s%s%s%s\n",
		strings.Repeat(" ", gap),
		padRight("ID", idW), strings.Repeat(" ", gap),
		"PARAMS",
	)

	for _, r := range rows {
		fmt.Fprintf(w, "%s%s%s%s%s%s%s%s%s\n",
			padRight(truncateRunes(r.name, nameW), nameW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.size, sizeW), sizeW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.ctx, ctxW), ctxW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.perf, perfW), perfW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.modified, modW), modW),
		)
		paramsW := width - gap - idW - gap
		if paramsW < 8 {
			paramsW = 8
		}
		fmt.Fprintf(w, "%s%s%s%s\n",
			strings.Repeat(" ", gap),
			padRight(truncateRunes(r.id, idW), idW), strings.Repeat(" ", gap),
			truncateRunes(r.params, paramsW),
		)
	}
}

type psTableRow struct {
	name      string
	project   string
	session   string
	id        string
	size      string
	processor string
	context   string
	until     string
}

func printPsTable(w io.Writer, rows []psTableRow, showProjects bool, width int) {
	if useCompactPs(width, showProjects) {
		printPsCompact(w, rows, showProjects, width)
		return
	}
	printPsWide(w, rows, showProjects)
}

func printPsWide(w io.Writer, rows []psTableRow, showProjects bool) {
	table := newPlainTable(w)
	header := []string{"NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL"}
	if showProjects {
		header = []string{"NAME", "PROJECT", "SESSION", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL"}
	}
	table.SetHeader(header)
	for _, r := range rows {
		if showProjects {
			table.Append([]string{r.name, r.project, r.session, r.id, r.size, r.processor, r.context, r.until})
			continue
		}
		table.Append([]string{r.name, r.id, r.size, r.processor, r.context, r.until})
	}
	table.Render()
}

func printPsCompact(w io.Writer, rows []psTableRow, showProjects bool, width int) {
	if width < 40 {
		width = 80
	}
	const (
		sizeW  = 7
		procW  = 10
		ctxW   = 8
		untilW = 16
		gap    = 2
	)
	nameW := width - (sizeW + procW + ctxW + untilW + gap*4)
	if nameW < 12 {
		nameW = 12
	}

	fmt.Fprintf(w, "%s%s%s%s%s%s%s%s%s\n",
		padRight("NAME", nameW), strings.Repeat(" ", gap),
		padRight("SIZE", sizeW), strings.Repeat(" ", gap),
		padRight("PROCESSOR", procW), strings.Repeat(" ", gap),
		padRight("CONTEXT", ctxW), strings.Repeat(" ", gap),
		padRight("UNTIL", untilW),
	)
	if showProjects {
		projW := min(28, width/2)
		fmt.Fprintf(w, "%s%s%s%s\n",
			strings.Repeat(" ", gap),
			padRight("PROJECT", projW), strings.Repeat(" ", gap),
			"SESSION",
		)
	} else {
		fmt.Fprintf(w, "%s%s\n", strings.Repeat(" ", gap), "ID")
	}

	for _, r := range rows {
		fmt.Fprintf(w, "%s%s%s%s%s%s%s%s%s\n",
			padRight(truncateRunes(r.name, nameW), nameW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.size, sizeW), sizeW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.processor, procW), procW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.context, ctxW), ctxW), strings.Repeat(" ", gap),
			padRight(truncateRunes(r.until, untilW), untilW),
		)
		if showProjects {
			projW := min(28, width/2)
			sessW := width - gap - projW - gap
			if sessW < 8 {
				sessW = 8
			}
			fmt.Fprintf(w, "%s%s%s%s\n",
				strings.Repeat(" ", gap),
				padRight(truncateRunes(r.project, projW), projW), strings.Repeat(" ", gap),
				truncateRunes(r.session, sessW),
			)
			continue
		}
		fmt.Fprintf(w, "%s%s\n", strings.Repeat(" ", gap), truncateRunes(r.id, width-gap))
	}
}

func padRight(s string, width int) string {
	n := runewidth.StringWidth(s)
	if n >= width {
		return s
	}
	return s + strings.Repeat(" ", width-n)
}

func truncateRunes(s string, width int) string {
	if width <= 0 {
		return ""
	}
	if runewidth.StringWidth(s) <= width {
		return s
	}
	if width <= 1 {
		return "…"
	}
	return runewidth.Truncate(s, width, "…")
}
