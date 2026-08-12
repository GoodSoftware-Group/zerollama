package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/ollama/ollama/internal/modelrepair"
)

// runDoctorRepairModels implements `zerollama doctor --repair-models`.
//
// Why a dedicated flag (not doctor --fix / zerollama repair):
//   - --fix is host bootstrap; rewriting tags under it is surprising.
//   - zerollama repair only refreshes GGUF metadata layers — no live empty-
//     response or slash-collapse detection.
//   - Dry-run by default; --apply is required to POST /api/create.
//
// See docs/doctor-model-repair.md.
func runDoctorRepairModels(args []string, apply, jsonOut, allLocal bool) error {
	base, _ := doctorProbeGoAPI()
	if base == "" {
		return fmt.Errorf("no Go API reachable — start zerollama serve, then re-run doctor --repair-models")
	}
	api := modelrepair.NewHTTPAPI(base)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	names, err := modelrepair.ListTargets(ctx, api, args, allLocal)
	if err != nil {
		return err
	}
	if len(names) == 0 {
		if allLocal {
			return fmt.Errorf("no local models in /api/tags")
		}
		return fmt.Errorf("no models to scan — pass MODEL args, --all-local, or load a runner (zerollama run …)")
	}

	opts := modelrepair.Options{Apply: apply}
	if !jsonOut {
		opts.Progress = func(s string) { fmt.Fprintln(os.Stderr, s) }
	}

	reports, err := modelrepair.DiagnoseAll(ctx, api, names, opts)
	if err != nil {
		return err
	}

	if jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(reports); err != nil {
			return err
		}
	} else {
		printDoctorRepairHuman(reports, apply)
	}

	var findings, applyErrs, manual int
	for _, r := range reports {
		if r.HasFindings() {
			findings++
		}
		if r.ApplyError != "" {
			applyErrs++
		}
		if r.Skipped {
			findings++
		}
		manual += len(r.ManualReview)
	}
	if applyErrs > 0 {
		return fmt.Errorf("doctor --repair-models: %d apply error(s)", applyErrs)
	}
	if findings > 0 && !apply {
		return fmt.Errorf("doctor --repair-models: %d model(s) need repair (re-run with --apply to write)", findings)
	}
	if findings == 0 && manual > 0 {
		return fmt.Errorf("doctor --repair-models: %d manual-review note(s) (invasive recipes skipped for non-qwen3)", manual)
	}
	return nil
}

func printDoctorRepairHuman(reports []modelrepair.Report, apply bool) {
	fmt.Println("== doctor --repair-models ==")
	if !apply {
		fmt.Println("(dry-run — pass --apply to recreate tags in place)")
	}
	fmt.Println()
	for _, r := range reports {
		fmt.Println(r.Summary())
		for _, f := range r.Findings {
			fmt.Printf("  [%s] %s\n", f.Recipe, f.Detail)
			if f.FixHint != "" {
				fmt.Printf("      → %s\n", f.FixHint)
			}
		}
		for _, u := range r.Unfixable {
			fmt.Printf("  [unfixable] %s\n", u)
		}
		for _, m := range r.ManualReview {
			fmt.Printf("  [manual-review] %s\n", m)
		}
		for _, s := range r.StillBroken {
			fmt.Printf("  [still-broken] %s\n", s)
		}
		if r.ApplyError != "" {
			fmt.Printf("  [apply-error] %s\n", r.ApplyError)
		}
		if r.Patch != nil && !apply {
			fmt.Println("  --- proposed Modelfile ---")
			fmt.Print(r.Patch.Modelfile)
			fmt.Println("  --- end ---")
		}
		if r.Applied {
			fmt.Println("  applied via /api/create")
		}
		fmt.Println()
	}
}
