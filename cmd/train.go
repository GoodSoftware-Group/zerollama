// Package cmd — T11b train helpers.
//
// WHY a Go CLI that shells to Python: operators already have `zerollama` on PATH after
// install; recipe logic stays in training_recipes.py (one engine, stdlib CSV/JSONL, optional
// pypdf/docx). Reimplementing ingest in Go would drift from the Python unit tests.
//
// WHY not auto-submit to /api/train: ingest is offline; operators choose host, T6 window,
// and priority when they POST.
package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/x/trainingworker"
)

// NewTrainCommand groups offline training helpers (recipes today; not the GPU job API).
// WHY separate from serve/embed: no CPython job queue, no VRAM broker — just file transforms.
func NewTrainCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "train",
		Short: "Offline training helpers (data recipes)",
	}
	cmd.AddCommand(newTrainRecipeCommand())
	return cmd
}

func newTrainRecipeCommand() *cobra.Command {
	var (
		outPath     string
		format      string
		preset      string
		promptCol   string
		responseCol string
		chunkChars  int
		modelName   string
		pythonBin   string
	)
	cmd := &cobra.Command{
		Use:   "recipe INPUT",
		Short: "Ingest CSV/JSONL/TXT/(PDF|DOCX) into training_data JSON/JSONL",
		Long: `Convert a source file into rows for POST /api/train/jobs.

Delegates to repo-root training_recipes.py (stdlib CSV/JSONL/TXT; optional pypdf /
python-docx). Output .jsonl is one object per line; .json wraps {training_data:[…]}
(and optional model_name) for easy merge into a train payload.

Chat-export presets (JSON/JSONL): --preset sharegpt|openai|alpaca|prompt_response|chatgpt_export|claude_export|auto

Examples:
  zerollama train recipe data.csv -o sft.jsonl
  zerollama train recipe sharegpt.json --preset sharegpt -o sft.jsonl
  zerollama train recipe conversations.json --preset chatgpt_export -o sft.jsonl
  zerollama train recipe claude.json --preset claude_export -o sft.jsonl
  zerollama train recipe sft.jsonl -o payload.json --model-name Qwen/Qwen2.5-0.5B-Instruct
`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runTrainRecipe(args[0], trainRecipeOpts{
				Out:         outPath,
				Format:      format,
				Preset:      preset,
				PromptCol:   promptCol,
				ResponseCol: responseCol,
				ChunkChars:  chunkChars,
				ModelName:   modelName,
				PythonBin:   pythonBin,
			})
		},
	}
	cmd.Flags().StringVarP(&outPath, "output", "o", "", "Output .jsonl or .json (required)")
	cmd.Flags().StringVar(&format, "format", "auto", "auto|csv|tsv|jsonl|json|txt|pdf|docx")
	cmd.Flags().StringVar(&preset, "preset", "auto", "auto|sharegpt|openai|alpaca|prompt_response|chatgpt_export|claude_export")
	cmd.Flags().StringVar(&promptCol, "prompt-col", "prompt", "CSV prompt column")
	cmd.Flags().StringVar(&responseCol, "response-col", "response", "CSV response column")
	cmd.Flags().IntVar(&chunkChars, "chunk-chars", 1200, "PDF/DOCX chunk size")
	cmd.Flags().StringVar(&modelName, "model-name", "", "Embed in .json payload wrapper")
	cmd.Flags().StringVar(&pythonBin, "python", "", "Python interpreter (default: .venv-training or python3)")
	_ = cmd.MarkFlagRequired("output")
	return cmd
}

type trainRecipeOpts struct {
	Out, Format, Preset, PromptCol, ResponseCol, ModelName, PythonBin string
	ChunkChars                                                        int
}

func runTrainRecipe(input string, opts trainRecipeOpts) error {
	// Same repo resolution as embed training / doctor — typo in env must not pick a wrong tree.
	root, err := trainingworker.RepoRoot()
	if err != nil {
		return err
	}
	if root == "" {
		return fmt.Errorf("cannot locate repository root containing training.py (set ZEROLLAMA_REPO or OLLAMA_TRAINING_PYTHONPATH)")
	}
	script := filepath.Join(root, "training_recipes.py")
	if st, err := os.Stat(script); err != nil || st.IsDir() {
		return fmt.Errorf("training_recipes.py not found under %s", root)
	}
	py, err := resolveTrainRecipePython(root, opts.PythonBin)
	if err != nil {
		return err
	}

	preset := strings.TrimSpace(opts.Preset)
	if preset == "" {
		preset = "auto"
	}
	args := []string{
		script,
		input,
		"-o", opts.Out,
		"--format", opts.Format,
		"--preset", preset,
		"--prompt-col", opts.PromptCol,
		"--response-col", opts.ResponseCol,
		"--chunk-chars", fmt.Sprintf("%d", opts.ChunkChars),
	}
	if strings.TrimSpace(opts.ModelName) != "" {
		args = append(args, "--model-name", opts.ModelName)
	}

	c := exec.Command(py, args...)
	c.Dir = root
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	c.Env = os.Environ()
	if err := c.Run(); err != nil {
		return fmt.Errorf("train recipe: %w", err)
	}
	return nil
}

// resolveTrainRecipePython prefers explicit → TRAINING_PYTHON → .venv-training → python3.
// WHY .venv-training first: matches 5080 / doctor ABI story for optional pypdf/docx.
func resolveTrainRecipePython(repoRoot, explicit string) (string, error) {
	if p := strings.TrimSpace(explicit); p != "" {
		return p, nil
	}
	if p := strings.TrimSpace(os.Getenv("TRAINING_PYTHON")); p != "" {
		return p, nil
	}
	venvName := ".venv-training"
	if runtime.GOOS == "windows" {
		cand := filepath.Join(repoRoot, venvName, "Scripts", "python.exe")
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
	} else {
		cand := filepath.Join(repoRoot, venvName, "bin", "python")
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
	}
	if _, err := exec.LookPath("python3"); err == nil {
		return "python3", nil
	}
	if _, err := exec.LookPath("python"); err == nil {
		return "python", nil
	}
	return "", fmt.Errorf("no python3/python on PATH (or pass --python / set TRAINING_PYTHON)")
}
