package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/weightedit"
	_ "github.com/ollama/ollama/x/weightedit/gpatch"
)

// NewWeightsCommand registers `zerollama weights edit inspect|apply|restore`.
func NewWeightsCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "weights",
		Short: "Inspect and apply stored weight edits",
	}
	edit := &cobra.Command{
		Use:   "edit",
		Short: "Work with weight-edit patch files",
	}
	edit.AddCommand(newWeightsInspectCmd(), newWeightsApplyCmd(), newWeightsRestoreCmd())
	cmd.AddCommand(edit)
	return cmd
}

func newWeightsInspectCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "inspect PATCH",
		Short: "Decode a weight-edit patch and print a summary",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			edit, codec, err := decodePatchFile(args[0])
			if err != nil {
				return err
			}
			out := map[string]any{
				"format":          edit.Format,
				"codec":           codec.Format(),
				"name":            edit.Name,
				"id":              edit.ID,
				"changed_weights": edit.ChangedWeights(),
			}
			tensors := make([]map[string]any, 0, len(edit.Changes))
			for _, ch := range edit.Changes {
				changed := len(ch.Indices)
				if changed == 0 {
					changed = int(ch.Numel())
				}
				tensors = append(tensors, map[string]any{
					"name":         ch.Name,
					"dtype":        ch.Dtype,
					"shape":        ch.Shape,
					"changed":      changed,
					"touched_rows": ch.TouchedRows(),
				})
			}
			out["tensors"] = tensors
			enc := json.NewEncoder(cmd.OutOrStdout())
			enc.SetIndent("", "  ")
			return enc.Encode(out)
		},
	}
}

func newWeightsApplyCmd() *cobra.Command {
	var modelName, destination, outDir string
	cmd := &cobra.Command{
		Use:   "apply PATCH",
		Short: "Apply a weight-edit patch to a linked safetensors model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runWeightsWrite(cmd, args[0], modelName, destination, outDir, "source")
		},
	}
	cmd.Flags().StringVar(&modelName, "model", "", "source model tag (linked safetensors)")
	cmd.Flags().StringVar(&destination, "destination", "", "new model tag")
	cmd.Flags().StringVar(&outDir, "out", "", "output checkpoint directory (default: sibling of source_dir)")
	_ = cmd.MarkFlagRequired("model")
	_ = cmd.MarkFlagRequired("destination")
	return cmd
}

func newWeightsRestoreCmd() *cobra.Command {
	var modelName, destination, outDir string
	cmd := &cobra.Command{
		Use:   "restore PATCH",
		Short: "Restore a patched linked model back to the patch's source checkpoint",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runWeightsWrite(cmd, args[0], modelName, destination, outDir, "result")
		},
	}
	cmd.Flags().StringVar(&modelName, "model", "", "patched model tag")
	cmd.Flags().StringVar(&destination, "destination", "", "restored model tag")
	cmd.Flags().StringVar(&outDir, "out", "", "output checkpoint directory (default: sibling of source_dir)")
	_ = cmd.MarkFlagRequired("model")
	_ = cmd.MarkFlagRequired("destination")
	return cmd
}

func decodePatchFile(path string) (*weightedit.Edit, weightedit.Codec, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	return weightedit.Decode(f)
}

func runWeightsWrite(cmd *cobra.Command, patchPath, modelName, destination, outDir, side string) error {
	edit, codec, err := decodePatchFile(patchPath)
	if err != nil {
		return err
	}
	m, err := server.GetModel(modelName)
	if err != nil {
		return err
	}
	if m.Config.ModelFormat != "safetensors" || strings.TrimSpace(m.Config.SourceDir) == "" {
		return fmt.Errorf("model %q is not a linked safetensors checkpoint (need source_dir)", modelName)
	}
	srcDir := m.Config.SourceDir
	dst := model.ParseName(destination)
	if !dst.IsValid() {
		return fmt.Errorf("destination %q is invalid", destination)
	}
	if _, err := manifest.ParseNamedManifest(dst); err == nil {
		return fmt.Errorf("destination %q already exists", destination)
	}
	if outDir == "" {
		outDir = siblingDir(srcDir, dst)
	}
	store, err := weightedit.OpenDirectory(srcDir)
	if err != nil {
		return err
	}
	bound, err := weightedit.Bind(store, edit, side)
	if err != nil {
		return err
	}
	res, err := weightedit.Apply(bound, outDir)
	if err != nil {
		return err
	}
	if err := registerCLILinkedTag(m, dst, res.Out); err != nil {
		_ = os.RemoveAll(res.Out)
		return err
	}
	fmt.Fprintf(cmd.OutOrStdout(), "wrote %s (%s id=%s changed=%d) -> %s\n",
		res.Out, codec.Format(), shortID(edit.ID), res.ChangedWeights, dst.DisplayShortest())
	return nil
}

func siblingDir(srcDir string, dst model.Name) string {
	base := dst.DisplayShortest()
	var b strings.Builder
	for _, r := range base {
		switch {
		case r == '/' || r == ':' || r == ' ':
			b.WriteByte('-')
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	name := b.String()
	if name == "" {
		name = "edited"
	}
	return filepath.Join(filepath.Dir(srcDir), name)
}

func registerCLILinkedTag(src *server.Model, dst model.Name, outDir string) error {
	cfg := src.Config
	abs, err := filepath.Abs(outDir)
	if err != nil {
		return err
	}
	cfg.SourceDir = abs
	if cfg.FileType == "" {
		cfg.FileType = "linked"
	}
	cfg.ModelFormat = "safetensors"
	raw, err := json.Marshal(cfg)
	if err != nil {
		return err
	}
	configLayer, err := manifest.NewLayer(bytes.NewReader(raw), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return err
	}
	mf, err := manifest.ParseNamedManifest(model.ParseName(src.Name))
	if err != nil {
		return err
	}
	return manifest.WriteManifest(dst, configLayer, append([]manifest.Layer(nil), mf.Layers...))
}

func shortID(id string) string {
	if len(id) > 12 {
		return id[:12]
	}
	return id
}
