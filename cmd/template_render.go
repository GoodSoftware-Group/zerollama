package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

// NewTemplateCommand exposes Modelfile TEMPLATE helpers for training (T8).
func NewTemplateCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "template",
		Short: "Modelfile TEMPLATE utilities",
	}
	cmd.AddCommand(newTemplateRenderCommand())
	return cmd
}

type templateRenderRequest struct {
	Template string        `json:"template"`
	Messages []api.Message `json:"messages"`
	Train    *bool         `json:"train,omitempty"`
}

func newTemplateRenderCommand() *cobra.Command {
	var (
		templateFile string
		templateStr  string
		train        bool
		messagesJSON string
	)
	cmd := &cobra.Command{
		Use:   "render",
		Short: "Render a Modelfile TEMPLATE (Go text/template) against messages",
		Long: `Render a Modelfile TEMPLATE the same way serve does.

With --train (default for SFT), trailing empty assistant priming is stripped so
completed assistant turns are included without a second open generation header.

Input: --messages JSON array, or a JSON object on stdin:
  {"template":"...", "messages":[{"role":"user","content":"..."}, ...], "train":true}
`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			req := templateRenderRequest{Train: &train}
			if templateStr != "" {
				req.Template = templateStr
			}
			if templateFile != "" {
				b, err := os.ReadFile(templateFile)
				if err != nil {
					return err
				}
				req.Template = string(b)
			}
			if messagesJSON != "" {
				if err := json.Unmarshal([]byte(messagesJSON), &req.Messages); err != nil {
					return fmt.Errorf("messages: %w", err)
				}
			}

			// Stdin JSON fills any missing fields (training.py uses this).
			stat, _ := os.Stdin.Stat()
			if stat != nil && (stat.Mode()&os.ModeCharDevice) == 0 {
				body, err := io.ReadAll(os.Stdin)
				if err != nil {
					return err
				}
				body = []byte(strings.TrimSpace(string(body)))
				if len(body) > 0 {
					var stdinReq templateRenderRequest
					if err := json.Unmarshal(body, &stdinReq); err != nil {
						return fmt.Errorf("stdin json: %w", err)
					}
					if req.Template == "" {
						req.Template = stdinReq.Template
					}
					if len(req.Messages) == 0 {
						req.Messages = stdinReq.Messages
					}
					if stdinReq.Train != nil && !cmd.Flags().Changed("train") {
						train = *stdinReq.Train
					}
				}
			}

			if strings.TrimSpace(req.Template) == "" {
				return fmt.Errorf("template required (--template, --file, or stdin JSON)")
			}
			if len(req.Messages) == 0 {
				return fmt.Errorf("messages required (--messages or stdin JSON)")
			}

			forTrain := train
			if !cmd.Flags().Changed("train") && req.Train != nil {
				forTrain = *req.Train
			}
			// Default true when flag unset and stdin omitted train — SFT-first CLI.
			if !cmd.Flags().Changed("train") && req.Train == nil {
				forTrain = true
			}

			tmpl, err := template.Parse(req.Template)
			if err != nil {
				return err
			}
			var b strings.Builder
			if err := tmpl.Execute(&b, template.Values{
				Messages: req.Messages,
				ForTrain: forTrain,
			}); err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write([]byte(b.String()))
			return err
		},
	}
	cmd.Flags().StringVarP(&templateFile, "file", "f", "", "TEMPLATE file (e.g. template/chatml.gotmpl)")
	cmd.Flags().StringVar(&templateStr, "template", "", "TEMPLATE string")
	cmd.Flags().StringVar(&messagesJSON, "messages", "", "JSON array of {role,content}")
	cmd.Flags().BoolVar(&train, "train", true, "SFT mode: strip trailing generation priming")
	return cmd
}
