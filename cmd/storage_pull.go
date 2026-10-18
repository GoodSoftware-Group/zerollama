package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/storaged"
)

// storageServePull bridges storaged → server.PullModelOpts (same process ModelsDir).
func storageServePull(ctx context.Context, modelName string, opts storaged.PullOptions, fn func(storaged.ProgressEvent)) error {
	return server.PullModelOpts(ctx, modelName, opts.Insecure, opts.Source, func(p api.ProgressResponse) {
		fn(storaged.ProgressEvent{
			Status:    p.Status,
			Digest:    p.Digest,
			Total:     p.Total,
			Completed: p.Completed,
		})
	})
}

func newStoragePullCommand() *cobra.Command {
	var secret, source string
	var insecure bool
	cmd := &cobra.Command{
		Use:   "pull [server-url] <model>",
		Short: "Ask a storage server to pull a model from a registry into its tree",
		Long: `Remote registry ingest on the storage host (POST /v1/pull).

The storage server downloads into its $OLLAMA_MODELS and serves the blobs to
inference clients. Prefer this over pull-then-push when the storage box has
the big disk and the client should not hold the weights.

  zerollama storage pull http://storage:18090 llama3.2:3b
  zerollama storage pull llama3.2:3b   # URL from STORAGE_SERVERS / mDNS

Requires a storage serve built with remote pull (capability.pull=true).`,
		Args: cobra.RangeArgs(1, 2),
		RunE: func(_ *cobra.Command, args []string) error {
			base, modelName, err := storagePullResolveArgs(args)
			if err != nil {
				return err
			}
			auth, err := storageAuth(secret)
			if err != nil {
				return err
			}
			return runStoragePull(context.Background(), auth, base, modelName, source, insecure)
		},
	}
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	cmd.Flags().StringVar(&source, "source", "", "Optional huggingface:// URI (model is the local tag)")
	cmd.Flags().BoolVar(&insecure, "insecure", false, "Allow insecure registry HTTP")
	return cmd
}

func storagePullResolveArgs(args []string) (base, modelName string, err error) {
	if len(args) == 2 {
		if !(strings.HasPrefix(args[0], "http://") || strings.HasPrefix(args[0], "https://")) {
			return "", "", fmt.Errorf("first argument must be a storage URL when two args are given")
		}
		return strings.TrimRight(args[0], "/"), args[1], nil
	}
	modelName = args[0]
	if strings.HasPrefix(modelName, "http://") || strings.HasPrefix(modelName, "https://") {
		return "", "", fmt.Errorf("model name required: zerollama storage pull [url] <model>")
	}
	base, _, err = storageLSResolveArgs(nil)
	if err != nil {
		return "", "", err
	}
	return base, modelName, nil
}

func runStoragePull(ctx context.Context, auth *remotestore.Auth, base, modelName, source string, insecure bool) error {
	reqBody := storaged.PullRequest{
		Model:    modelName,
		Source:   source,
		Insecure: insecure,
	}
	stream := true
	reqBody.Stream = &stream
	raw, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/pull", bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := auth.SignRequest(req, raw); err != nil {
		return err
	}
	client := &http.Client{Timeout: 0} // pulls can take hours
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotImplemented {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s: remote pull not available (upgrade storage serve)\nHint: %s", resp.Status, strings.TrimSpace(string(b)))
	}
	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("%s: check ZEROLLAMA_STORAGE_SECRET\nHint: zerollama storage doctor", resp.Status)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("storage pull: %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()
	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	dec := json.NewDecoder(resp.Body)
	for {
		var ev storaged.ProgressEvent
		if err := dec.Decode(&ev); err == io.EOF {
			break
		} else if err != nil {
			return fmt.Errorf("decode progress: %w", err)
		}
		if ev.Error != "" || ev.Status == "error" {
			msg := ev.Error
			if msg == "" {
				msg = ev.Status
			}
			return fmt.Errorf("storage pull: %s\nHint: check registry name / disk on storage host", msg)
		}
		if ev.Status == "success" {
			continue
		}
		if ev.Digest != "" {
			if ev.Completed == 0 {
				continue
			}
			if spinner != nil {
				spinner.Stop()
			}
			bar, ok := bars[ev.Digest]
			if !ok {
				name, isDigest := strings.CutPrefix(ev.Digest, "sha256:")
				name = strings.TrimSpace(name)
				if isDigest {
					name = name[:min(12, len(name))]
				}
				bar = progress.NewBar(fmt.Sprintf("pulling %s:", name), ev.Total, ev.Completed)
				bars[ev.Digest] = bar
				p.Add(ev.Digest, bar)
			}
			bar.Set(ev.Completed)
			continue
		}
		if status != ev.Status && ev.Status != "" {
			if spinner != nil {
				spinner.Stop()
			}
			status = ev.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}
	}
	fmt.Fprintf(os.Stderr, "pulled %s onto %s\n", modelName, base)
	return nil
}
