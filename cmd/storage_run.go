package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/types/model"
)

func newStorageRunCommand() *cobra.Command {
	var secret string
	var fetchOnly bool
	cmd := &cobra.Command{
		Use:   "run [server-url] <model> [prompt]",
		Short: "Fetch a model from a storage server into local OLLAMA_MODELS, then run it",
		Long: `Client-side ingest from remote storage, then chat like zerollama run.

Unlike "storage pull" (registry → storage host), this copies
manifest + blobs from a storage server onto this machine, then opens
the normal interactive/run path against OLLAMA_HOST (zerollama serve).

  zerollama storage run http://192.168.10.39:18090 qwen3.8:latest
  zerollama storage run qwen3.8:latest   # URL from STORAGE_SERVERS / mDNS
  zerollama storage run http://storage:18090 qwen3.8:latest "hello"

Requires ZEROLLAMA_STORAGE_SECRET (same as the storage host).`,
		Args: cobra.RangeArgs(1, 3),
		RunE: func(cmd *cobra.Command, args []string) error {
			base, modelName, prompts, err := storageRunResolveArgs(args)
			if err != nil {
				return err
			}
			auth, err := storageAuth(secret)
			if err != nil {
				return err
			}
			if err := runStorageFetch(cmd.Context(), auth, base, modelName); err != nil {
				return err
			}
			if fetchOnly {
				fmt.Fprintf(os.Stderr, "fetched %s from %s into %s\n", modelName, base, envconfig.Models())
				return nil
			}
			runArgs := append([]string{modelName}, prompts...)
			return RunHandler(storageRunDelegateCmd(cmd), runArgs)
		},
	}
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	cmd.Flags().BoolVar(&fetchOnly, "fetch-only", false, "Only materialize locally; do not open chat")
	// Flags RunHandler reads (same defaults as `zerollama run`).
	cmd.Flags().String("keepalive", "", "Duration to keep a model loaded (e.g. 5m)")
	cmd.Flags().Bool("verbose", false, "Show timings for response")
	cmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	cmd.Flags().String("format", "", "Response format (e.g. json)")
	cmd.Flags().String("think", "", "Enable thinking mode: true/false or high/medium/low")
	cmd.Flags().Lookup("think").NoOptDefVal = "true"
	cmd.Flags().Bool("hidethinking", false, "Hide thinking output")
	cmd.Flags().Bool("truncate", false, "For embedding models: truncate inputs exceeding context length")
	cmd.Flags().Int("dimensions", 0, "Truncate output embeddings to specified dimension")
	cmd.Flags().Bool("experimental", false, "Enable experimental agent loop with tools")
	cmd.Flags().Bool("experimental-yolo", false, "Skip all tool approval prompts")
	cmd.Flags().Bool("experimental-websearch", false, "Enable web search tool in experimental mode")
	cmd.Flags().Bool("insecure", false, "Unused (kept for RunHandler flag parity)")
	return cmd
}

func storageRunResolveArgs(args []string) (base, modelName string, prompts []string, err error) {
	if len(args) >= 2 && (strings.HasPrefix(args[0], "http://") || strings.HasPrefix(args[0], "https://")) {
		base = strings.TrimRight(args[0], "/")
		modelName = args[1]
		if len(args) >= 3 {
			prompts = args[2:]
		}
		return base, modelName, prompts, nil
	}
	modelName = args[0]
	if strings.HasPrefix(modelName, "http://") || strings.HasPrefix(modelName, "https://") {
		return "", "", nil, fmt.Errorf("model name required: zerollama storage run [url] <model>")
	}
	base, _, err = storageLSResolveArgs(nil)
	if err != nil {
		return "", "", nil, err
	}
	if len(args) >= 2 {
		prompts = args[1:]
	}
	return base, modelName, prompts, nil
}

// storageRunDelegateCmd returns cmd itself when it already has run flags;
// kept as a seam for tests.
func storageRunDelegateCmd(cmd *cobra.Command) *cobra.Command {
	return cmd
}

func runStorageFetch(ctx context.Context, auth *remotestore.Auth, base, modelName string) error {
	n := model.ParseName(modelName) // fills registry.ollama.ai/library + latest
	if n.Model == "" || !n.IsFullyQualified() {
		return fmt.Errorf("invalid model name %q", modelName)
	}
	host, ns, name, tag := n.Host, n.Namespace, n.Model, n.Tag

	r := &remotestore.Resolver{
		Servers:  []string{base},
		Auth:     auth,
		CacheDir: envconfig.Models(),
		MaxBytes: envconfig.RemoteCacheMaxBytes(),
		Mode:     remotestore.CachePersist,
		Chain:    remotestore.PreferRDMAThenTCP(auth),
		HTTP:     &http.Client{Timeout: 30 * time.Minute},
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()
	spin := progress.NewSpinner(fmt.Sprintf("fetching %s from %s", modelName, base))
	p.Add("manifest", spin)

	if _, err := r.FetchManifest(ctx, host, ns, name, tag); err != nil {
		spin.Stop()
		return fmt.Errorf("fetch manifest: %w\nHint: zerollama storage ls %s — check name and ZEROLLAMA_STORAGE_SECRET", err, base)
	}
	spin.Stop()

	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return fmt.Errorf("open local manifest after fetch: %w", err)
	}

	digests := make([]string, 0, 1+len(mf.Layers))
	if mf.Config.Digest != "" {
		digests = append(digests, mf.Config.Digest)
	}
	for _, layer := range mf.Layers {
		if layer.Digest != "" {
			digests = append(digests, layer.Digest)
		}
	}

	for i, digest := range digests {
		short := digest
		if name, ok := strings.CutPrefix(digest, "sha256:"); ok {
			short = name
		}
		if len(short) > 12 {
			short = short[:12]
		}
		layerSpin := progress.NewSpinner(fmt.Sprintf("pulling %s (%d/%d)", short, i+1, len(digests)))
		p.Add(digest, layerSpin)
		if _, err := r.Fetch(ctx, digest); err != nil {
			layerSpin.Stop()
			return fmt.Errorf("fetch %s: %w\nHint: storage doctor; free disk under OLLAMA_MODELS; if malformed HTTP status — restart storage serve on the peer", short, err)
		}
		layerSpin.Stop()
	}
	fmt.Fprintf(os.Stderr, "fetched %s (%d layers) from %s\n", modelName, len(digests), base)
	return nil
}
