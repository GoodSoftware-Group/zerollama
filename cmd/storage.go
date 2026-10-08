package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/storaged"
)

// NewStorageCommand registers `zerollama storage serve|push|ls|verify|gc|sync`.
// Why a subcommand of zerollama (not a second binary): one package for operators,
// shared auth/env with the inference daemon, same OLLAMA_MODELS layout.
func NewStorageCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "storage",
		Short: "Remote model storage (central blob/manifest server and migration)",
		Long: `Serve or migrate content-addressed model blobs over a trusted LAN.

Why: keep one canonical $OLLAMA_MODELS tree on a big disk; inference nodes
fetch on miss. Lab listen defaults to :18090 — never bind :11434/:8081.

See docs/remote-model-storage.md.`,
	}
	cmd.AddCommand(newStorageServeCommand())
	cmd.AddCommand(newStoragePushCommand())
	cmd.AddCommand(newStorageLSCommand())
	cmd.AddCommand(newStorageVerifyCommand())
	cmd.AddCommand(newStorageGCCommand())
	cmd.AddCommand(newStorageSyncCommand())
	return cmd
}

func newStorageServeCommand() *cobra.Command {
	var listen, models, secret, tlsCert, tlsKey string
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Serve local OLLAMA_MODELS over the remote storage protocol",
		RunE: func(_ *cobra.Command, _ []string) error {
			if models == "" {
				models = envconfig.Models()
			}
			if secret == "" {
				secret = envconfig.StorageSecret()
			}
			auth, err := remotestore.NewAuth(secret)
			if err != nil {
				return fmt.Errorf("storage secret required (ZEROLLAMA_STORAGE_SECRET): %w", err)
			}
			if listen == "" {
				listen = envconfig.StorageListen()
			}
			if tlsCert == "" {
				tlsCert = envconfig.StorageTLSCert()
			}
			if tlsKey == "" {
				tlsKey = envconfig.StorageTLSKey()
			}
			if (tlsCert == "") != (tlsKey == "") {
				return fmt.Errorf("both --tls-cert and --tls-key (or ZEROLLAMA_STORAGE_TLS_CERT/KEY) are required together")
			}
			srv := storaged.New(models, auth)
			srv.Peers = envconfig.ParseStoragePeers()
			httpSrv := &http.Server{
				Addr:              listen,
				Handler:           srv,
				ReadHeaderTimeout: 30 * time.Second,
			}
			ln, err := net.Listen("tcp", listen)
			if err != nil {
				return err
			}
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			srv.StartReplicator(ctx)
			slog.Info("zerollama storage serve", "listen", ln.Addr().String(), "models", models, "tls", tlsCert != "", "peers", len(srv.Peers))

			go func() {
				<-ctx.Done()
				shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				_ = httpSrv.Shutdown(shutdownCtx)
			}()
			if tlsCert != "" {
				err = httpSrv.ServeTLS(ln, tlsCert, tlsKey)
			} else {
				err = httpSrv.Serve(ln)
			}
			if err == http.ErrServerClosed {
				return nil
			}
			return err
		},
	}
	cmd.Flags().StringVar(&listen, "listen", "", "Listen address (default ZEROLLAMA_STORAGE_LISTEN or 0.0.0.0:18090)")
	cmd.Flags().StringVar(&models, "models", "", "Models root (default OLLAMA_MODELS)")
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret (default ZEROLLAMA_STORAGE_SECRET)")
	cmd.Flags().StringVar(&tlsCert, "tls-cert", "", "TLS certificate file (optional)")
	cmd.Flags().StringVar(&tlsKey, "tls-key", "", "TLS private key file (optional)")
	return cmd
}

func newStoragePushCommand() *cobra.Command {
	var reclaim bool
	var secret string
	cmd := &cobra.Command{
		Use:   "push <server-url>",
		Short: "Upload local manifests and referenced blobs to a storage server",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			base := strings.TrimRight(args[0], "/")
			if secret == "" {
				secret = envconfig.StorageSecret()
			}
			auth, err := remotestore.NewAuth(secret)
			if err != nil {
				return err
			}
			return runStoragePush(context.Background(), auth, base, envconfig.Models(), reclaim)
		},
	}
	cmd.Flags().BoolVar(&reclaim, "reclaim", false, "Delete local blobs only after all referencing manifests pushed successfully (safe for shared layers)")
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	return cmd
}

func newStorageLSCommand() *cobra.Command {
	var secret string
	var kind string
	cmd := &cobra.Command{
		Use:   "ls <server-url>",
		Short: "List blobs or manifests on a storage server",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			base := strings.TrimRight(args[0], "/")
			auth, err := storageAuth(secret)
			if err != nil {
				return err
			}
			path := "/v1/blobs"
			if kind == "manifests" {
				path = "/v1/manifests"
			}
			client := &http.Client{Timeout: 2 * time.Minute}
			cursor := ""
			for {
				url := base + path
				if cursor != "" {
					url += "?cursor=" + cursor
				}
				var resp struct {
					Items      []string `json:"items"`
					NextCursor string   `json:"next_cursor"`
				}
				if err := storageJSON(context.Background(), auth, client, http.MethodGet, url, nil, &resp); err != nil {
					return err
				}
				for _, item := range resp.Items {
					fmt.Println(item)
				}
				if resp.NextCursor == "" {
					break
				}
				cursor = resp.NextCursor
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	cmd.Flags().StringVar(&kind, "kind", "blobs", "blobs or manifests")
	return cmd
}

func newStorageVerifyCommand() *cobra.Command {
	var secret string
	cmd := &cobra.Command{
		Use:   "verify <server-url> [digest...]",
		Short: "Re-hash blobs on a storage server and report mismatches",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			base := strings.TrimRight(args[0], "/")
			auth, err := storageAuth(secret)
			if err != nil {
				return err
			}
			body := map[string]any{}
			if len(args) > 1 {
				body["digests"] = args[1:]
			}
			var resp struct {
				Results []struct {
					Digest string `json:"digest"`
					OK     bool   `json:"ok"`
					Error  string `json:"error"`
				} `json:"results"`
			}
			if err := storageJSON(context.Background(), auth, &http.Client{Timeout: 60 * time.Minute}, http.MethodPost, base+"/v1/verify", body, &resp); err != nil {
				return err
			}
			var bad int
			for _, r := range resp.Results {
				if r.OK {
					fmt.Printf("OK  %s\n", r.Digest)
				} else {
					bad++
					fmt.Printf("BAD %s %s\n", r.Digest, r.Error)
				}
			}
			if bad > 0 {
				return fmt.Errorf("%d digest mismatches", bad)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	return cmd
}

func newStorageGCCommand() *cobra.Command {
	var secret string
	var apply bool
	cmd := &cobra.Command{
		Use:   "gc <server-url>",
		Short: "List (or delete) blobs not referenced by any remote manifest",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			base := strings.TrimRight(args[0], "/")
			auth, err := storageAuth(secret)
			if err != nil {
				return err
			}
			body := map[string]any{"apply": apply}
			var resp struct {
				Orphans []string `json:"orphans"`
				Deleted []string `json:"deleted"`
				DryRun  bool     `json:"dry_run"`
			}
			if err := storageJSON(context.Background(), auth, &http.Client{Timeout: 5 * time.Minute}, http.MethodPost, base+"/v1/gc", body, &resp); err != nil {
				return err
			}
			if resp.DryRun {
				fmt.Printf("dry-run: %d orphan blobs\n", len(resp.Orphans))
			} else {
				fmt.Printf("deleted %d of %d orphans\n", len(resp.Deleted), len(resp.Orphans))
			}
			for _, d := range resp.Orphans {
				fmt.Println(d)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	cmd.Flags().BoolVar(&apply, "apply", false, "Delete orphans (default is dry-run)")
	return cmd
}

func newStorageSyncCommand() *cobra.Command {
	var secret, models string
	cmd := &cobra.Command{
		Use:   "sync <peer-url>",
		Short: "Push all local blobs and manifests to a peer storage server",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			peer := strings.TrimRight(args[0], "/")
			if secret == "" {
				secret = envconfig.StorageSecret()
			}
			if models == "" {
				models = envconfig.Models()
			}
			auth, err := remotestore.NewAuth(secret)
			if err != nil {
				return err
			}
			srv := storaged.New(models, auth)
			return srv.SyncToPeer(context.Background(), peer)
		},
	}
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	cmd.Flags().StringVar(&models, "models", "", "Models root (default OLLAMA_MODELS)")
	return cmd
}

func storageAuth(secret string) (*remotestore.Auth, error) {
	if secret == "" {
		secret = envconfig.StorageSecret()
	}
	return remotestore.NewAuth(secret)
}

func storageJSON(ctx context.Context, auth *remotestore.Auth, client *http.Client, method, url string, body any, out any) error {
	var rdr io.Reader
	var raw []byte
	if body != nil {
		var err error
		raw, err = json.Marshal(body)
		if err != nil {
			return err
		}
		rdr = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return err
	}
	if raw != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if err := auth.SignRequest(req, raw); err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("%s: %s: %s", url, resp.Status, strings.TrimSpace(string(b)))
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(b, out)
}

func runStoragePush(ctx context.Context, auth *remotestore.Auth, base, modelsDir string, reclaim bool) error {
	// Why two-phase reclaim: blobs are shared across manifests. Deleting after
	// the first upload breaks later models that still need the local file.
	// Collect reference counts, push everything, then delete only digests whose
	// full reference set was successfully pushed.
	manifestsRoot := filepath.Join(modelsDir, "manifests")
	blobsRoot := filepath.Join(modelsDir, "blobs")
	client := &http.Client{Timeout: 60 * time.Minute}

	// blobRefs tracks how many manifests reference each blob path (for safe reclaim).
	// blobOK tracks digests that were confirmed uploaded or already present.
	blobRefs := map[string]int{} // blobPath → total reference count across all manifests
	blobOK := map[string]bool{}  // digest → pushed/confirmed-remote

	var uploaded, skipped, failed int

	// Phase 1: walk every manifest, push blobs, push manifest.
	// Reclaim-eligible blobs are collected but NOT deleted yet.
	type manifestEntry struct {
		path string
		rel  string
	}
	var manifests []manifestEntry

	err := filepath.WalkDir(manifestsRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(manifestsRoot, path)
		if err != nil {
			return err
		}
		if len(strings.Split(filepath.ToSlash(rel), "/")) != 4 {
			return nil
		}
		manifests = append(manifests, manifestEntry{path: path, rel: rel})
		return nil
	})
	if err != nil {
		return err
	}

	// Collect per-blob reference counts so we know when it's safe to delete.
	for _, m := range manifests {
		mfBytes, err := os.ReadFile(m.path)
		if err != nil {
			continue
		}
		for _, dig := range extractDigests(string(mfBytes)) {
			bp := blobLocalPath(blobsRoot, dig)
			blobRefs[bp]++
		}
	}

	// Track how many manifests have been successfully pushed that reference each blob.
	blobPushedBy := map[string]int{}

	for _, m := range manifests {
		parts := strings.Split(filepath.ToSlash(m.rel), "/")
		host, ns, model, tag := parts[0], parts[1], parts[2], parts[3]
		mfBytes, err := os.ReadFile(m.path)
		if err != nil {
			failed++
			continue
		}
		digests := extractDigests(string(mfBytes))

		// Push blobs first.
		manifestOK := true
		for _, dig := range digests {
			bp := blobLocalPath(blobsRoot, dig)
			if _, err := os.Stat(bp); err != nil {
				// Blob already missing locally; skip (may be on remote already).
				continue
			}
			if blobOK[dig] {
				// Already pushed in a previous manifest iteration.
				continue
			}
			tcp := remotestore.NewTCPTransport(auth)
			tcp.Client = client
			_, ok, herr := tcp.HeadBlob(ctx, base, dig)
			if herr == nil && ok {
				skipped++
				blobOK[dig] = true
			} else {
				if err := remotestore.PushBlob(ctx, auth, client, base, dig, bp); err != nil {
					slog.Error("push blob failed", "digest", dig, "error", err)
					failed++
					manifestOK = false
					continue
				}
				uploaded++
				slog.Info("pushed blob", "digest", dig)
				blobOK[dig] = true
			}
		}

		if !manifestOK {
			failed++
			continue
		}

		if err := remotestore.PushManifest(ctx, auth, client, base, host, ns, model, tag, mfBytes); err != nil {
			slog.Error("push manifest failed", "path", m.rel, "error", err)
			failed++
			continue
		}
		slog.Info("pushed manifest", "path", m.rel)

		// Record that this manifest (and its blobs) were successfully handled.
		for _, dig := range digests {
			bp := blobLocalPath(blobsRoot, dig)
			blobPushedBy[bp]++
		}
	}

	// Phase 2: reclaim only blobs whose every referencing manifest was pushed successfully.
	if reclaim {
		for bp, total := range blobRefs {
			if blobPushedBy[bp] >= total {
				if err := os.Remove(bp); err != nil && !os.IsNotExist(err) {
					slog.Warn("reclaim remove failed", "path", bp, "error", err)
				} else {
					slog.Info("reclaimed blob", "path", bp)
				}
			} else {
				slog.Info("skipping reclaim: not all manifests pushed", "path", bp,
					"pushed", blobPushedBy[bp], "total", total)
			}
		}
	}

	fmt.Printf("storage push complete: uploaded=%d skipped=%d failed=%d\n", uploaded, skipped, failed)
	if failed > 0 {
		return fmt.Errorf("%d failures during push", failed)
	}
	return nil
}

func blobLocalPath(blobsRoot, dig string) string {
	name := strings.ReplaceAll(dig, ":", "-")
	if !strings.HasPrefix(name, "sha256-") {
		name = "sha256-" + strings.TrimPrefix(name, "sha256-")
	}
	return filepath.Join(blobsRoot, name)
}

func extractDigests(s string) []string {
	var out []string
	seen := map[string]struct{}{}
	const needle = "sha256:"
	for {
		i := strings.Index(s, needle)
		if i < 0 {
			break
		}
		rest := s[i:]
		if len(rest) < 7+64 {
			break
		}
		d := rest[:7+64]
		ok := true
		for _, c := range d[7:] {
			if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
				ok = false
				break
			}
		}
		if ok {
			if _, exists := seen[d]; !exists {
				seen[d] = struct{}{}
				out = append(out, d)
			}
		}
		s = rest[7:]
	}
	return out
}
