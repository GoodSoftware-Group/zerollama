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
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fleet/mdns"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/server/remotestore"
	"github.com/ollama/ollama/server/remotestore/storaged"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

// NewStorageCommand registers `zerollama storage serve|push|pull|ls|verify|gc|sync`.
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
	cmd.AddCommand(newStoragePullCommand())
	cmd.AddCommand(newStorageLSCommand())
	cmd.AddCommand(newStorageVerifyCommand())
	cmd.AddCommand(newStorageGCCommand())
	cmd.AddCommand(newStorageSyncCommand())
	cmd.AddCommand(newStorageDiscoverCommand())
	cmd.AddCommand(newStorageDoctorCommand())
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
			// Own OLLAMA_MODELS for this process so registry pulls land in ModelsDir.
			if err := os.Setenv("OLLAMA_MODELS", models); err != nil {
				return fmt.Errorf("set OLLAMA_MODELS: %w", err)
			}
			srv := storaged.New(models, auth)
			srv.Peers = envconfig.ParseStoragePeers()
			srv.Pull = storageServePull
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
			startStorageMDNS(ctx, ln)

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
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "ls [server-url] [filter]",
		Short: "List models on a storage server (table like zerollama ls)",
		Long: `List tagged models on remote storage in a human table (NAME, ID, SIZE, LAYERS, MODIFIED).

Default kind is models. For ops dumps:
  zerollama storage ls --kind blobs http://host:18090
  zerollama storage ls --kind manifests http://host:18090

Server URL is optional when ZEROLLAMA_STORAGE_SERVERS is set, or when mDNS
finds a peer (ZEROLLAMA_MDNS=1 on the storage host).`,
		Args: cobra.MaximumNArgs(2),
		RunE: func(_ *cobra.Command, args []string) error {
			base, filter, err := storageLSResolveArgs(args)
			if err != nil {
				return err
			}
			auth, err := storageAuth(secret)
			if err != nil {
				return err
			}
			client := &http.Client{Timeout: 2 * time.Minute}
			ctx := context.Background()
			switch strings.ToLower(kind) {
			case "blobs", "manifests":
				return storageLSRaw(ctx, auth, client, base, kind, asJSON)
			case "models", "":
				return storageLSModels(ctx, auth, client, base, filter, asJSON)
			default:
				return fmt.Errorf("unknown --kind %q (models|blobs|manifests)", kind)
			}
		},
	}
	cmd.Flags().StringVar(&secret, "secret", "", "Shared HMAC secret")
	cmd.Flags().StringVar(&kind, "kind", "models", "models (default table), blobs, or manifests")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Print JSON instead of a table")
	return cmd
}

func storageLSResolveArgs(args []string) (base, filter string, err error) {
	if len(args) >= 1 && (strings.HasPrefix(args[0], "http://") || strings.HasPrefix(args[0], "https://")) {
		base = strings.TrimRight(args[0], "/")
		if len(args) >= 2 {
			filter = args[1]
		}
		return base, filter, nil
	}
	if len(args) >= 1 {
		filter = args[0]
	}
	if servers := remotestore.StorageServers(); len(servers) > 0 {
		return servers[0], filter, nil
	}
	found := remotestore.BrowseStorage(context.Background(), envconfig.StorageMDNSBrowseWait())
	if len(found) == 0 {
		return "", "", fmt.Errorf("storage server URL required (or set ZEROLLAMA_STORAGE_SERVERS / enable mDNS)\nHint: zerollama storage discover")
	}
	if len(found) > 1 {
		fmt.Fprintf(os.Stderr, "using first of %d discovered storage peers: %s\n", len(found), found[0])
	}
	return found[0], filter, nil
}

func storageLSRaw(ctx context.Context, auth *remotestore.Auth, client *http.Client, base, kind string, asJSON bool) error {
	path := "/v1/blobs"
	if kind == "manifests" {
		path = "/v1/manifests"
	}
	var all []string
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
		if err := storageJSON(ctx, auth, client, http.MethodGet, url, nil, &resp); err != nil {
			return err
		}
		all = append(all, resp.Items...)
		if resp.NextCursor == "" {
			break
		}
		cursor = resp.NextCursor
	}
	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(map[string]any{"kind": kind, "items": all})
	}
	for _, item := range all {
		fmt.Println(item)
	}
	if len(all) == 0 {
		fmt.Fprintf(os.Stderr, "no %s on %s\n", kind, base)
	}
	return nil
}

func storageLSModels(ctx context.Context, auth *remotestore.Auth, client *http.Client, base, filter string, asJSON bool) error {
	var models []storaged.ModelListItem
	cursor := ""
	for {
		u, err := url.Parse(base + "/v1/models")
		if err != nil {
			return err
		}
		q := u.Query()
		if cursor != "" {
			q.Set("cursor", cursor)
		}
		if filter != "" {
			q.Set("q", filter)
		}
		u.RawQuery = q.Encode()
		var page storaged.ModelsListResponse
		if err := storageJSON(ctx, auth, client, http.MethodGet, u.String(), nil, &page); err != nil {
			// Older storage servers: fall back to manifests + per-manifest GET.
			if cursor == "" && strings.Contains(err.Error(), "404") {
				return storageLSModelsLegacy(ctx, auth, client, base, filter, asJSON)
			}
			return fmt.Errorf("list models: %w\nHint: upgrade storage serve, or --kind manifests", err)
		}
		models = append(models, page.Models...)
		if page.NextCursor == "" {
			break
		}
		cursor = page.NextCursor
	}
	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(map[string]any{"models": models})
	}
	if len(models) == 0 {
		fmt.Fprintf(os.Stderr, "no models on %s\n", base)
		fmt.Fprintln(os.Stderr, "hint: zerollama storage push "+base+"   # migrate local tags")
		return nil
	}
	rows := make([]listTableRow, 0, len(models))
	for _, m := range models {
		digest := m.Digest
		if len(digest) > 12 {
			if i := strings.Index(digest, ":"); i >= 0 && i+1 < len(digest) {
				digest = digest[i+1:]
			}
			if len(digest) > 12 {
				digest = digest[:12]
			}
		}
		layers := "--"
		if m.Layers > 0 {
			layers = fmt.Sprintf("%d", m.Layers)
		}
		rows = append(rows, listTableRow{
			name:     m.Name,
			id:       digest,
			size:     format.HumanBytes(m.Size),
			params:   layers, // reuse PARAMS column as LAYERS for storage
			ctx:      "--",
			perf:     "--",
			modified: format.HumanTime(m.ModifiedAt, "Never"),
		})
	}
	// Storage-specific header: LAYERS instead of PARAMS/CTX/PERF.
	printStorageListTable(os.Stdout, rows, terminalWidth())
	return nil
}

func printStorageListTable(w io.Writer, rows []listTableRow, width int) {
	table := newPlainTable(w)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "LAYERS", "MODIFIED"})
	for _, r := range rows {
		table.Append([]string{r.name, r.id, r.size, r.params, r.modified})
	}
	_ = width
	table.Render()
}

func storageLSModelsLegacy(ctx context.Context, auth *remotestore.Auth, client *http.Client, base, filter string, asJSON bool) error {
	// Pre-/v1/models server: list manifest paths then GET each.
	var paths []string
	cursor := ""
	for {
		url := base + "/v1/manifests"
		if cursor != "" {
			url += "?cursor=" + cursor
		}
		var resp struct {
			Items      []string `json:"items"`
			NextCursor string   `json:"next_cursor"`
		}
		if err := storageJSON(ctx, auth, client, http.MethodGet, url, nil, &resp); err != nil {
			return err
		}
		paths = append(paths, resp.Items...)
		if resp.NextCursor == "" {
			break
		}
		cursor = resp.NextCursor
	}
	models := make([]storaged.ModelListItem, 0, len(paths))
	for _, rel := range paths {
		var mf struct {
			Config struct {
				Digest string `json:"digest"`
				Size   int64  `json:"size"`
			} `json:"config"`
			Layers []struct {
				Digest string `json:"digest"`
				Size   int64  `json:"size"`
			} `json:"layers"`
		}
		if err := storageJSON(ctx, auth, client, http.MethodGet, base+"/v1/manifest/"+rel, nil, &mf); err != nil {
			continue
		}
		var size int64
		size += mf.Config.Size
		for _, layer := range mf.Layers {
			size += layer.Size
		}
		name := rel
		// best-effort display name from path
		parts := strings.Split(rel, "/")
		if len(parts) == 4 {
			n := model.Name{Host: parts[0], Namespace: parts[1], Model: parts[2], Tag: parts[3]}
			if n.IsValid() {
				name = n.DisplayShortest()
			}
		}
		if filter != "" && !strings.Contains(strings.ToLower(name), strings.ToLower(filter)) {
			continue
		}
		digest := mf.Config.Digest
		models = append(models, storaged.ModelListItem{
			Name:   name,
			Path:   rel,
			Digest: digest,
			Size:   size,
			Layers: len(mf.Layers),
		})
	}
	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(map[string]any{"models": models})
	}
	rows := make([]listTableRow, 0, len(models))
	for _, m := range models {
		digest := m.Digest
		if i := strings.Index(digest, ":"); i >= 0 && i+1 < len(digest) {
			digest = digest[i+1:]
		}
		if len(digest) > 12 {
			digest = digest[:12]
		}
		layers := "--"
		if m.Layers > 0 {
			layers = fmt.Sprintf("%d", m.Layers)
		}
		rows = append(rows, listTableRow{
			name: m.Name, id: digest, size: format.HumanBytes(m.Size),
			params: layers, modified: "—",
		})
	}
	printStorageListTable(os.Stdout, rows, terminalWidth())
	return nil
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

// startStorageMDNS advertises _zerollama-storage._tcp when ZEROLLAMA_STORAGE_MDNS
// (or inherited ZEROLLAMA_MDNS) is on.
func startStorageMDNS(ctx context.Context, ln net.Listener) {
	if !envconfig.StorageMDNSAdvertise() || ln == nil {
		return
	}
	port, err := mdns.PortFromAddr(ln.Addr())
	if err != nil || port <= 0 {
		slog.Warn("storage mdns not started", "error", err)
		return
	}
	srv, err := mdns.Register(mdns.RegisterOpts{
		Service: mdns.ServiceStorage,
		Port:    port,
		TXT: map[string]string{
			"role":    "storage",
			"version": version.Version,
		},
	})
	if err != nil {
		slog.Warn("storage mdns registration failed", "error", err)
		return
	}
	go func() {
		<-ctx.Done()
		srv.Shutdown()
		slog.Info("storage mdns unregistered", "service", mdns.ServiceStorage)
	}()
}

func newStorageDiscoverCommand() *cobra.Command {
	var wait time.Duration
	cmd := &cobra.Command{
		Use:   "discover",
		Short: "Browse LAN for zerollama storage peers (mDNS)",
		Long: `Browse _zerollama-storage._tcp for a short window and print base URLs.

On the storage host: ZEROLLAMA_MDNS=1 zerollama storage serve
On this host:        zerollama storage discover
Then:                export ZEROLLAMA_STORAGE_SERVERS=<url>
                     export ZEROLLAMA_STORAGE_SECRET=…  # same as storage host`,
		RunE: func(_ *cobra.Command, _ []string) error {
			if wait <= 0 {
				wait = envconfig.StorageMDNSBrowseWait()
			}
			ctx, cancel := context.WithTimeout(context.Background(), wait+time.Second)
			defer cancel()
			found := remotestore.BrowseStorage(ctx, wait)
			if len(found) == 0 {
				fmt.Println("no storage peers found")
				fmt.Println("hint: ZEROLLAMA_MDNS=1 zerollama storage serve on the model host")
				fmt.Println("hint: same Wi‑Fi/LAN; firewall allow UDP 5353 + TCP 18090")
				return nil
			}
			for _, u := range found {
				fmt.Println(u)
			}
			fmt.Fprintf(os.Stderr, "hint: export ZEROLLAMA_STORAGE_SERVERS=%s\n", strings.Join(found, ","))
			if envconfig.StorageSecret() == "" {
				fmt.Fprintln(os.Stderr, "hint: export ZEROLLAMA_STORAGE_SECRET=… (must match storage host)")
			}
			return nil
		},
	}
	cmd.Flags().DurationVar(&wait, "wait", 0, "Browse window (default ZEROLLAMA_STORAGE_MDNS_WAIT or 2s)")
	return cmd
}

func newStorageDoctorCommand() *cobra.Command {
	var wait time.Duration
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check storage secret, configured/discovered servers, and HMAC reachability",
		RunE: func(_ *cobra.Command, _ []string) error {
			return runStorageDoctor(wait)
		},
	}
	cmd.Flags().DurationVar(&wait, "wait", 0, "mDNS browse window when STORAGE_SERVERS empty")
	return cmd
}

func runStorageDoctor(wait time.Duration) error {
	failed := 0
	secret := envconfig.StorageSecret()
	if secret == "" {
		fmt.Println("FAIL  storage secret unset")
		fmt.Println("  fix: export ZEROLLAMA_STORAGE_SECRET='long-random-shared-secret'")
		failed++
	} else {
		fmt.Println("OK    storage secret set")
	}

	servers := remotestore.StorageServers()
	if len(servers) == 0 {
		fmt.Println("WARN  ZEROLLAMA_STORAGE_SERVERS empty — browsing mDNS…")
		if wait <= 0 {
			wait = envconfig.StorageMDNSBrowseWait()
		}
		ctx, cancel := context.WithTimeout(context.Background(), wait+time.Second)
		defer cancel()
		servers = remotestore.BrowseStorage(ctx, wait)
		if len(servers) == 0 {
			fmt.Println("FAIL  no storage servers configured or discovered")
			fmt.Println("  fix: export ZEROLLAMA_STORAGE_SERVERS=http://host:18090")
			fmt.Println("  or:  ZEROLLAMA_MDNS=1 on storage host + zerollama storage discover")
			failed++
		} else {
			fmt.Printf("OK    discovered %d server(s) via mDNS\n", len(servers))
		}
	} else {
		fmt.Printf("OK    STORAGE_SERVERS (%d)\n", len(servers))
	}

	if secret == "" || len(servers) == 0 {
		if failed > 0 {
			return fmt.Errorf("storage doctor: %d check(s) failed", failed)
		}
		return nil
	}

	auth, err := remotestore.NewAuth(secret)
	if err != nil {
		fmt.Printf("FAIL  auth: %v\n", err)
		return err
	}
	client := &http.Client{Timeout: 10 * time.Second}
	ctx := context.Background()
	for _, base := range servers {
		cap, err := remotestore.GetCapability(ctx, auth, client, base)
		if err != nil {
			fmt.Printf("FAIL  %s capability: %v\n", base, err)
			fmt.Println("  fix: check firewall TCP :18090, matching secret, storage serve up")
			failed++
			continue
		}
		verbs := "tcp"
		if cap.RDMA != nil && cap.RDMA.Verbs {
			verbs = "tcp+rdma"
		} else if len(cap.Transports) > 0 {
			verbs = strings.Join(cap.Transports, "+")
		}
		pull := "pull=no"
		if cap.Pull {
			pull = "pull=yes"
		}
		fmt.Printf("OK    %s (%s, %s)\n", base, verbs, pull)

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/v1/health", nil)
		if err != nil {
			fmt.Printf("FAIL  %s health build: %v\n", base, err)
			failed++
			continue
		}
		if err := auth.SignRequest(req, nil); err != nil {
			fmt.Printf("FAIL  %s health sign: %v\n", base, err)
			failed++
			continue
		}
		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("FAIL  %s health: %v\n", base, err)
			failed++
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		_ = resp.Body.Close()
		if resp.StatusCode >= 300 {
			fmt.Printf("FAIL  %s health HTTP %d: %s\n", base, resp.StatusCode, strings.TrimSpace(string(body)))
			failed++
			continue
		}
		fmt.Printf("OK    %s health\n", base)
	}

	if failed > 0 {
		return fmt.Errorf("storage doctor: %d check(s) failed", failed)
	}
	fmt.Println("PASS  storage doctor")
	return nil
}
