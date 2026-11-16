// Pirate Face pull importer: pirateface:// URI → torrent metadata → one GGUF → local manifest.
// Why: searchable catalog + peers for repos removed from Hugging Face; Hub outage still fails.
package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/internal/ssrf"
	typesmodel "github.com/ollama/ollama/types/model"
)

var (
	errNotPirateFaceSource = errors.New("not a pirateface source")
	errPFNoGGUF            = errors.New("no .gguf files found in pirateface listing")
	errPFMultipleGGUF      = errors.New("multiple .gguf files found; specify file in URI")
	errPFSplitGGUF         = errors.New("split GGUF shards are not importable")
	errPFNoSeeders         = errors.New("model is not on Hugging Face and has zero seeders")
	errPFTermsWithheld     = errors.New("magnet withheld until license terms are accepted")
)

const (
	pirateFacePinnedPkg   = "pirateface==0.2.3"
	pirateFaceIdleSeedMin = 10
)

// PirateFaceRef is a parsed pirateface:// source.
type PirateFaceRef struct {
	Repo     string
	Revision string
	File     string
}

type pirateFaceFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type pirateFaceTorrentInfo struct {
	RepoID      string           `json:"repoId"`
	Revision    string           `json:"revision"`
	Seeders     int              `json:"seeders"`
	Leechers    int              `json:"leechers"`
	Files       []pirateFaceFile `json:"files"`
	HF          bool             `json:"hf"`
	Magnet      string           `json:"magnet"`
	Torrent     string           `json:"torrent"`
	Error       string           `json:"error"`
	Terms       any              `json:"terms"`
	TermLinks   any              `json:"termLinks"`
	AcceptTerms any              `json:"acceptTerms"`
	Message     string           `json:"message"`
}

// pirateFaceDownloader downloads one repo file into destDir. Tests inject a stub.
var pirateFaceDownloader = pirateFaceDownloadFile

// pirateFaceSeedOpts configures post-import seeding from the verified GGUF.
type pirateFaceSeedOpts struct {
	Magnet     string
	TorrentURL string
	DataFile   string // absolute path to verified bytes
	RelPath    string // path inside the torrent listing
	Ratio      float64
}

// pirateFaceSeeder seeds completed files until ratio or idle timeout. Tests inject a stub.
var pirateFaceSeeder = pirateFaceSeedUntilRatio

// pirateFaceGET fetches a Pirate Face URL. Tests inject httptest without SSRF.
var pirateFaceGET = pirateFaceHTTPGet

func pirateFaceHTTPGet(ctx context.Context, rawURL string) ([]byte, error) {
	return pirateFaceHTTPGetLimit(ctx, rawURL, 4<<20)
}

func pirateFaceHTTPGetLimit(ctx context.Context, rawURL string, limit int64) ([]byte, error) {
	if err := ssrf.ValidateExternalURL(rawURL); err != nil {
		return nil, fmt.Errorf("pirateface url: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 60 * time.Second, CheckRedirect: ssrf.CheckRedirect}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, limit))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("pirateface %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	return body, nil
}

func pirateFaceAbsoluteURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if strings.HasPrefix(raw, "http://") || strings.HasPrefix(raw, "https://") {
		return raw
	}
	if strings.HasPrefix(raw, "/") {
		return PirateFaceAPIBase() + raw
	}
	return PirateFaceAPIBase() + "/" + raw
}

// IsPirateFacePull reports whether s is a pirateface:// pull target.
func IsPirateFacePull(s string) bool {
	_, err := ParsePirateFaceSource(s)
	return err == nil
}

// ParsePirateFaceSource parses pirateface://owner/repo[@rev][/file.gguf].
func ParsePirateFaceSource(s string) (PirateFaceRef, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return PirateFaceRef{}, errNotPirateFaceSource
	}
	if strings.HasPrefix(s, "pirateface:hf:") {
		s = "pirateface://" + strings.TrimPrefix(s, "pirateface:hf:")
	}
	u, err := url.Parse(s)
	if err != nil {
		return PirateFaceRef{}, err
	}
	if u.Scheme != "pirateface" {
		return PirateFaceRef{}, errNotPirateFaceSource
	}
	var parts []string
	if u.Host != "" {
		parts = append(parts, u.Host)
	}
	if u.Path != "" {
		parts = append(parts, strings.Split(strings.Trim(u.Path, "/"), "/")...)
	}
	if len(parts) == 0 {
		return PirateFaceRef{}, fmt.Errorf("pirateface URI missing repository")
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return PirateFaceRef{}, fmt.Errorf("pirateface URI has invalid path segment %q", part)
		}
	}
	if len(parts) < 2 {
		return PirateFaceRef{}, fmt.Errorf("pirateface URI missing owner/repo")
	}

	ref := PirateFaceRef{}
	if strings.HasSuffix(strings.ToLower(parts[len(parts)-1]), ".gguf") {
		ref.File = parts[len(parts)-1]
		parts = parts[:len(parts)-1]
	}
	if len(parts) < 2 {
		return PirateFaceRef{}, fmt.Errorf("pirateface URI missing owner/repo")
	}
	repo := strings.Join(parts, "/")
	if at := strings.LastIndex(repo, "@"); at > 0 {
		ref.Revision = repo[at+1:]
		repo = repo[:at]
	}
	ref.Repo = repo
	if !strings.Contains(ref.Repo, "/") || strings.HasPrefix(ref.Repo, "/") {
		return PirateFaceRef{}, fmt.Errorf("pirateface URI missing owner/repo")
	}
	return ref, nil
}

func pfLocalName(ref PirateFaceRef) typesmodel.Name {
	return hfLocalName(HFRef{Repo: ref.Repo, File: ref.File, Revision: cmpOr(ref.Revision, defaultHFRevision)})
}

func isSplitGGUF(name string) bool {
	low := strings.ToLower(filepath.Base(name))
	return strings.Contains(low, "-of-") && strings.Contains(low, "-000")
}

func pirateFaceGGUFCandidates(files []pirateFaceFile) []pirateFaceFile {
	var out []pirateFaceFile
	for _, f := range files {
		base := filepath.Base(f.Path)
		low := strings.ToLower(base)
		if !strings.HasSuffix(low, ".gguf") {
			continue
		}
		if strings.HasPrefix(low, "mmproj") {
			continue
		}
		if isSplitGGUF(base) {
			continue
		}
		out = append(out, f)
	}
	return out
}

func pirateFacePickGGUF(files []pirateFaceFile, want string) (pirateFaceFile, error) {
	if want != "" {
		want = strings.TrimPrefix(want, "/")
		for _, f := range files {
			if f.Path == want || filepath.Base(f.Path) == filepath.Base(want) {
				if isSplitGGUF(f.Path) {
					return pirateFaceFile{}, errPFSplitGGUF
				}
				return f, nil
			}
		}
		return pirateFaceFile{}, fmt.Errorf("pirateface file %q not in listing", want)
	}
	cands := pirateFaceGGUFCandidates(files)
	switch len(cands) {
	case 0:
		return pirateFaceFile{}, errPFNoGGUF
	case 1:
		return cands[0], nil
	default:
		return pirateFaceFile{}, errPFMultipleGGUF
	}
}

func pirateFaceTermsWithheld(info pirateFaceTorrentInfo) bool {
	if strings.TrimSpace(info.Magnet) != "" {
		return false
	}
	if info.Terms != nil || info.TermLinks != nil || info.AcceptTerms != nil {
		return true
	}
	msg := strings.ToLower(info.Message + " " + info.Error)
	return strings.Contains(msg, "accept") && strings.Contains(msg, "term")
}

func fetchPirateFaceTorrent(ctx context.Context, repo string) (pirateFaceTorrentInfo, error) {
	u := PirateFaceAPIBase() + "/api/torrents?repo=" + url.QueryEscape(repo)
	body, err := pirateFaceGET(ctx, u)
	if err != nil {
		return pirateFaceTorrentInfo{}, err
	}
	var info pirateFaceTorrentInfo
	if err := json.Unmarshal(body, &info); err != nil {
		return pirateFaceTorrentInfo{}, fmt.Errorf("pirateface torrents decode: %w", err)
	}
	if info.Error != "" && info.RepoID == "" {
		return info, fmt.Errorf("pirateface: %s", info.Error)
	}
	if info.RepoID == "" {
		info.RepoID = repo
	}
	return info, nil
}

func pirateFaceSeedRatio() float64 {
	raw := strings.TrimSpace(os.Getenv("ZEROLLAMA_PIRATEFACE_SEED_RATIO"))
	if raw == "" {
		return 1.0
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v < 0 {
		return 1.0
	}
	return v
}

func fileSHA256Hex(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func pirateFaceDownloadFile(ctx context.Context, repo, revision, file, destDir string, fn func(api.ProgressResponse)) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	// Why PF_NO_RESCUE: Hub resolves first; Pirate Face peers only when Hub 404s / missing.
	env := append(os.Environ(), "PF_NO_RESCUE=1")
	fn(api.ProgressResponse{Status: "preparing downloader"})

	// Single-file path: huggingface_hub's `hf download` with the pirateface hook installed
	// as an extra. `--from pirateface` alone does not provide the `hf` CLI.
	if bin := strings.TrimSpace(os.Getenv("ZEROLLAMA_PF_BIN")); bin != "" {
		base := strings.ToLower(filepath.Base(bin))
		var args []string
		switch base {
		case "pf", "pf.exe":
			// Stock pf CLI has no single-file flag; download the repo then import file.
			args = []string{"download", repo, "--local-dir", destDir}
		default:
			// hf-compatible wrapper: download REPO FILE --local-dir DIR
			args = []string{"download", repo, file, "--local-dir", destDir}
		}
		if revision != "" {
			args = append(args, "--revision", revision)
		}
		cmd := exec.CommandContext(ctx, bin, args...)
		cmd.Env = env
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("%s download: %w", bin, err)
		}
		return nil
	}

	uvx, err := exec.LookPath("uvx")
	if err != nil {
		return fmt.Errorf("pirateface pull needs uv (uvx on PATH), or set ZEROLLAMA_PF_BIN to an hf-compatible download wrapper: install uv from https://docs.astral.sh/uv/")
	}
	args := pirateFaceUVXDownloadArgs(repo, revision, file, destDir)
	cmd := exec.CommandContext(ctx, uvx, args...)
	cmd.Env = env
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func pirateFaceUVXDownloadArgs(repo, revision, file, destDir string) []string {
	args := []string{
		"-p", "3.13",
		"--from", "huggingface_hub",
		"--with", pirateFacePinnedPkg,
		"hf", "download", repo, file,
		"--local-dir", destDir,
	}
	if revision != "" {
		args = append(args, "--revision", revision)
	}
	return args
}

// bencodeStringAfterKey reads a bencode string value immediately after key (e.g. "name").
func bencodeStringAfterKey(data []byte, key string) (string, bool) {
	marker := []byte(fmt.Sprintf("%d:%s", len(key), key))
	i := bytes.Index(data, marker)
	if i < 0 {
		return "", false
	}
	rest := data[i+len(marker):]
	j := 0
	for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
		j++
	}
	if j == 0 || j >= len(rest) || rest[j] != ':' {
		return "", false
	}
	n, err := strconv.Atoi(string(rest[:j]))
	if err != nil || n < 0 || j+1+n > len(rest) {
		return "", false
	}
	return string(rest[j+1 : j+1+n]), true
}

// bencodeTorrentFilePaths returns multi-file torrent path lists (joined with /).
func bencodeTorrentFilePaths(data []byte) []string {
	var paths []string
	marker := []byte("4:path")
	for i := 0; ; {
		at := bytes.Index(data[i:], marker)
		if at < 0 {
			break
		}
		at += i
		rest := data[at+len(marker):]
		i = at + len(marker)
		if len(rest) == 0 || rest[0] != 'l' {
			continue
		}
		rest = rest[1:]
		var parts []string
		ok := true
		for len(rest) > 0 && rest[0] != 'e' {
			j := 0
			for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
				j++
			}
			if j == 0 || j >= len(rest) || rest[j] != ':' {
				ok = false
				break
			}
			n, err := strconv.Atoi(string(rest[:j]))
			if err != nil || n < 0 || j+1+n > len(rest) {
				ok = false
				break
			}
			parts = append(parts, string(rest[j+1:j+1+n]))
			rest = rest[j+1+n:]
		}
		if ok && len(parts) > 0 {
			paths = append(paths, strings.Join(parts, "/"))
		}
	}
	return paths
}

func pirateFaceTorrentSelectIndex(paths []string, rel string) int {
	rel = strings.TrimPrefix(filepath.ToSlash(rel), "/")
	base := filepath.Base(rel)
	for i, p := range paths {
		p = filepath.ToSlash(p)
		if p == rel || filepath.Base(p) == base {
			return i + 1 // aria2c --select-file is 1-based
		}
	}
	return 0
}

func stagePirateFaceBlobCopy(src, digest string) error {
	dst, err := manifest.BlobsPath(digest)
	if err != nil {
		return err
	}
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	// Why copy: createLink prefers symlink; tmpDir is removed after import.
	if err := copyFile(src, dst); err != nil {
		return fmt.Errorf("stage pirateface blob: %w", err)
	}
	return nil
}

func pirateFacePrepareSeedLayout(seedRoot, torrentName, relPath, dataFile string) (string, error) {
	relPath = strings.TrimPrefix(filepath.ToSlash(relPath), "/")
	if torrentName == "" {
		return "", fmt.Errorf("torrent missing name")
	}
	// Multi-file torrents: dir/<name>/<rel>. Single-file: dir/<name> is the file itself.
	var target string
	if relPath == "" || relPath == torrentName || filepath.Base(relPath) == torrentName {
		target = filepath.Join(seedRoot, torrentName)
	} else {
		target = filepath.Join(seedRoot, torrentName, filepath.FromSlash(relPath))
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return "", err
	}
	_ = os.Remove(target)
	if err := os.Link(dataFile, target); err != nil {
		if err := copyFile(dataFile, target); err != nil {
			return "", err
		}
	}
	return target, nil
}

func pirateFaceSeedUntilRatio(ctx context.Context, opts pirateFaceSeedOpts, fn func(api.ProgressResponse)) error {
	if opts.Ratio <= 0 {
		return nil
	}
	aria2c, err := exec.LookPath("aria2c")
	if err != nil {
		fn(api.ProgressResponse{Status: "seeding skipped: install aria2c to meet ratio 1.0"})
		return nil
	}
	if strings.TrimSpace(opts.DataFile) == "" {
		fn(api.ProgressResponse{Status: "seeding skipped: no verified file"})
		return nil
	}

	seedRoot, err := os.MkdirTemp("", "zerollama-pf-seed-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(seedRoot) }()

	var torrentPath string
	var torrentName string
	var selectIdx int
	if u := pirateFaceAbsoluteURL(opts.TorrentURL); u != "" {
		// Torrents carry piece hashes; allow up to 16 MiB.
		body, err := pirateFaceHTTPGetLimit(ctx, u, 16<<20)
		if err != nil {
			// Fall back to magnet-only with best-effort layout.
			slog.Warn("pirateface torrent fetch failed; seeding via magnet", "error", err)
		} else if !bytes.Contains(body, []byte("4:info")) {
			slog.Warn("pirateface torrent response was not a .torrent")
		} else {
			torrentName, _ = bencodeStringAfterKey(body, "name")
			paths := bencodeTorrentFilePaths(body)
			selectIdx = pirateFaceTorrentSelectIndex(paths, opts.RelPath)
			torrentPath = filepath.Join(seedRoot, "model.torrent")
			if err := os.WriteFile(torrentPath, body, 0o644); err != nil {
				return err
			}
		}
	}
	if torrentName == "" {
		// Best-effort when only magnet is available: HF repo leaf as torrent name.
		torrentName = filepath.Base(filepath.Dir(opts.DataFile))
		if torrentName == "." || torrentName == "" || strings.HasPrefix(torrentName, "zerollama-pf-") || strings.HasPrefix(torrentName, "seed-") {
			torrentName = strings.TrimSuffix(filepath.Base(opts.RelPath), filepath.Ext(opts.RelPath))
		}
	}
	if _, err := pirateFacePrepareSeedLayout(seedRoot, torrentName, opts.RelPath, opts.DataFile); err != nil {
		return err
	}

	fn(api.ProgressResponse{Status: fmt.Sprintf("seeding 0.0 of %.1f (peers can see this IP until seeding stops)", opts.Ratio)})
	args := []string{
		"--seed-ratio=" + strconv.FormatFloat(opts.Ratio, 'f', -1, 64),
		// Cap wall time so quiet swarms do not hang the pull forever (aria2 stops on ratio OR time).
		"--seed-time=" + strconv.Itoa(pirateFaceIdleSeedMin),
		"--bt-enable-lpd=true",
		"--enable-dht=true",
		"--check-integrity=true",
		"--allow-overwrite=true",
		"--dir=" + seedRoot,
		"--console-log-level=notice",
		"--summary-interval=5",
	}
	if selectIdx > 0 {
		args = append(args, "--select-file="+strconv.Itoa(selectIdx))
	}
	if torrentPath != "" {
		args = append(args, torrentPath)
	} else if strings.TrimSpace(opts.Magnet) != "" {
		args = append(args, opts.Magnet)
	} else {
		fn(api.ProgressResponse{Status: "seeding skipped: no magnet or torrent"})
		return nil
	}

	cmd := exec.CommandContext(ctx, aria2c, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Non-zero is common when seed-time expires before ratio; import already succeeded.
		fn(api.ProgressResponse{Status: "seeding stopped (ratio reached or time limit)"})
		return nil
	}
	fn(api.ProgressResponse{Status: fmt.Sprintf("seeded to ratio %.1f", opts.Ratio)})
	return nil
}

func formatPFMultiple(ref PirateFaceRef, files []pirateFaceFile) error {
	cands := pirateFaceGGUFCandidates(files)
	var b strings.Builder
	b.WriteString("multiple .gguf files found; pick one:\n")
	for _, f := range cands {
		fmt.Fprintf(&b, "  zerollama pull pirateface://%s/%s\n", ref.Repo, f.Path)
	}
	return errors.New(strings.TrimSpace(b.String()))
}

func pullFromPirateFace(ctx context.Context, local typesmodel.Name, ref PirateFaceRef, deleteMap map[string]struct{}, fn func(api.ProgressResponse)) error {
	fn(api.ProgressResponse{Status: fmt.Sprintf("contacting pirateface %s", ref.Repo)})

	info, err := fetchPirateFaceTorrent(ctx, ref.Repo)
	if err != nil {
		return err
	}
	if pirateFaceTermsWithheld(info) {
		return fmt.Errorf("%w: open the model page on pirateface.co and accept the license terms", errPFTermsWithheld)
	}
	if !info.HF && info.Seeders == 0 {
		return errPFNoSeeders
	}

	file, err := pirateFacePickGGUF(info.Files, ref.File)
	if err != nil {
		if errors.Is(err, errPFMultipleGGUF) {
			return formatPFMultiple(ref, info.Files)
		}
		return err
	}
	ref.File = file.Path

	if info.HF {
		fn(api.ProgressResponse{Status: fmt.Sprintf("%d seeders (peers can see this IP until seeding stops)", info.Seeders)})
	} else {
		fn(api.ProgressResponse{Status: fmt.Sprintf("not on Hugging Face; %d seeders (peers can see this IP until seeding stops)", info.Seeders)})
	}

	tmpDir, err := os.MkdirTemp("", "zerollama-pf-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	rev := cmpOr(ref.Revision, info.Revision)
	if err := pirateFaceDownloader(ctx, ref.Repo, rev, file.Path, tmpDir, fn); err != nil {
		return fmt.Errorf("pirateface download: %w", err)
	}

	dest := filepath.Join(tmpDir, filepath.FromSlash(file.Path))
	if _, err := os.Stat(dest); err != nil {
		// hf download may flatten to basename.
		alt := filepath.Join(tmpDir, filepath.Base(file.Path))
		if _, err2 := os.Stat(alt); err2 == nil {
			dest = alt
		} else {
			return fmt.Errorf("pirateface download missing %s", file.Path)
		}
	}

	got, err := fileSHA256Hex(dest)
	if err != nil {
		return err
	}
	want := strings.ToLower(strings.TrimSpace(file.SHA256))
	if want == "" || got != want {
		return fmt.Errorf("pirateface sha256 mismatch for %s: got %s want %s", file.Path, got, want)
	}

	digest := "sha256:" + got
	fn(api.ProgressResponse{Status: fmt.Sprintf("importing %s", filepath.Base(file.Path)), Digest: digest, Total: 1, Completed: 1})

	if err := stagePirateFaceBlobCopy(dest, digest); err != nil {
		return err
	}

	config := &typesmodel.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
	}
	r := api.CreateRequest{}
	relFiles := map[string]string{filepath.Base(file.Path): digest}
	baseLayers, err := convertModelFromFiles(relFiles, nil, false, fn)
	if err != nil {
		return err
	}
	if err := createModel(r, local, baseLayers, config, fn); err != nil {
		return err
	}

	if !envconfig.NoPrune() && len(deleteMap) > 0 {
		fn(api.ProgressResponse{Status: "removing unused layers"})
		if err := deleteUnusedLayers(deleteMap); err != nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("couldn't remove unused layers: %v", err)})
		}
	}

	ratio := pirateFaceSeedRatio()
	if ratio > 0 {
		// Seed from the durable blob so tmp cleanup cannot race aria2c.
		blobPath, err := manifest.BlobsPath(digest)
		if err != nil {
			return err
		}
		if err := pirateFaceSeeder(ctx, pirateFaceSeedOpts{
			Magnet:     info.Magnet,
			TorrentURL: info.Torrent,
			DataFile:   blobPath,
			RelPath:    file.Path,
			Ratio:      ratio,
		}, fn); err != nil {
			slog.Warn("pirateface seeding ended with error", "error", err)
			fn(api.ProgressResponse{Status: fmt.Sprintf("seeding ended: %v", err)})
		}
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
}

func tryPullPirateFace(ctx context.Context, pullName string, localName typesmodel.Name, regOpts *registryOptions, deleteMap map[string]struct{}, fn func(api.ProgressResponse)) (bool, error) {
	var ref PirateFaceRef
	var local typesmodel.Name
	var err error

	switch {
	case regOpts != nil && IsPirateFacePull(strings.TrimSpace(regOpts.HFSource)):
		ref, err = ParsePirateFaceSource(regOpts.HFSource)
		if err != nil {
			return false, err
		}
		local = localName
		if !local.IsValid() {
			return false, fmt.Errorf("invalid local model name %q", pullName)
		}
	case IsPirateFacePull(pullName):
		ref, err = ParsePirateFaceSource(pullName)
		if err != nil {
			return false, err
		}
		local = pfLocalName(ref)
	default:
		return false, nil
	}

	slog.Info("pulling from pirateface", "repo", ref.Repo, "file", ref.File, "local", local.DisplayShortest())
	if err := pullFromPirateFace(ctx, local, ref, deleteMap, fn); err != nil {
		return true, err
	}
	return true, nil
}
