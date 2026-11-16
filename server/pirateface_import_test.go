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
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	typesmodel "github.com/ollama/ollama/types/model"
)

func TestParsePirateFaceSource(t *testing.T) {
	ref, err := ParsePirateFaceSource("pirateface://Divij/Qwen2.5-0.5B-Instruct-Q4_K_M/qwen2.5-0.5b-instruct-q4_k_m.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if ref.Repo != "Divij/Qwen2.5-0.5B-Instruct-Q4_K_M" || ref.File != "qwen2.5-0.5b-instruct-q4_k_m.gguf" {
		t.Fatalf("ref=%+v", ref)
	}
}

func TestParsePirateFaceSourceRevision(t *testing.T) {
	ref, err := ParsePirateFaceSource("pirateface://owner/repo@abc123/file.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if ref.Repo != "owner/repo" || ref.Revision != "abc123" || ref.File != "file.gguf" {
		t.Fatalf("ref=%+v", ref)
	}
}

func TestParsePirateFaceSourceRejectsDotDot(t *testing.T) {
	if _, err := ParsePirateFaceSource("pirateface://owner/../etc/passwd"); err == nil {
		t.Fatal("expected error")
	}
}

func TestLooksLikePirateFaceGGUF(t *testing.T) {
	cases := map[string]bool{
		"tensorblock/Qwen2.5-0.5B-Instruct-GGUF":   true,
		"Divij/Qwen2.5-0.5B-Instruct-Q4_K_M":       true,
		"owner/model-IQ4_XS":                       true,
		"mlx-community/Qwen2.5-0.5B-Instruct-4bit": false,
		"Qwen/Qwen2.5-0.5B-Instruct-AWQ":           false,
	}
	for repo, want := range cases {
		if got := LooksLikePirateFaceGGUF(repo); got != want {
			t.Fatalf("%s: got %v want %v", repo, got, want)
		}
	}
}

func TestParsePirateFaceSearchLine(t *testing.T) {
	line := "tensorblock/Qwen2.5-0.5B-Instruct-GGUF                         694 MB   8 seeds  magnet:?xt=urn:btih:abc"
	row, ok := ParsePirateFaceSearchLine(line)
	if !ok {
		t.Fatal("parse failed")
	}
	if row.Repo != "tensorblock/Qwen2.5-0.5B-Instruct-GGUF" || row.Size != "694 MB" || row.Seeds != 8 {
		t.Fatalf("row=%+v", row)
	}
	if !strings.HasPrefix(row.Magnet, "magnet:") {
		t.Fatalf("magnet=%q", row.Magnet)
	}
}

func TestFilterPirateFaceSearch(t *testing.T) {
	rows := []PirateFaceSearchRow{
		{Repo: "a/finetune", Size: "1 GB", Seeds: 10},
		{Repo: "b/model-GGUF", Size: "2 GB", Seeds: 5},
		{Repo: "c/model-Q4_K_M", Size: "400 MB", Seeds: 3},
		{Repo: "d/model-GGUF", Size: "1 GB", Seeds: 0},
		{Repo: "e/model-Q5_K_S", Size: "500 MB", Seeds: 0},
	}
	shown, hidden, zeroOnly := FilterPirateFaceSearch(rows, false)
	if zeroOnly {
		t.Fatal("expected seeded rows")
	}
	if len(shown) != 2 {
		t.Fatalf("shown=%v", shown)
	}
	if shown[0].Repo != "b/model-GGUF" || shown[1].Repo != "c/model-Q4_K_M" {
		t.Fatalf("shown=%v", shown)
	}
	if hidden < 1 {
		t.Fatalf("hidden=%d", hidden)
	}

	onlyEmpty := []PirateFaceSearchRow{
		{Repo: "d/model-GGUF", Size: "1 GB", Seeds: 0},
		{Repo: "e/model-Q5_K_S", Size: "500 MB", Seeds: 0},
	}
	shown, _, zeroOnly = FilterPirateFaceSearch(onlyEmpty, false)
	if !zeroOnly || len(shown) != 2 {
		t.Fatalf("zeroOnly=%v shown=%v", zeroOnly, shown)
	}
}

func TestFilterPirateFaceSearchCap(t *testing.T) {
	var rows []PirateFaceSearchRow
	for i := 0; i < 30; i++ {
		rows = append(rows, PirateFaceSearchRow{
			Repo:  fmt.Sprintf("owner/model-GGUF-%d", i),
			Size:  "1 GB",
			Seeds: 1 + i,
		})
	}
	shown, hidden, _ := FilterPirateFaceSearch(rows, false)
	if len(shown) != pirateFaceSearchCap {
		t.Fatalf("len=%d", len(shown))
	}
	if hidden != 10 {
		t.Fatalf("hidden=%d", hidden)
	}
}

func TestPirateFacePickGGUF(t *testing.T) {
	files := []pirateFaceFile{
		{Path: "README.md", SHA256: "a"},
		{Path: "a.Q4_K_M.gguf", SHA256: "b"},
		{Path: "a.Q5_K_M.gguf", SHA256: "c"},
		{Path: "mmproj.gguf", SHA256: "d"},
	}
	_, err := pirateFacePickGGUF(files, "")
	if !errors.Is(err, errPFMultipleGGUF) {
		t.Fatalf("err=%v", err)
	}
	got, err := pirateFacePickGGUF(files, "a.Q4_K_M.gguf")
	if err != nil || got.Path != "a.Q4_K_M.gguf" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestPirateFacePickGGUFSplit(t *testing.T) {
	files := []pirateFaceFile{{Path: "model-00001-of-00002.gguf", SHA256: "a"}}
	_, err := pirateFacePickGGUF(files, "model-00001-of-00002.gguf")
	if !errors.Is(err, errPFSplitGGUF) {
		t.Fatalf("err=%v", err)
	}
}

func withPirateFaceTestGET(t *testing.T, srv *httptest.Server) {
	t.Helper()
	t.Setenv("PF_API", srv.URL)
	old := pirateFaceGET
	pirateFaceGET = func(ctx context.Context, rawURL string) ([]byte, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
		if err != nil {
			return nil, err
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		return io.ReadAll(resp.Body)
	}
	t.Cleanup(func() { pirateFaceGET = old })
}

func TestPullPirateFaceHFFalseZeroSeeders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(pirateFaceTorrentInfo{
			RepoID:  "owner/gone-GGUF",
			HF:      false,
			Seeders: 0,
			Files:   []pirateFaceFile{{Path: "m.gguf", SHA256: strings.Repeat("ab", 32)}},
		})
	}))
	defer srv.Close()
	withPirateFaceTestGET(t, srv)

	err := pullFromPirateFace(context.Background(), typesmodel.ParseName("gone:latest"), PirateFaceRef{Repo: "owner/gone-GGUF"}, nil, func(api.ProgressResponse) {})
	if !errors.Is(err, errPFNoSeeders) {
		t.Fatalf("err=%v", err)
	}
}

func TestPullPirateFaceTermsWithheld(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"repoId":    "owner/terms-GGUF",
			"hf":        true,
			"seeders":   2,
			"magnet":    "",
			"termLinks": []string{"https://example.com/terms"},
			"files":     []pirateFaceFile{{Path: "m.gguf", SHA256: strings.Repeat("cd", 32)}},
		})
	}))
	defer srv.Close()
	withPirateFaceTestGET(t, srv)

	err := pullFromPirateFace(context.Background(), typesmodel.ParseName("terms:latest"), PirateFaceRef{Repo: "owner/terms-GGUF"}, nil, func(api.ProgressResponse) {})
	if !errors.Is(err, errPFTermsWithheld) {
		t.Fatalf("err=%v", err)
	}
}

func TestPullPirateFaceMultipleFilesListsChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(pirateFaceTorrentInfo{
			RepoID:  "owner/multi-GGUF",
			HF:      true,
			Seeders: 2,
			Magnet:  "magnet:?xt=urn:btih:deadbeef",
			Files: []pirateFaceFile{
				{Path: "a.Q4_K_M.gguf", SHA256: strings.Repeat("11", 32)},
				{Path: "a.Q5_K_M.gguf", SHA256: strings.Repeat("22", 32)},
			},
		})
	}))
	defer srv.Close()
	withPirateFaceTestGET(t, srv)

	err := pullFromPirateFace(context.Background(), typesmodel.ParseName("multi:latest"), PirateFaceRef{Repo: "owner/multi-GGUF"}, nil, func(api.ProgressResponse) {})
	if err == nil || !strings.Contains(err.Error(), "pirateface://owner/multi-GGUF/a.Q4_K_M.gguf") {
		t.Fatalf("err=%v", err)
	}
}

func TestPullPirateFaceStagesVerifiedFile(t *testing.T) {
	payload := []byte("pirateface-fixture-gguf")
	sum := sha256.Sum256(payload)
	hexSum := hex.EncodeToString(sum[:])

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(pirateFaceTorrentInfo{
			RepoID:   "owner/one-GGUF",
			Revision: "rev1",
			HF:       true,
			Seeders:  3,
			Magnet:   "magnet:?xt=urn:btih:cafebabe",
			Torrent:  "/api/torrents/cafebabe.torrent",
			Files:    []pirateFaceFile{{Path: "one.gguf", SHA256: hexSum}},
		})
	}))
	defer srv.Close()
	withPirateFaceTestGET(t, srv)
	t.Setenv("ZEROLLAMA_PIRATEFACE_SEED_RATIO", "0")

	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)

	oldDL := pirateFaceDownloader
	oldSeed := pirateFaceSeeder
	defer func() {
		pirateFaceDownloader = oldDL
		pirateFaceSeeder = oldSeed
	}()
	pirateFaceDownloader = func(ctx context.Context, repo, revision, file, destDir string, fn func(api.ProgressResponse)) error {
		path := filepath.Join(destDir, file)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		return os.WriteFile(path, payload, 0o644)
	}
	pirateFaceSeeder = func(ctx context.Context, opts pirateFaceSeedOpts, fn func(api.ProgressResponse)) error {
		t.Fatal("seeder should not run when ratio is 0")
		return nil
	}

	local := typesmodel.ParseName("one:latest")
	err := pullFromPirateFace(context.Background(), local, PirateFaceRef{Repo: "owner/one-GGUF", File: "one.gguf"}, nil, func(api.ProgressResponse) {})

	blob := filepath.Join(models, "blobs", "sha256-"+hexSum)
	st, statErr := os.Lstat(blob)
	if statErr != nil {
		t.Fatalf("durable blob missing after stage: %v (pull err=%v)", statErr, err)
	}
	if st.Mode()&os.ModeSymlink != 0 {
		t.Fatal("blob must be a real file, not a symlink into a deleted temp dir")
	}
	got, readErr := os.ReadFile(blob)
	if readErr != nil || !bytes.Equal(got, payload) {
		t.Fatalf("blob bytes mismatch: readErr=%v", readErr)
	}

	if err != nil {
		// createModel needs a real GGUF; staging+convert may fail on fake bytes.
		if strings.Contains(err.Error(), "pirateface sha256") || strings.Contains(err.Error(), "pirateface download") {
			t.Fatalf("unexpected: %v", err)
		}
		t.Logf("createModel on fixture bytes (expected for non-GGUF payload): %v", err)
	}
}

func TestBencodeTorrentNameAndPaths(t *testing.T) {
	// Minimal multi-file info dict fragment (not a full torrent).
	raw := []byte("d4:infod4:name28:qwen2.5-0.5b-instruct-q4-k-m5:filesld6:lengthi1e4:pathl7:LICENSEeed6:lengthi2e4:pathl33:qwen2.5-0.5b-instruct-q4_k_m.ggufeeed6:lengthi3e4:pathl9:README.mdeeee")
	name, ok := bencodeStringAfterKey(raw, "name")
	if !ok || name != "qwen2.5-0.5b-instruct-q4-k-m" {
		t.Fatalf("name=%q ok=%v", name, ok)
	}
	paths := bencodeTorrentFilePaths(raw)
	if len(paths) != 3 || paths[1] != "qwen2.5-0.5b-instruct-q4_k_m.gguf" {
		t.Fatalf("paths=%v", paths)
	}
	if idx := pirateFaceTorrentSelectIndex(paths, "qwen2.5-0.5b-instruct-q4_k_m.gguf"); idx != 2 {
		t.Fatalf("idx=%d", idx)
	}
}

func TestPirateFacePrepareSeedLayout(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "one.gguf")
	if err := os.WriteFile(src, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	seed := filepath.Join(dir, "seed")
	target, err := pirateFacePrepareSeedLayout(seed, "torrent-name", "one.gguf", src)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(seed, "torrent-name", "one.gguf")
	if target != want {
		t.Fatalf("target=%s want=%s", target, want)
	}
	got, err := os.ReadFile(target)
	if err != nil || string(got) != "gguf" {
		t.Fatalf("seed file: %v %q", err, got)
	}
}

func TestPirateFaceAbsoluteURL(t *testing.T) {
	t.Setenv("PF_API", "https://pirateface.co")
	if got := pirateFaceAbsoluteURL("/api/torrents/x.torrent"); got != "https://pirateface.co/api/torrents/x.torrent" {
		t.Fatalf("got=%s", got)
	}
	if got := pirateFaceAbsoluteURL("https://example.com/a.torrent"); got != "https://example.com/a.torrent" {
		t.Fatalf("got=%s", got)
	}
}

func TestPirateFaceDownloadUVXArgs(t *testing.T) {
	args := pirateFaceUVXDownloadArgs("owner/repo", "rev1", "a.gguf", "/tmp/out")
	want := []string{
		"-p", "3.13",
		"--from", "huggingface_hub",
		"--with", "pirateface==0.2.3",
		"hf", "download", "owner/repo", "a.gguf",
		"--local-dir", "/tmp/out",
		"--revision", "rev1",
	}
	if len(args) != len(want) {
		t.Fatalf("args=%v", args)
	}
	for i := range want {
		if args[i] != want[i] {
			t.Fatalf("args[%d]=%q want %q (full=%v)", i, args[i], want[i], args)
		}
	}
}
