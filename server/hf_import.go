// Hugging Face pull importer (LA8): huggingface:// URI → local manifest without gallery.
// Why: operators often have GGUF on HF but not on registry.ollama.ai; LocalAI-compatible URI.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	typesmodel "github.com/ollama/ollama/types/model"
)

var (
	errNotHuggingFaceSource = errors.New("not a huggingface source")
	errHFNoGGUF             = errors.New("no .gguf files found in huggingface repository")
	errHFMultipleGGUF       = errors.New("multiple .gguf files found; specify file in URI")
)

const defaultHFRevision = "main"

// HFRef is a parsed huggingface:// source.
type HFRef struct {
	Repo     string
	Revision string
	File     string
}

// IsHFPull reports whether s is a huggingface:// or hf:// pull target.
func IsHFPull(s string) bool {
	_, err := ParseHFSource(s)
	return err == nil
}

// ParseHFSource parses huggingface://org/repo[/file.gguf] or hf://… or https://huggingface.co/….
func ParseHFSource(s string) (HFRef, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return HFRef{}, errNotHuggingFaceSource
	}

	if strings.HasPrefix(s, "hf://") {
		s = "huggingface://" + strings.TrimPrefix(s, "hf://")
	}

	if strings.HasPrefix(s, "https://huggingface.co/") || strings.HasPrefix(s, "http://huggingface.co/") {
		u, err := url.Parse(s)
		if err != nil {
			return HFRef{}, err
		}
		p := strings.TrimPrefix(u.Path, "/")
		if strings.Contains(p, "/resolve/") {
			parts := strings.SplitN(p, "/resolve/", 2)
			repo := parts[0]
			rest := parts[1]
			if i := strings.Index(rest, "/"); i >= 0 {
				rev := rest[:i]
				file := rest[i+1:]
				return HFRef{Repo: repo, Revision: rev, File: file}, nil
			}
		}
		return parseHFPath(p)
	}

	u, err := url.Parse(s)
	if err != nil {
		return HFRef{}, err
	}
	if u.Scheme != "huggingface" {
		return HFRef{}, errNotHuggingFaceSource
	}

	p := u.Host
	if u.Path != "" {
		if p != "" {
			p = path.Join(p, strings.TrimPrefix(u.Path, "/"))
		} else {
			p = strings.TrimPrefix(u.Path, "/")
		}
	}
	if p == "" {
		return HFRef{}, fmt.Errorf("huggingface URI missing repository")
	}
	return parseHFPath(p)
}

func parseHFPath(p string) (HFRef, error) {
	p = strings.Trim(p, "/")
	parts := strings.Split(p, "/")
	if len(parts) < 2 {
		return HFRef{}, fmt.Errorf("huggingface URI missing org/repo")
	}

	ref := HFRef{Revision: defaultHFRevision}
	if strings.HasSuffix(strings.ToLower(parts[len(parts)-1]), ".gguf") {
		ref.File = parts[len(parts)-1]
		parts = parts[:len(parts)-1]
	}
	repo := strings.Join(parts, "/")
	if at := strings.LastIndex(repo, "@"); at > 0 {
		ref.Revision = repo[at+1:]
		repo = repo[:at]
	}
	ref.Repo = repo
	if ref.Repo == "" {
		return HFRef{}, fmt.Errorf("huggingface URI missing org/repo")
	}
	return ref, nil
}

func hfLocalName(ref HFRef) typesmodel.Name {
	base := path.Base(ref.Repo)
	modelPart := sanitizeHFModelPart(base)
	tag := "latest"
	if ref.File != "" {
		stem := strings.TrimSuffix(path.Base(ref.File), ".gguf")
		if i := strings.LastIndex(stem, "."); i >= 0 && i < len(stem)-1 {
			tag = sanitizeHFModelPart(stem[i+1:])
		}
	}
	return typesmodel.ParseName(modelPart + ":" + tag)
}

func sanitizeHFModelPart(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	for i, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else if r == '-' || r == '_' || r == '.' {
			if i > 0 && b.Len() > 0 {
				b.WriteByte('-')
			}
		}
	}
	out := b.String()
	if out == "" {
		return "hf-model"
	}
	return out
}

type hfTreeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
}

func (r HFRef) treeURL() string {
	rev := cmpOr(r.Revision, defaultHFRevision)
	return fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s?recursive=true", r.Repo, rev)
}

func (r HFRef) resolveURL(file string) string {
	rev := cmpOr(r.Revision, defaultHFRevision)
	return fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s", r.Repo, rev, file)
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func hfHTTPClient() *http.Client {
	return &http.Client{Timeout: 0}
}

func hfAuthHeader(req *http.Request) {
	token := strings.TrimSpace(os.Getenv("HF_TOKEN"))
	if token == "" {
		token = strings.TrimSpace(os.Getenv("HUGGING_FACE_HUB_TOKEN"))
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

func hfListGGUFFiles(ctx context.Context, ref HFRef) ([]hfTreeEntry, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ref.treeURL(), nil)
	if err != nil {
		return nil, err
	}
	hfAuthHeader(req)
	resp, err := hfHTTPClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("huggingface api %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var entries []hfTreeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, err
	}
	var gguf []hfTreeEntry
	for _, e := range entries {
		if e.Type != "file" {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			continue
		}
		gguf = append(gguf, e)
	}
	return gguf, nil
}

func hfPickGGUF(entries []hfTreeEntry) (string, error) {
	var primary []hfTreeEntry
	for _, e := range entries {
		base := strings.ToLower(path.Base(e.Path))
		if strings.HasPrefix(base, "mmproj") {
			continue
		}
		primary = append(primary, e)
	}
	if len(primary) == 0 {
		if len(entries) == 1 {
			return entries[0].Path, nil
		}
		if len(entries) == 0 {
			return "", errHFNoGGUF
		}
		return "", errHFMultipleGGUF
	}
	if len(primary) == 1 {
		return primary[0].Path, nil
	}
	// Prefer largest non-projector quant when unspecified.
	best := primary[0]
	for _, e := range primary[1:] {
		if e.Size > best.Size {
			best = e
		}
	}
	return best.Path, nil
}

func hfResolveFile(ctx context.Context, ref HFRef) (HFRef, error) {
	if ref.File != "" {
		return ref, nil
	}
	entries, err := hfListGGUFFiles(ctx, ref)
	if err != nil {
		return ref, err
	}
	file, err := hfPickGGUF(entries)
	if err != nil {
		return ref, err
	}
	ref.File = file
	return ref, nil
}

func hfDownloadFile(ctx context.Context, ref HFRef, dest string, fn func(api.ProgressResponse)) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}

	downloadURL := ref.resolveURL(ref.File)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, downloadURL, nil)
	if err != nil {
		return err
	}
	hfAuthHeader(req)

	resp, err := hfHTTPClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("huggingface download %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	total := resp.ContentLength
	tmp, err := os.CreateTemp(filepath.Dir(dest), "hf-*.gguf")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()

	var completed int64
	buf := make([]byte, 32*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := tmp.Write(buf[:n]); werr != nil {
				return werr
			}
			completed += int64(n)
			if fn != nil {
				fn(api.ProgressResponse{
					Status:    "pulling from huggingface",
					Digest:    ref.File,
					Total:     total,
					Completed: completed,
				})
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, dest)
}

func pullFromHuggingFace(ctx context.Context, local typesmodel.Name, ref HFRef, deleteMap map[string]struct{}, fn func(api.ProgressResponse)) error {
	fn(api.ProgressResponse{Status: fmt.Sprintf("pulling from huggingface %s", ref.Repo)})

	resolved, err := hfResolveFile(ctx, ref)
	if err != nil {
		return err
	}
	ref = resolved

	tmpDir, err := os.MkdirTemp("", "zerollama-hf-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	dest := filepath.Join(tmpDir, path.Base(ref.File))
	if err := hfDownloadFile(ctx, ref, dest, fn); err != nil {
		return err
	}

	f, err := os.Open(dest)
	if err != nil {
		return err
	}
	digest, _ := GetSHA256Digest(f)
	_ = f.Close()

	files := map[string]string{dest: digest}
	if err := stageFilesToBlobs(files); err != nil {
		return err
	}

	config := &typesmodel.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: typesmodel.RootFS{
			Type: "layers",
		},
	}
	r := api.CreateRequest{}
	relFiles := map[string]string{filepath.Base(ref.File): digest}
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

	fn(api.ProgressResponse{Status: "success"})
	return nil
}

func tryPullHuggingFace(ctx context.Context, pullName string, localName typesmodel.Name, regOpts *registryOptions, deleteMap map[string]struct{}, fn func(api.ProgressResponse)) (bool, error) {
	var ref HFRef
	var local typesmodel.Name
	var err error

	switch {
	case regOpts != nil && strings.TrimSpace(regOpts.HFSource) != "":
		ref, err = ParseHFSource(regOpts.HFSource)
		if err != nil {
			return false, err
		}
		local = localName
		if !local.IsValid() {
			return false, fmt.Errorf("invalid local model name %q", pullName)
		}
	case IsHFPull(pullName):
		ref, err = ParseHFSource(pullName)
		if err != nil {
			return false, err
		}
		local = hfLocalName(ref)
	default:
		return false, nil
	}

	slog.Info("pulling from huggingface", "repo", ref.Repo, "file", ref.File, "local", local.DisplayShortest())
	if err := pullFromHuggingFace(ctx, local, ref, deleteMap, fn); err != nil {
		return true, err
	}
	return true, nil
}
