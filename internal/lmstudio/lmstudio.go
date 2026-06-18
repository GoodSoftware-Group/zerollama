// Package lmstudio discovers models installed by LM Studio under the default
// user directory layout (~/.lmstudio/models/...).
package lmstudio

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

var (
	ggufShardName = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}\.gguf$`)
	stShardName   = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}\.safetensors$`)
	quantTag      = regexp.MustCompile(`(?i)(q\d+_[kmgs0-9]+|q\d+|fp16|bf16|f16|mxfp4|[0-9]+bit)`)
)

const remoteHost = "lmstudio"

// ImportHeadroomBytes is reserved on top of model size for MLX safetensors import
// (manifest layers, metadata, and temporary packing). Why 512 MiB: import creates
// new blob files under OLLAMA_MODELS; GGUF symlinks need no headroom.
const ImportHeadroomBytes = 512 << 20 // 512 MiB

// Entry describes one LM Studio model directory.
type Entry struct {
	Dir        string
	Root       string
	Name       string
	Format     string
	Size       int64
	Modified   time.Time
	WeightFile string // non-empty when the entry is one quant variant in a multi-GGUF dir
}

// ModelPath returns the concrete weight file to inspect/import when the entry
// maps to a single GGUF file. For sharded GGUF directories it returns the first
// shard, which is the path readers use to discover the shard set.
func (e Entry) ModelPath() string {
	if e.Format != "gguf" {
		return ""
	}
	if e.WeightFile != "" {
		return filepath.Join(e.Dir, e.WeightFile)
	}
	weights, ok := ggufWeightFiles(e.Dir)
	if !ok || len(weights) == 0 {
		return ""
	}
	return weights[0]
}

// ConfigPath returns the Hugging Face/MLX config.json path when present.
func (e Entry) ConfigPath() string {
	p := filepath.Join(e.Dir, "config.json")
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}

// Roots returns directories to scan for LM Studio models. If
// OLLAMA_LMSTUDIO_MODELS is set, only those roots are used (comma- or
// filepath.ListSeparator-separated). Otherwise the default is ~/.lmstudio/models
// when present, and on macOS also Library/Application Support/LM Studio/models.
func Roots() []string {
	seen := make(map[string]struct{})
	var out []string

	add := func(p string) {
		p = filepath.Clean(p)
		if p == "" || p == "." {
			return
		}
		if _, ok := seen[p]; ok {
			return
		}
		if st, err := os.Stat(p); err != nil || !st.IsDir() {
			return
		}
		seen[p] = struct{}{}
		out = append(out, p)
	}

	if raw := strings.TrimSpace(envconfig.Var("OLLAMA_LMSTUDIO_MODELS")); raw != "" {
		sep := ","
		if strings.Contains(raw, string(filepath.ListSeparator)) {
			sep = string(filepath.ListSeparator)
		}
		for _, p := range strings.Split(raw, sep) {
			add(strings.TrimSpace(p))
		}
		return out
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return out
	}
	add(filepath.Join(home, ".lmstudio", "models"))

	// Some installs use Application Support on macOS; only add if present.
	if runtime.GOOS == "darwin" {
		add(filepath.Join(home, "Library", "Application Support", "LM Studio", "models"))
	}

	return out
}

// List returns LM Studio model directories under [Roots].
func List() []Entry {
	var out []Entry
	for _, root := range Roots() {
		_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil || !d.IsDir() {
				return nil
			}
			rel, err := filepath.Rel(root, path)
			if err != nil || rel == "." {
				return nil
			}
			if len(strings.Split(rel, string(filepath.Separator))) < 2 {
				return nil
			}
			out = append(out, listEntriesForDir(root, path)...)
			return nil
		})
	}
	return out
}

func listEntriesForDir(root, path string) []Entry {
	format, weights, ok := dirWeightFiles(path)
	if !ok {
		return nil
	}

	size, modified := dirStats(path)
	if format == "gguf" && len(weights) > 1 && !allGGUFShards(weights) {
		var out []Entry
		for _, w := range weights {
			name := suggestedNameForWeight(root, path, filepath.Base(w))
			if name == "" {
				continue
			}
			out = append(out, Entry{
				Dir:        path,
				Root:       root,
				Name:       name,
				Format:     format,
				Size:       fileSize(w),
				Modified:   modified,
				WeightFile: filepath.Base(w),
			})
		}
		return out
	}

	name := SuggestedName(root, path)
	if name == "" {
		return nil
	}
	return []Entry{{
		Dir:      path,
		Root:     root,
		Name:     name,
		Format:   format,
		Size:     size,
		Modified: modified,
	}}
}

// RemoteHost is the ListModelResponse remote_host value for LM Studio entries.
func RemoteHost() string {
	return remoteHost
}

// SuggestedName returns a stable pull/run name for a model directory.
func SuggestedName(root, dir string) string {
	format, weights, ok := dirWeightFiles(dir)
	if !ok {
		return ""
	}
	if format == "gguf" && len(weights) == 1 {
		return suggestedNameForWeight(root, dir, filepath.Base(weights[0]))
	}
	return suggestedNameForWeight(root, dir, "")
}

func suggestedNameForWeight(root, dir, weightBase string) string {
	rel, err := filepath.Rel(root, dir)
	if err != nil || rel == "." {
		return ""
	}
	parts := strings.Split(rel, string(filepath.Separator))
	if len(parts) < 2 {
		return ""
	}

	publisher := strings.ToLower(parts[0])
	folder := parts[1]
	base := folder
	for _, suffix := range []string{
		"-GGUF", "-gguf",
		"-MLX-8bit", "-mlx-8bit",
		"-Q8-mlx", "-q8-mlx",
	} {
		if strings.HasSuffix(base, suffix) {
			base = strings.TrimSuffix(base, suffix)
			break
		}
	}
	base = strings.ToLower(base)

	tag := "latest"
	if weightBase != "" {
		if q := quantTag.FindString(weightBase); q != "" {
			tag = strings.ToLower(q)
		}
	} else if q := detectQuantTag(dir); q != "" {
		tag = q
	}

	return publisher + "/" + base + ":" + tag
}

// MatchSelection returns the LM Studio directory and optional single GGUF weight
// basename to import for the requested model name.
func MatchSelection(n model.Name) (dir string, weightFile string, ok bool) {
	if !n.IsValid() {
		return "", "", false
	}

	want := strings.ToLower(strings.TrimSpace(n.DisplayShortest()))
	var exactMatches []Entry
	for _, e := range List() {
		if strings.EqualFold(e.Name, want) {
			exactMatches = append(exactMatches, e)
		}
	}
	if len(exactMatches) == 1 {
		return exactMatches[0].Dir, exactMatches[0].WeightFile, true
	}
	if len(exactMatches) > 1 {
		return "", "", false
	}

	dir, ok = matchFuzzyDir(n)
	if !ok {
		return "", "", false
	}

	format, weights, ok := dirWeightFiles(dir)
	if !ok || format != "gguf" || len(weights) <= 1 || allGGUFShards(weights) {
		return dir, "", true
	}

	tag := strings.ToLower(strings.TrimSpace(n.Tag))
	for _, tok := range tagTokens(tag) {
		for _, w := range weights {
			base := strings.ToLower(filepath.Base(w))
			if strings.Contains(base, tok) {
				return dir, filepath.Base(w), true
			}
		}
	}
	return "", "", false
}

// MatchDir returns a model directory under LM Studio roots whose name and tag
// heuristically match the requested Ollama model. The second return is false if
// no unambiguous usable directory was found.
func MatchDir(n model.Name) (string, bool) {
	dir, _, ok := MatchSelection(n)
	return dir, ok
}

func matchFuzzyDir(n model.Name) (string, bool) {
	if !n.IsValid() {
		return "", false
	}

	var bestPath string
	var bestScore int
	var tie bool

	for _, root := range Roots() {
		_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil || !d.IsDir() {
				return nil
			}
			rel, err := filepath.Rel(root, path)
			if err != nil || rel == "." {
				return nil
			}
			if len(strings.Split(rel, string(filepath.Separator))) < 2 {
				return nil
			}
			if _, ok := dirModelFormat(path); !ok {
				return nil
			}
			score := scorePath(path, n)
			if score <= 0 {
				return nil
			}
			switch {
			case score > bestScore:
				bestScore = score
				bestPath = path
				tie = false
			case score == bestScore && bestPath != "" && path != bestPath:
				tie = true
			}
			return nil
		})
	}

	if bestPath == "" || tie || bestScore < 2 {
		return "", false
	}

	format, weights, ok := dirWeightFiles(bestPath)
	if ok && format == "gguf" && len(weights) > 1 && !allGGUFShards(weights) {
		tag := strings.ToLower(strings.TrimSpace(n.Tag))
		if tag == "" || tag == "latest" {
			return "", false
		}
		matched := false
		for _, tok := range tagTokens(tag) {
			for _, w := range weights {
				if strings.Contains(strings.ToLower(filepath.Base(w)), tok) {
					matched = true
					break
				}
			}
		}
		if !matched {
			return "", false
		}
	}

	return bestPath, true
}

func dirModelFormat(dir string) (string, bool) {
	format, _, ok := dirWeightFiles(dir)
	return format, ok
}

func dirWeightFiles(dir string) (format string, weights []string, ok bool) {
	if weights, ok = ggufWeightFiles(dir); ok {
		return "gguf", weights, true
	}
	if weights, ok = safetensorsWeightFiles(dir); ok {
		return "safetensors", weights, true
	}
	return "", nil, false
}

func dirLooksLikeLMStudioModel(dir string) bool {
	_, ok := dirModelFormat(dir)
	return ok
}

func ggufWeightFiles(dir string) ([]string, bool) {
	ggufs, err := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if err != nil || len(ggufs) == 0 {
		return nil, false
	}

	var weights []string
	for _, g := range ggufs {
		base := strings.ToLower(filepath.Base(g))
		if strings.HasPrefix(base, "mmproj") {
			continue
		}
		weights = append(weights, g)
	}
	if len(weights) == 0 {
		return nil, false
	}
	if len(weights) == 1 || allGGUFShards(weights) {
		return weights, true
	}
	// Multiple quant variants in one directory (e.g. Q4_K_M and Q8_0).
	return weights, true
}

func allGGUFShards(weights []string) bool {
	if len(weights) <= 1 {
		return false
	}
	for _, p := range weights {
		if !ggufShardName.MatchString(filepath.Base(p)) {
			return false
		}
	}
	return true
}

func safetensorsWeightFiles(dir string) ([]string, bool) {
	st, err := filepath.Glob(filepath.Join(dir, "model*.safetensors"))
	if err != nil || len(st) == 0 {
		if st, err = filepath.Glob(filepath.Join(dir, "consolidated*.safetensors")); err != nil || len(st) == 0 {
			return nil, false
		}
	}

	var weights []string
	for _, p := range st {
		base := strings.ToLower(filepath.Base(p))
		if base == "model.safetensors.index.json" {
			continue
		}
		weights = append(weights, p)
	}
	if len(weights) == 0 {
		return nil, false
	}
	if len(weights) == 1 {
		return weights, true
	}
	for _, p := range weights {
		if !stShardName.MatchString(filepath.Base(p)) {
			return nil, false
		}
	}
	return weights, true
}

func dirHasGGUFWeights(dir string) bool {
	_, ok := ggufWeightFiles(dir)
	return ok
}

func dirHasSafetensorsWeights(dir string) bool {
	_, ok := safetensorsWeightFiles(dir)
	return ok
}

func fileSize(path string) int64 {
	st, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return st.Size()
}

func detectQuantTag(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") &&
			!strings.HasSuffix(strings.ToLower(name), ".safetensors") {
			continue
		}
		if m := quantTag.FindString(name); m != "" {
			return strings.ToLower(m)
		}
	}
	return ""
}

func dirStats(dir string) (size int64, modified time.Time) {
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		size += info.Size()
		if info.ModTime().After(modified) {
			modified = info.ModTime()
		}
		return nil
	})
	return size, modified
}

func scorePath(path string, n model.Name) int {
	lower := strings.ToLower(path)
	modelPart := strings.ToLower(strings.TrimSpace(n.Model))
	tag := strings.ToLower(strings.TrimSpace(n.Tag))

	score := 0
	for _, tok := range modelTokens(modelPart) {
		if tok == "" {
			continue
		}
		if strings.Contains(lower, tok) {
			score += 2
		}
	}

	if tag != "" && tag != "latest" {
		matched := false
		for _, tok := range tagTokens(tag) {
			if tok == "" {
				continue
			}
			if strings.Contains(lower, tok) {
				matched = true
				break
			}
		}
		if !matched {
			return 0
		}
		score += 2
	}

	return score
}

func modelTokens(s string) []string {
	seen := make(map[string]struct{})
	var out []string
	add := func(t string) {
		t = strings.TrimSpace(t)
		if t == "" {
			return
		}
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}

	add(s)
	add(strings.ReplaceAll(s, ".", ""))

	// qwen3.5 -> qwen-3-5
	if strings.Contains(s, ".") {
		parts := strings.Split(s, ".")
		if len(parts) >= 2 {
			add(strings.Join(parts, "-"))
		}
	}

	// gemma4 -> gemma-4
	for i := 0; i < len(s)-1; i++ {
		if s[i] >= 'a' && s[i] <= 'z' && s[i+1] >= '0' && s[i+1] <= '9' {
			add(s[:i+1] + "-" + s[i+1:])
			break
		}
	}

	return out
}

func tagTokens(tag string) []string {
	seen := make(map[string]struct{})
	var out []string
	add := func(t string) {
		t = strings.TrimSpace(strings.ToLower(t))
		if t == "" {
			return
		}
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}

	add(tag)
	add(strings.ReplaceAll(tag, "_", "-"))

	// 31b vs 31-b
	if strings.HasSuffix(tag, "b") && len(tag) > 1 {
		num := strings.TrimSuffix(tag, "b")
		if num != "" {
			add(num + "b")
			add(num + "-b")
		}
	}

	return out
}

// dirIsMLXSafetensors reports whether dir uses the MLX native import path:
// config.json present + safetensors weight files. Mirrors x/create.IsSafetensorsModelDir
// without importing that package (import cycle: server → x/create/client → x/create).
func dirIsMLXSafetensors(dir string) bool {
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return false
	}
	return dirHasSafetensorsWeights(dir)
}

// ImportCopyBytes returns additional disk space required to import e into
// OLLAMA_MODELS. GGUF imports symlink existing files (near-zero copy); MLX
// safetensors imports repack tensors into new blobs (~full model size).
// Legacy safetensors without config.json are also symlinked, so they return 0.
func ImportCopyBytes(e Entry) int64 {
	if e.Format != "safetensors" {
		return 0
	}
	if !dirIsMLXSafetensors(e.Dir) {
		return 0
	}
	return e.Size + ImportHeadroomBytes
}

// DirImportCopyBytes estimates copy bytes for an MLX safetensors model directory.
func DirImportCopyBytes(dir string) int64 {
	if !dirIsMLXSafetensors(dir) {
		return 0
	}
	size, _ := dirStats(dir)
	return size + ImportHeadroomBytes
}

// FreeBytesAtModels returns available bytes on the filesystem holding OLLAMA_MODELS.
// Why resolveExistingPath: OLLAMA_MODELS may not exist yet on first run; walk up to
// find a mount point for statfs/GetDiskFreeSpaceEx.
func FreeBytesAtModels() (int64, error) {
	return modelsFreeBytes(envconfig.Models())
}

// modelsFreeBytes is swappable in tests.
var modelsFreeBytes = freeBytes

// SetModelsFreeBytesHook replaces free-space lookup for tests. Returns restore.
func SetModelsFreeBytesHook(fn func(string) (int64, error)) func() {
	prev := modelsFreeBytes
	if fn == nil {
		modelsFreeBytes = freeBytes
	} else {
		modelsFreeBytes = fn
	}
	return func() { modelsFreeBytes = prev }
}

// HasDiskForImport reports whether there is enough free space to import e.
func HasDiskForImport(e Entry) (ok bool, free int64, need int64, err error) {
	need = ImportCopyBytes(e)
	if need == 0 {
		return true, 0, 0, nil
	}
	free, err = FreeBytesAtModels()
	if err != nil {
		return false, 0, need, err
	}
	return free >= need, free, need, nil
}

// HasDiskForDirImport reports whether dir (safetensors tree) can be imported.
func HasDiskForDirImport(dir string) (ok bool, free int64, need int64, err error) {
	need = DirImportCopyBytes(dir)
	if need == 0 {
		return true, 0, 0, nil
	}
	free, err = FreeBytesAtModels()
	if err != nil {
		return false, 0, need, err
	}
	return free >= need, free, need, nil
}

func freeBytes(path string) (int64, error) {
	path, err := resolveExistingPath(path)
	if err != nil {
		return 0, err
	}
	return freeBytesOS(path)
}

func resolveExistingPath(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("empty path")
	}
	st, err := os.Stat(path)
	if err == nil {
		if st.IsDir() {
			return path, nil
		}
		return filepath.Dir(path), nil
	}
	if !os.IsNotExist(err) {
		return "", err
	}
	parent := filepath.Dir(path)
	if parent == path {
		return "", fmt.Errorf("path does not exist: %s", path)
	}
	return resolveExistingPath(parent)
}
