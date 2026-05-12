// Package lmstudio discovers GGUF models installed by LM Studio under the
// default user directory layout (~/.lmstudio/models/...).
package lmstudio

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

var shardName = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}\.gguf$`)

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

// MatchDir returns a model directory under LM Studio roots whose name and tag
// heuristically match the requested Ollama model. The second return is false if
// no unambiguous usable directory was found (including sharded multi-file GGUF
// layouts, which are not auto-imported).
func MatchDir(n model.Name) (string, bool) {
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
			// Expect at least publisher/model-name (two components under root).
			if len(strings.Split(rel, string(filepath.Separator))) < 2 {
				return nil
			}
			if !dirLooksLikeLMStudioModel(path) {
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
	return bestPath, true
}

func dirLooksLikeLMStudioModel(dir string) bool {
	ggufs, err := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if err != nil || len(ggufs) == 0 {
		return false
	}

	var nonProj []string
	for _, g := range ggufs {
		base := strings.ToLower(filepath.Base(g))
		if strings.HasPrefix(base, "mmproj") {
			continue
		}
		nonProj = append(nonProj, g)
	}
	if len(nonProj) == 0 {
		return false
	}
	if len(nonProj) == 1 {
		return true
	}
	// Multiple weight files: only allow if not a multi-part shard layout.
	for _, p := range nonProj {
		if !shardName.MatchString(filepath.Base(p)) {
			return false
		}
	}
	return false
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
