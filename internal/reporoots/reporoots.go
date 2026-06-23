// Package reporoots discovers zerollama checkout roots for locating local build artifacts.
//
// Why a shared helper: ANE probe and Flash-MoE llama-server discovery both walk
// executable dir, cwd→go.mod, and ZEROLLAMA_REPO — duplicating that logic caused
// drift when one path gained an extra parent walk. See docs/flash-moe.md.
package reporoots

import (
	"os"
	"path/filepath"
	"strings"
)

// SearchRoots returns candidate zerollama checkout roots, nearest first.
func SearchRoots() []string {
	return SearchRootsWithEnv("ZEROLLAMA_REPO", "OLLAMA_TRAINING_PYTHONPATH")
}

// SearchRootsWithEnv returns checkout roots using the given env keys as extra hints.
func SearchRootsWithEnv(extraEnvKeys ...string) []string {
	var roots []string
	seen := map[string]bool{}
	add := func(p string) {
		p = filepath.Clean(p)
		if p == "" || p == "." || seen[p] {
			return
		}
		seen[p] = true
		roots = append(roots, p)
	}

	if exe, err := os.Executable(); err == nil {
		add(filepath.Dir(exe))
		add(filepath.Join(filepath.Dir(exe), ".."))
		add(filepath.Join(filepath.Dir(exe), "../.."))
	}
	if wd, err := os.Getwd(); err == nil {
		for dir := wd; ; dir = filepath.Dir(dir) {
			add(dir)
			if st, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !st.IsDir() {
				break
			}
			if dir == filepath.Dir(dir) {
				break
			}
		}
	}
	for _, key := range extraEnvKeys {
		if p := strings.TrimSpace(os.Getenv(key)); p != "" {
			add(p)
		}
	}
	return roots
}
